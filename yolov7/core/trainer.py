import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss, ComputeLossOTA
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, hyp, opt, device, tb_writer=None):
        self.hyp = hyp
        self.opt = opt
        self.device = device
        self.tb_writer = tb_writer

        logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        self.save_dir = Path(opt.save_dir)
        self.epochs = opt.epochs
        self.batch_size = opt.batch_size
        self.total_batch_size = opt.total_batch_size
        self.weights = opt.weights
        self.rank = opt.global_rank
        self.freeze = opt.freeze     


    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.i, (imgs, targets, paths, _) in self.pbar:
            self.before_iter(imgs, targets, paths)
            self.train_one_iter(imgs, targets, paths)
            self.after_iter(imgs, targets, paths)

    def train_one_iter(self, imgs, targets, paths):
        self.ni = self.i + self.nb * self.epoch  # number integrated batches (since train start)
        imgs = imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if self.ni <= self.nw:
            xi = [0, self.nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(self.ni, xi, [1, self.nbs / self.total_batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(self.ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(self.ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        # Multi-scale
        if self.opt.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        # Forward
        with amp.autocast(enabled=self.cuda):
            pred = self.model(imgs)  # forward
            if 'loss_ota' not in self.hyp or self.hyp['loss_ota'] == 1:
                self.loss, self.loss_items = self.compute_loss_ota(pred, targets.to(self.device), imgs)  # loss scaled by batch_size
            else:
                self.loss, self.loss_items = self.compute_loss(pred, targets.to(self.device))  # loss scaled by batch_size
            if self.rank != -1:
                self.loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
            if self.opt.quad:
                self.loss *= 4.

        # Backward
        self.scaler.scale(self.loss).backward()

        # Optimize
        if self.ni % accumulate == 0:
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

    def init_dirs(self):
        # Directories
        self.wdir = self.save_dir / 'weights'
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.last = self.wdir / 'last.pt'
        self.best = self.wdir / 'best.pt'
        self.results_file = self.save_dir / 'results.txt'

        # Save run settings
        with open(self.save_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        with open(self.save_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)
        
    def setup_optimizer(self):# Optimizer
        self.nbs = 64  # nominal batch size
        accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * accumulate / self.nbs  # scale weight_decay
        logger.info(f"Scaled weight_decay = {self.hyp['weight_decay']}")

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):           
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imc'):
                if hasattr(v.imc, 'implicit'):           
                    pg0.append(v.imc.implicit)
                else:
                    for iv in v.imc:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imb'):
                if hasattr(v.imb, 'implicit'):           
                    pg0.append(v.imb.implicit)
                else:
                    for iv in v.imb:
                        pg0.append(iv.implicit)
            if hasattr(v, 'imo'):
                if hasattr(v.imo, 'implicit'):           
                    pg0.append(v.imo.implicit)
                else:
                    for iv in v.imo:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):           
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)
            if hasattr(v, 'attn'):
                if hasattr(v.attn, 'logit_scale'):   
                    pg0.append(v.attn.logit_scale)
                if hasattr(v.attn, 'q_bias'):   
                    pg0.append(v.attn.q_bias)
                if hasattr(v.attn, 'v_bias'):  
                    pg0.append(v.attn.v_bias)
                if hasattr(v.attn, 'relative_position_bias_table'):  
                    pg0.append(v.attn.relative_position_bias_table)
            if hasattr(v, 'rbr_dense'):
                if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                    pg0.append(v.rbr_dense.weight_rbr_origin)
                if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                    pg0.append(v.rbr_dense.weight_rbr_avg_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                    pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
                if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                    pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
                if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                    pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
                if hasattr(v.rbr_dense, 'vector'):   
                    pg0.append(v.rbr_dense.vector)

        if self.opt.adam:
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2


    def before_train(self):
        self.init_dirs()
        # Configure
        self.plots = not self.opt.evolve  # create plots
        self.cuda = self.device.type != 'cpu'
        init_seeds(2 + self.rank)
        with open(self.opt.data) as f:
            self.data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.is_coco = self.opt.data.endswith('coco.yaml')

        # Logging- Doing this before checking the dataset. Might update data_dict
        loggers = {'wandb': None}  # loggers dict
        if self.rank in [-1, 0]:
            self.opt.hyp = self.hyp  # add hyperparameters
            run_id = torch.load(self.weights, map_location=self.device).get('wandb_id') if self.weights.endswith('.pt') and os.path.isfile(self.weights) else None
            self.wandb_logger = WandbLogger(self.opt, Path(self.opt.save_dir).stem, run_id, self.data_dict)
            loggers['wandb'] = self.wandb_logger.wandb
            self.data_dict = self.wandb_logger.data_dict
            if self.wandb_logger.wandb:
                weights, epochs, hyp = self.opt.weights, self.opt.epochs, self.opt.hyp  # WandbLogger might update weights, epochs if resuming


        self.nc = 1 if self.opt.single_cls else int(self.data_dict['nc'])  # number of classes
        names = ['item'] if self.opt.single_cls and len(self.data_dict['names']) != 1 else self.data_dict['names']  # class names
        assert len(names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(names), self.nc, self.opt.data)  # check

        # Model
        pretrained = self.weights.endswith('.pt')
        if pretrained:
            with torch_distributed_zero_first(self.rank):
                attempt_download(self.weights)  # download if not found locally
            ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            self.model = Model(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
            exclude = ['anchor'] if (self.opt.cfg or self.hyp.get('anchors')) and not self.opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), self.weights))  # report
        else:
            self.model = Model(self.opt.cfg, ch=3, nc=self.nc, anchors=self.hyp.get('anchors')).to(self.device)  # create
        with torch_distributed_zero_first(self.rank):
            check_dataset(self.data_dict)  # check
        train_path = self.data_dict['train']
        test_path = self.data_dict['val']

        # Freeze
        self.freeze = [f'model.{x}.' for x in (self.freeze if len(self.freeze) > 1 else range(self.freeze[0]))]  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in self.freeze):
                print('freezing %s' % k)
                v.requires_grad = False

        self.setup_optimizer()

        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        if self.opt.linear_lr:
            self.lf = lambda x: (1 - x / (self.epochs - 1)) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        else:
            self.lf = one_cycle(1, self.hyp['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)

        # EMA
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None

        # Resume
        self.start_epoch, self.best_fitness = 0, 0.0
        if pretrained:
            # Optimizer
            if ckpt['optimizer'] is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.best_fitness = ckpt['best_fitness']

            # EMA
            if self.ema and ckpt.get('ema'):
                self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                self.ema.updates = ckpt['updates']

            # Results
            if ckpt.get('training_results') is not None:
                self.results_file.write_text(ckpt['training_results'])  # write results.txt

            # Epochs
            self.start_epoch = ckpt['epoch'] + 1
            if self.opt.resume:
                assert self.start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (self.weights, epochs)
            if epochs < self.start_epoch:
                logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                            (self.weights, ckpt['epoch'], epochs))
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt, state_dict

        # Image sizes
        self.gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        nl = self.model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            logger.info('Using SyncBatchNorm()')

        # Trainloader
        self.dataloader, self.dataset = create_dataloader(train_path, self.imgsz, self.batch_size, self.gs, self.opt,
                                                hyp=self.hyp, augment=True, cache=self.opt.cache_images, rect=self.opt.rect, rank=self.rank,
                                                world_size=self.opt.world_size, workers=self.opt.workers,
                                                image_weights=self.opt.image_weights, quad=self.opt.quad, prefix=colorstr('train: '))
        mlc = np.concatenate(self.dataset.labels, 0)[:, 0].max()  # max label class
        self.nb = len(self.dataloader)  # number of batches
        assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, self.nc, self.opt.data, self.nc - 1)

        # Process 0
        if self.rank in [-1, 0]:
            self.testloader = create_dataloader(test_path, self.imgsz_test, self.batch_size * 2, self.gs, self.opt,  # testloader
                                        hyp=self.hyp, cache=self.opt.cache_images and not self.opt.notest, rect=True, rank=-1,
                                        world_size=self.opt.world_size, workers=self.opt.workers,
                                        pad=0.5, prefix=colorstr('val: '))[0]

            if not self.opt.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(self.device))
                if self.plots:
                    #plot_labels(labels, names, save_dir, loggers)
                    if self.tb_writer:
                        self.tb_writer.add_histogram('classes', c, 0)

                # Anchors
                if not self.opt.noautoanchor:
                    check_anchors(self.dataset, model=self.model, thr=self.hyp['anchor_t'], imgsz=self.imgsz)
                self.model.half().float()  # pre-reduce anchor precision

        # DDP mode
        if self.cuda and self.rank != -1:
            self.model = DDP(self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank,
                        # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                        find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in self.model.modules()))

        # Model parameters
        self.hyp['box'] *= 3. / nl  # scale to layers
        self.hyp['cls'] *= self.nc / 80. * 3. / nl  # scale to classes and layers
        self.hyp['obj'] *= (self.imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        self.hyp['label_smoothing'] = self.opt.label_smoothing
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device) * self.nc  # attach class weights
        self.model.names = names

        # Start training
        self.t0 = time.time()
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        self.maps = np.zeros(self.nc)  # mAP per class
        self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        self.compute_loss_ota = ComputeLossOTA(self.model)  # init loss class
        self.compute_loss = ComputeLoss(self.model)  # init loss class
        logger.info(f'Image sizes {self.imgsz} train, {self.imgsz_test} test\n'
                    f'Using {self.dataloader.num_workers} dataloader workers\n'
                    f'Logging results to {self.save_dir}\n'
                    f'Starting training for {self.epochs} epochs...')
        torch.save(self.model, self.wdir / 'init.pt')

    def after_train(self):
        pass

    def before_epoch(self):
        self.model.train()

        # Update image weights (optional)
        if self.opt.image_weights:
            # Generate indices
            if self.rank in [-1, 0]:
                cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.nc  # class weights
                iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
                self.dataset.indices = random.choices(range(self.dataset.n), weights=iw, k=self.dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if self.rank != -1:
                indices = (torch.tensor(self.dataset.indices) if self.rank == 0 else torch.zeros(self.dataset.n)).int()
                dist.broadcast(indices, 0)
                if self.rank != 0:
                    self.dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        self.mloss = torch.zeros(4, device=self.device)  # mean losses
        if self.rank != -1:
            self.dataloader.sampler.set_epoch(self.epoch)
        self.pbar = enumerate(self.dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if self.rank in [-1, 0]:
            self.pbar = tqdm(self.pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()

    def after_epoch(self):
        # Scheduler
        lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
        self.scheduler.step()

        # DDP process 0 or single-GPU
        if self.rank in [-1, 0]:
            # mAP
            self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = self.epoch + 1 == self.epochs
            if not self.opt.notest or final_epoch:  # Calculate mAP
                self.wandb_logger.current_epoch = self.epoch + 1
                results, maps, times = test.test(self.data_dict,
                                                 batch_size=self.batch_size * 2,
                                                 imgsz=self.imgsz_test,
                                                 model=self.ema.ema,
                                                 single_cls=self.opt.single_cls,
                                                 dataloader=self.testloader,
                                                 save_dir=self.save_dir,
                                                 verbose=self.nc < 50 and final_epoch,
                                                 plots=self.plots and final_epoch,
                                                 wandb_logger=self.wandb_logger,
                                                 compute_loss=self.compute_loss,
                                                 is_coco=self.is_coco,
                                                 v5_metric=self.opt.v5_metric)

            # Write
            with open(self.results_file, 'a') as f:
                f.write(self.s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(self.opt.name) and self.opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (self.results_file, self.opt.bucket, self.opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(self.mloss[:-1]) + list(results) + lr, tags):
                if self.tb_writer:
                    self.tb_writer.add_scalar(tag, x, self.epoch)  # tensorboard
                if self.wandb_logger.wandb:
                    self.wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi
            self.wandb_logger.end_epoch(best_result=self.best_fitness == fi)

            # Save model
            if (not self.opt.nosave) or (final_epoch and not self.opt.evolve):  # if save
                ckpt = {'epoch': self.epoch,
                        'best_fitness': self.best_fitness,
                        'training_results': self.results_file.read_text(),
                        'model': deepcopy(self.model.module if is_parallel(self.model) else self.model).half(),
                        'ema': deepcopy(self.ema.ema).half(),
                        'updates': self.ema.updates,
                        'optimizer': self.optimizer.state_dict(),
                        'wandb_id': self.wandb_logger.wandb_run.id if self.wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                if (self.best_fitness == fi) and (self.epoch >= 200):
                    torch.save(ckpt, self.wdir / 'best_{:03d}.pt'.format(self.epoch))
                if self.epoch == 0:
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(self.epoch))
                elif ((self.epoch+1) % 25) == 0:
                    torch.save(ckpt, self.self.wdir / 'epoch_{:03d}.pt'.format(self.epoch))
                elif self.epoch >= (self.epochs-5):
                    torch.save(ckpt, self.wdir / 'epoch_{:03d}.pt'.format(self.epoch))
                if self.wandb_logger.wandb:
                    if ((self.epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                        self.wandb_logger.log_model(
                            self.last.parent, self.opt, self.epoch, fi, best_model=self.best_fitness == fi)
                del ckpt

    def before_iter(self, imgs, targets, paths):
        pass

    def after_iter(self, imgs, targets, paths):
        # Print
        if self.rank in [-1, 0]:
            self.mloss = (self.mloss * self.i + self.loss_items) / (self.i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            self.s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (self.epoch, self.epochs - 1), mem, *self.mloss, targets.shape[0], imgs.shape[-1])
            self.pbar.set_description(self.s)

            # Plot
            if self.plots and self.ni < 10:
                f = self.save_dir / f'train_batch{self.ni}.jpg'  # filename
                Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                # if tb_writer:
                #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
            elif self.plots and self.ni == 10 and self.wandb_logger.wandb:
                self.wandb_logger.log({"Mosaics": [self.wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                self.save_dir.glob('train*.jpg') if x.exists()]})


    def resume_train(self, model):
        model = 1

        return model

    def evaluate_and_save_model(self):
        pass

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        pass
