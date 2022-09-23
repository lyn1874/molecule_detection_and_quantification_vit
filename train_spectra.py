"""
Created on 16:30 at 17/11/2021
@author: bo
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import pickle
import wandb
from torch.utils.data import random_split, DataLoader
import data.prepare_sers_data as psd
import models.spectra_model as spectra_model
import models.resnet as resnet_model
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AMPType
import configs.common as config


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argsort(memory_available)


def global_batch_size_calc(batch_size, num_gpus, parallel_strategy):
    if parallel_strategy == "dp":
        global_batch_size = batch_size
    elif parallel_strategy == "ddp":
        global_batch_size = num_gpus * batch_size if num_gpus > 0 else batch_size
    else:
        global_batch_size = batch_size
    return global_batch_size


def get_lr_schedule(lr_init, epochs, num_global_iter_per_epoch, warmup_epochs=10):
    final_lr = 1e-5
    warmup_lr_schedule = np.linspace(0.0, lr_init, num_global_iter_per_epoch * warmup_epochs)
    iters = np.arange(num_global_iter_per_epoch * (epochs - warmup_epochs))
    cosine_lr_schedule = np.array([
        final_lr + 0.5 * (lr_init - final_lr) *
        (1 + math.cos(math.pi * t / (num_global_iter_per_epoch * (epochs - warmup_epochs))))
        for t in iters
    ])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    return lr_schedule


class XceptionLightning(LightningModule):
    def __init__(self, num_classes, batch_size=512, num_samples=45000, warmup_epochs=10, num_gpus=1, enc_lr=0.03,
                 quantification=False, detection=True, 
                 concentration_float=0, wavenumber=110, strategy="dp", epochs=50, reduce_channel_first=False,
                 model_type="xception", data_input_channel=1, dataset="SIMU_TYPE_12"):
        super().__init__()
        self.save_hyperparameters()
        if strategy == "test":
            within_dropout=False 
        else:
            within_dropout=True
        if model_type == "xception":
            self.model = spectra_model.XceptionCls(wavenumber, num_class=num_classes, stem_kernel=21,
                                                      depth=128, stem_max_dim=64,
                                                      within_dropout=within_dropout, quantification=quantification, detection=detection,
                                                      reduce_channel_first=reduce_channel_first, data_input_channel=data_input_channel)
        elif model_type == "unified_cnn":
            self.model = spectra_model.UnifiedCNN([data_input_channel, wavenumber], num_classes=num_classes, block_type="lenet",
                                                  quantification=quantification, detection=detection)
        elif model_type == "resnet":
            self.model = resnet_model.get_default_resnet_model(wavenumber, n_class=num_classes, quantification=quantification, 
                                                               detection=detection)

        global_batch_size = global_batch_size_calc(batch_size, num_gpus, strategy)
        self.train_iters_per_epoch = num_samples // global_batch_size
        self.lr_schedule = get_lr_schedule(enc_lr, epochs, self.train_iters_per_epoch, warmup_epochs=warmup_epochs)

    def configure_optimizers(self):
        params = self.parameters()
        if self.hparams.model_type == "unified_cnn":
            if self.hparams.quantification is True and "SIMU" in self.hparams.dataset:
                weight_decay = 1e-3
            else:
                weight_decay = 1e-4 
        elif self.hparams.model_type == "xception":
            weight_decay = 5e-4
        elif self.hparams.model_type == "resnet":
            weight_decay = 1e-3
            if self.hparams.dataset == "PA" and self.hparams.quantification == False:
                weight_decay=5e-2
        optimizer = torch.optim.SGD(params, self.hparams.enc_lr, momentum=0.9, weight_decay=weight_decay)
        for name, v in self.named_parameters():
            if not v.requires_grad:
                print(name, "------does not require gradient")
        return optimizer

    def optimizer_step(self, epoch=None,
                     batch_idx=None,
                     optimizer=None,
                     optimizer_idx=None,
                     optimizer_closure=None,
                     on_tpu=None,
                     using_native_amp=None,
                     using_lbfgs=None):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]
            # self.log("learning_rate_encoder", self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
            
    def get_quantification_performance(self, pred_quan, gt_quan, eval=False):
        if 0 < self.hparams.concentration_float < 1.0:
            pred_quan = torch.clamp(pred_quan, np.log(self.hparams.concentration_float))
        loss_quan = nn.MSELoss(reduction='sum')(pred_quan, gt_quan)
        quan_act = pred_quan.exp()
        y_quan_act = gt_quan.exp()
        top = (quan_act - y_quan_act).pow(2).sum()
        bottom = (y_quan_act - y_quan_act.mean()).pow(2).sum()
        accu_quan = 1.0 - top.div(bottom)
        if eval:
            accu_quan_rmae = ((quan_act - y_quan_act).abs().div(y_quan_act)).sum()
            accu_quan_rae = ((quan_act - y_quan_act).pow(2).div(y_quan_act.pow(2))).sum()                
            top = (pred_quan - gt_quan).pow(2).sum()
            bottom = (gt_quan - gt_quan.mean()).pow(2).sum()
            accu_quan_subset_rsquare = 1.0 - top.div(bottom)                
        else:
            accu_quan_rmae = accu_quan
            accu_quan_subset_rsquare = accu_quan 
            accu_quan_rae = accu_quan 
        return loss_quan, accu_quan, accu_quan_subset_rsquare, accu_quan_rmae, accu_quan_rae

    def training_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.model(x.squeeze(1))
        if self.hparams.detection:
            loss = nn.CrossEntropyLoss()(pred, y.long())
        else:
            loss = 0.0
        if self.hparams.quantification:
            loss_quan, r_square, _, _, _ = self.get_quantification_performance(quan, y_conc)
            loss += loss_quan.div(len(x))
        if self.hparams.detection:
            accu = (pred.argmax(axis=-1) == y).sum().div(len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.model(x.squeeze(1))
        if self.hparams.detection:
            loss = nn.CrossEntropyLoss()(pred, y.long())
        else:
            loss = 0.0
        if self.hparams.quantification:
            loss_quan, accu_quan, rsquare_subset, accu_quan_rmae, accu_quan_rae = self.get_quantification_performance(quan, y_conc, eval=True)
            loss += loss_quan.div(len(x))

        if self.hparams.detection:
            accu = (pred.argmax(axis=-1) == y).sum().div(len(x))
            if self.hparams.dataset == "PA" and self.hparams.model_type == "resnet":
                self.log("validation_accuracy", accu, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.model.forward_test(x.squeeze(1))
        if self.hparams.detection:
            loss = nn.CrossEntropyLoss()(pred, y.long())
            accu = (pred.argmax(axis=-1) == y).sum().div(len(x))
        else:
            loss = 0.0
        # if self.hparams.quantification:
        #     loss_quan, r_square, r_square_subset, accu_quan_rmae, accu_quan_rae = self.get_quantification_performance(quan, y_conc, eval=True)
        return loss 
    

def get_monitors_for_quantification_exp(model_dir):
    monitor = ["validation_quantification_rmae"]
    check_point_g = []
    for s_monitor in monitor:
        if "loss" in s_monitor or "rae" in s_monitor or "rmae" in s_monitor:
            mode = "min"
        elif "rsquare" in s_monitor:
            mode="max"
        checkpoint_callback = ModelCheckpoint(monitor=s_monitor,
                                                dirpath=model_dir,
                                                filename="model-{epoch:02d}-{%s:.2f}" % s_monitor,
                                                save_top_k=3,
                                                mode=mode)
        check_point_g.append(checkpoint_callback)
    return check_point_g


def get_monitors_for_detection_exp(model_dir):
    monitor = ["validation_accuracy"]
    check_point_g = []
    for s_monitor in monitor:
        if "accuracy" in s_monitor or "f1_score" in s_monitor or "false_positive" in s_monitor:
            mode="max"
        else:
            mode="min"
        checkpoint_callback = ModelCheckpoint(monitor=s_monitor,
                                              dirpath=model_dir,
                                                filename="model-{epoch:02d}-{%s:.2f}" % s_monitor,
                                                save_top_k=3,
                                                mode=mode)
        check_point_g.append(checkpoint_callback)
    return check_point_g            
        

def train(args, model_dir, data_dir="../image_dataset/"):
    create_dir(model_dir)
    np.random.seed(args.augment_seed_use)
    data_obj_loader = psd.ReadSERSData(args.dataset,
                                       target_shape=[args.target_h, args.target_w],
                                       top_selection_method=args.top_selection_method,
                                       percentage=args.percentage,
                                       path_mom=data_dir, 
                                       avg=args.avg_spectra, 
                                       quantification=args.quantification,
                                       detection=args.detection,
                                       leave_index=args.leave_index,
                                       leave_method=args.leave_method,
                                       concentration_float=args.concentration_float)
    tr_data, val_data, tt_data, num_samples, imshape, num_class, data_input_channel = data_obj_loader.forward()
    if len(tr_data) / args.batch_size < 1:
        args.batch_size = len(tr_data)
    
    pl.seed_everything(args.seed_use)
    model_use = XceptionLightning(num_class, batch_size=args.batch_size,
                                  num_samples=num_samples, warmup_epochs=args.warmup_epochs,
                                  num_gpus=args.num_gpu, enc_lr=args.enc_lr,
                                  quantification=args.quantification, detection=args.detection, concentration_float=args.concentration_float,
                                  wavenumber=imshape[-1],
                                  strategy=args.strategy, epochs=args.epochs,
                                  reduce_channel_first=True,
                                  model_type=args.model_type, data_input_channel=data_input_channel,
                                  dataset=args.dataset)
    project_name="%s-Spectra_%s_detection_%s_quantification_%s" % (args.dataset, args.model_type, args.detection, args.quantification)
    
    if args.dataset == "PA" and args.detection == True and args.model_type == "resnet":
        wandb_logger = WandbLogger(project=project_name, config=args,
                                   save_dir=model_dir+"/")
        wandb_logger.watch(log_freq=500, log_graph=False, model=model_use)
    if args.detection:
        checkpoint_callback = get_monitors_for_detection_exp(model_dir)
    if args.quantification:
        checkpoint_callback = get_monitors_for_quantification_exp(model_dir)
    train_loader, val_loader, test_loader = psd.get_dataloader(tr_data, val_data, tt_data, args.batch_size,
                                                               args.workers)
    print("The length of the training loader", len(train_loader), "testing loader", len(val_loader),
          "testing loader", len(test_loader))

    log_steps = int(len(train_loader) / args.num_gpu) if args.num_gpu > 0 else len(train_loader)
    log_steps = [30 if log_steps > 30 else log_steps][0]
    if args.dataset == "PA" and args.detection == True and args.model_type == "resnet":
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=20,
                            gpus=args.num_gpu, log_every_n_steps=log_steps, logger=wandb_logger, 
                            sync_batchnorm=True if args.num_gpu > 1 else False,
                            strategy=args.strategy,
                            callbacks=checkpoint_callback, check_val_every_n_epoch=5,
                            deterministic=True,
                            gradient_clip_val=1.0)
    else:
        trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=20,
                            gpus=args.num_gpu, log_every_n_steps=log_steps, 
                            sync_batchnorm=True if args.num_gpu > 1 else False,
                            strategy=args.strategy,
                            callbacks=checkpoint_callback, check_val_every_n_epoch=5,
                            deterministic=True,
                            gradient_clip_val=1.0)

    trainer.fit(model_use, train_loader, val_loader)
    trainer.test(model_use, test_loader)
    trainer.save_checkpoint(model_dir + "/model-epoch=%02d.ckpt" % args.epochs)
    wandb.finish()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = config.get_model_args()
    if len(args.tr_limited_conc) > 0:
        args.tr_limited_conc = [float(v) for v in args.tr_limited_conc]
    for arg in vars(args):
        print(arg, getattr(args, arg))

    a = torch.from_numpy(np.zeros([1])).to(torch.device("cuda"))
    
    if args.loc == "scratch":
        model_mom = "/scratch/blia/exp_data/"
    elif args.loc == "nobackup":
        model_mom = "/nobackup/blia/exp_data/"
    else:
        model_mom = args.loc
    model_mom += "/Spectra_%s/%s" % (args.model_type, args.dataset)
    model_mom += "/detection_%s_quantification_%s_average_spectra_%s/" % (args.detection, args.quantification, args.avg_spectra)

    model_dir = model_mom + "/version_%d_selection_method_%s_select_percentage_%.3f" % (args.version,
                                                                                        args.top_selection_method,
                                                                                        args.percentage)
    model_dir += "_learning_rate_%.4f_concentration_float_%.4f" % (args.enc_lr, args.concentration_float) 
    model_dir_sub = model_dir + "_quantification_loss_%s/%s_repeat_%d/" % (args.quantification_loss, args.leave_method, args.leave_index) 
    if os.path.exists(model_dir_sub):
        ckpt = [v for v in os.listdir(model_dir_sub) if ".ckpt" in v]
        if len(ckpt) < 1:
            train(args, model_dir_sub, args.data_dir)
    else:
        train(args, model_dir_sub, args.data_dir)









