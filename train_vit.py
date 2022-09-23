"""
Created on 16:30 at 17/11/2021
@author: bo
"""
from multiprocessing import reduction
import os
import math
import numpy as np
import torch
import torch.nn as nn
import pickle
import wandb
from torch.utils.data import random_split, DataLoader
import sys, random
import data.prepare_sers_data as psd
import models.vit_model as vit_model
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
# from configs.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import AMPType
import configs.common as config
from sklearn.metrics import f1_score, precision_score
from typing import Any, Dict, Optional


log_base=2.3
if log_base==2:
    log_fun = np.log2 
elif log_base==10:
    log_fun=np.log10 
elif log_base==2.3:
    log_fun = np.log 



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


def get_lr_schedule(lr_init, epochs, num_global_iter_per_epoch, warmup_epochs=10, schedule_mode="cosine"):
    final_lr = 1e-5
    warmup_lr_schedule = np.linspace(0.0, lr_init, num_global_iter_per_epoch * warmup_epochs)
    iters = np.arange(num_global_iter_per_epoch * (epochs - warmup_epochs))
    if schedule_mode == "cosine":
        cosine_lr_schedule = np.array([
            final_lr + 0.5 * (lr_init - final_lr) *
            (1 + math.cos(math.pi * t / (num_global_iter_per_epoch * (epochs - warmup_epochs))))
            for t in iters
        ])        
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    return lr_schedule


class ViTLightning(LightningModule):
    def __init__(self, num_classes, patch_size, input_feature, num_layers, num_heads, mlp_dim, dropout=0.0,
                 batch_size=512, num_samples=45000, warmup_epochs=10, num_gpus=1, enc_lr=0.03,
                 add_positional_encoding=True, quantification=False, detection=True,
                 imshape=(224, 224, 3), strategy="dp", epochs=50,
                 concentration_float=1e-6,
                 schedule_mode="cosine", initialisation="xavier"):
        super().__init__()

        self.save_hyperparameters()
        self.vit = vit_model.ViT(tuple(imshape[:2]), tuple(patch_size),
                                 num_classes, input_feature, num_layers, num_heads, mlp_dim,
                                 pool="cls", channels=imshape[-1],
                                 dim_head=input_feature // num_heads, dropout=dropout,
                                 add_positional_encoding=add_positional_encoding,
                                 quantification=quantification,
                                 detection=detection, 
                                 initialisation=initialisation)

        global_batch_size = global_batch_size_calc(batch_size, num_gpus, strategy)
        self.train_iters_per_epoch = num_samples // global_batch_size
        self.lr_schedule = get_lr_schedule(enc_lr, epochs, self.train_iters_per_epoch, warmup_epochs=warmup_epochs, 
                                           schedule_mode=schedule_mode)

    def configure_optimizers(self):
        params = self.parameters()
        if self.hparams.quantification is True:
            weight_decay_param = 5e-4 
        if self.hparams.detection is True:
            weight_decay_param = 5e-4 #5e-4
        optimizer = torch.optim.SGD(params, self.hparams.enc_lr, momentum=0.9, weight_decay=weight_decay_param)
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
            self.log("learning_rate_encoder", self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)

        if self.trainer.amp_backend == AMPType.NATIVE:
            optimizer_closure()
            self.trainer.scaler.step(optimizer)
        elif self.trainer.amp_backend == AMPType.APEX:
            optimizer_closure()
            optimizer.step()
        else:
            optimizer.step(closure=optimizer_closure)
    
    def get_quantification_performance(self, pred_quan, gt_quan):
        pred_quan = torch.clamp(pred_quan, log_fun(self.hparams.concentration_float))
        loss_quan = nn.MSELoss(reduction='sum')(pred_quan, gt_quan)
        if log_base != 2.3:
            quan_act = torch.pow(log_base, pred_quan)
            y_quan_act = torch.pow(log_base, gt_quan)
        else:
            quan_act = pred_quan.exp()
            y_quan_act = gt_quan.exp()
        top = (quan_act - y_quan_act).pow(2).sum()
        bottom = (y_quan_act - y_quan_act.mean()).pow(2).sum()
        accu_quan = 1.0 - top.div(bottom)
        accu_quan_rmae = ((quan_act - y_quan_act).abs().div(y_quan_act)).sum()
        accu_quan_rae = ((quan_act - y_quan_act).pow(2).div(y_quan_act.pow(2))).sum()                
        top = (pred_quan - gt_quan).pow(2).sum()
        bottom = (gt_quan - gt_quan.mean()).pow(2).sum()
        accu_quan_rsquare_log = 1.0 - top.div(bottom)
        return loss_quan, accu_quan, accu_quan_rmae, accu_quan_rsquare_log, accu_quan_rae
        
    def training_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.vit(x)
        
        if self.hparams.detection:
            loss = nn.CrossEntropyLoss()(pred, y.long())
            self.log("train_loss", loss, sync_dist=True)
        else:
            loss = 0.0
        if self.hparams.quantification:       
            loss_quan, accu_quan, accu_quan_rmae, accu_quan_rsquare_log, accu_quan_rae = self.get_quantification_performance(quan, y_conc, subset=True)
            loss += loss_quan.div(len(x))
            self.log("train_quantification_loss", loss_quan.div(len(x)), sync_dist=True)
            self.log("train_quantification_rsquare", accu_quan, sync_dist=True)
            self.log("training_quantification_rae", accu_quan_rae, sync_dist=True)
            self.log("training_quantification_rmae", accu_quan_rmae, sync_dist=True)
            self.log("training_quantification_rsquare_log", accu_quan_rsquare_log, sync_dist=True)
        if self.hparams.detection:
            accu = (pred.argmax(axis=-1) == y).sum().div(len(x))
            self.log("train_acc", accu, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.vit(x)
        if self.hparams.detection is True:
            loss = nn.CrossEntropyLoss()(pred, y.long())
            self.log("validation_loss", loss, sync_dist=True)
        else:
            loss = 0.0
        if self.hparams.quantification:       
            loss_quan, accu_quan, accu_quan_rmae, accu_quan_rsquare_log, accu_quan_rae = self.get_quantification_performance(quan, y_conc)
            loss += loss_quan.div(len(x))
            self.log("validation_quantification_loss", loss_quan.div(len(x)), sync_dist=True)
            self.log("validation_quantification_rsquare", accu_quan, sync_dist=True)
            self.log("validation_quantification_rae", accu_quan_rae, sync_dist=True)
            self.log("validation_quantification_rmae", accu_quan_rmae, sync_dist=True)
            self.log("validation_quantification_rsquare_subset", accu_quan_rsquare_log, sync_dist=True)
        if self.hparams.detection:
            y_detach_cpu = y.detach().cpu()
            val_pred = pred.argmax(axis=-1).detach().cpu()
            accu = (val_pred == y_detach_cpu).sum().div(len(x))
            num_false_positive = (val_pred[y_detach_cpu == 0] == 0).sum().div((y_detach_cpu == 0).sum())
            val_f1_score = f1_score(y_detach_cpu, val_pred)
            self.log("validation_accuracy", accu, sync_dist=True)
            self.log("validation_f1_score", val_f1_score, sync_dist=True)
            self.log("validation_false_positive", num_false_positive, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, y_conc = batch
        feat, pred, quan = self.vit(x)
        if self.hparams.detection is True:
            loss = nn.CrossEntropyLoss()(pred, y.long())
            self.log("test_loss", loss, sync_dist=True)
        if self.hparams.quantification:
            loss_quan, accu_quan, accu_quan_rmae, accu_quan_rsquare_log, accu_quan_rae = self.get_quantification_performance(quan, y_conc)
            self.log("test_quantification_loss", loss_quan.div(len(x)), sync_dist=True)
            self.log("test_quantification_rsquare", accu_quan, sync_dist=True)
            self.log("test_quantification_rae", accu_quan_rae, sync_dist=True)
            self.log("test_quantification_rmae", accu_quan_rmae, sync_dist=True)
            self.log("test_quantification_rsquare_subset", accu_quan_rsquare_log, sync_dist=True)
        if self.hparams.detection:
            accu = (pred.argmax(axis=-1) == y).sum().div(len(x))
            tt_f1_score = f1_score(y.detach().cpu(), pred.argmax(axis=-1).detach().cpu())
            self.log("test_accuracy", accu, sync_dist=True)
            self.log("test_f1_score", tt_f1_score, sync_dist=True)
            
            
class modelcheck(ModelCheckpoint):
    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[torch.Tensor] = None) -> bool:
        if trainer.current_epoch < 150:
            return False
        return super().check_monitor_top_k(trainer, current)
            
            
class modelcheckdetect(ModelCheckpoint):
    def check_monitor_top_k(self, trainer: "pl.Trainer", current: Optional[torch.Tensor] = None) -> bool:
        if trainer.current_epoch < 100:
            return False
        return super().check_monitor_top_k(trainer, current)
            
            
def get_monitors_for_quantification_exp(model_dir):
    if "SIMU_TYPE" in model_dir:
        monitor = ["validation_quantification_rsquare", "validation_quantification_loss"]
    else:
        monitor = ["validation_quantification_loss", "validation_quantification_rae", 
                   "validation_quantification_rmae"]
    check_point_g = []
    for s_monitor in monitor:
        if "loss" in s_monitor or "rae" in s_monitor or "rmae" in s_monitor:
            mode = "min"
        elif "rsquare" in s_monitor:
            mode="max"
        checkpoint_callback = modelcheck(monitor=s_monitor,
                                                dirpath=model_dir,
                                                filename="model-{epoch:02d}-{%s:.2f}" % s_monitor,
                                                save_top_k=4,
                                                mode=mode)
        check_point_g.append(checkpoint_callback)
    return check_point_g


def get_monitors_for_detection_exp(model_dir):
    monitor = ["validation_accuracy", "validation_loss", "validation_f1_score"]
    check_point_g = []
    for s_monitor in monitor:
        if "accuracy" in s_monitor or "f1_score" in s_monitor or "false_positive" in s_monitor:
            mode="max"
        else:
            mode="min"
        checkpoint_callback = modelcheckdetect(monitor=s_monitor,
                                               dirpath=model_dir,
                                               filename="model-{epoch:02d}-{%s:.4f}" % s_monitor,
                                                save_top_k=5,
                                                mode=mode)
        check_point_g.append(checkpoint_callback)
    return check_point_g            
        

def train(args, model_dir, data_dir="../image_dataset/"):
    create_dir(model_dir)
    np.random.seed(args.augment_seed_use)
    args.seed_use=int(args.seed_use)
    
    data_obj_loader = psd.ReadSERSData(args.dataset,
                                       target_shape=[args.target_h, args.target_w],
                                       top_selection_method=args.top_selection_method,
                                       percentage=args.percentage,
                                       concentration_float=args.concentration_float,
                                       quantification=args.quantification,
                                       detection=args.detection, 
                                       leave_index=args.leave_index,
                                       leave_method=args.leave_method,
                                       path_mom=data_dir)
    tr_data, val_data, tt_data, num_samples, imshape, num_class, _ = data_obj_loader.forward()

    patch_size = [args.patch_height, args.patch_width]  # The original sed is 42
    pl.seed_everything(args.seed_use)
    
    if len(tr_data) / args.batch_size < 1:
        args.batch_size = len(tr_data)
    
    model_use = ViTLightning(num_classes=num_class, patch_size=patch_size,
                             input_feature=args.input_feature, num_layers=args.num_layers,
                             num_heads=args.num_heads, mlp_dim=args.mlp_dim, dropout=0.0,
                             batch_size=args.batch_size, num_samples=num_samples, warmup_epochs=args.warmup_epochs,
                             num_gpus=args.num_gpu, enc_lr=args.enc_lr,
                             add_positional_encoding=args.add_positional_encoding, quantification=args.quantification,
                             detection=args.detection,
                             strategy=args.strategy, imshape=imshape, epochs=args.epochs,
                             concentration_float=args.concentration_float, 
                             schedule_mode=args.lr_schedule,
                             initialisation=args.model_init)
    project_name = "%s-VIT-detection-%s-quantification-%s" % (args.dataset, args.detection, args.quantification)

    wandb_logger = WandbLogger(project=project_name, config=args,
                               save_dir=model_dir+"/")
    wandb_logger.watch(log_freq=500, log_graph=False, model=model_use)
    if args.quantification is True:
        checkpoint_group = get_monitors_for_quantification_exp(model_dir)
    if args.detection is True:
        checkpoint_group = get_monitors_for_detection_exp(model_dir)

    train_loader, val_loader, test_loader = psd.get_dataloader(tr_data, val_data, tt_data, args.batch_size,
                                                               args.workers)
    print("The length of the training loader", len(train_loader), "testing loader", len(val_loader),
          "testing loader", len(test_loader))

    log_steps = int(len(train_loader) / args.num_gpu) if args.num_gpu > 0 else len(train_loader)
    log_steps = [30 if log_steps > 30 else log_steps][0]
    trainer = pl.Trainer(max_epochs=args.epochs, progress_bar_refresh_rate=20,
                         gpus=args.num_gpu, logger=wandb_logger, log_every_n_steps=log_steps,
                         sync_batchnorm=True if args.num_gpu > 1 else False,
                         strategy=args.strategy,
                         callbacks=checkpoint_group, check_val_every_n_epoch=5,
                         deterministic=True,
                         gradient_clip_val=1.0)
    print("Evaluation with the validation loader")
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
    if args.loc == "home":
        model_mom = "../exp_data/VIT/%s" % args.dataset
    elif args.loc == "scratch":
        model_mom = "/scratch/blia/exp_data/VIT/%s" % args.dataset 
    elif args.loc == "nobackup":
        model_mom = "/nobackup/blia/exp_data/VIT/%s" % args.dataset 
    else:
        model_mom = args.loc + "/VIT/%s" % args.dataset
    
    model_mom += "/detection_%s_quantification_%s/" % (args.detection, args.quantification)
    model_dir = model_mom + "/version_%d_ViT%s" % (args.version, args.model_type)
    model_dir += "_normalization_%s" % "none"
    model_dir += "_learning_rate_%.4f" % args.enc_lr 
    model_dir += "_target_shape_%d_%s" % (args.target_h, args.model_init)
    model_dir_sub = model_dir + "_quantification_loss_%s/%s_repeat_%d/" % (args.quantification_loss, args.leave_method, args.leave_index) 
    print("Done creating dir")
    train(args, model_dir_sub, args.data_dir)










