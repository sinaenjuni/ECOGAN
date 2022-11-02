# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms import Compose, Normalize, ToTensor, Resize
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torch.optim import Adam, SGD
# from tqdm import tqdm
from torchmetrics import ConfusionMatrix
# import numpy as np
# from pathlib import Path
# from PIL import Image
# from utils import MergeDataset

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from utils.dataset import DataModule_

ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
# paths = {"EBGAN_gened_data_path" : "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data",
#          "ContraGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/contraGAN_restore_data",
#          "ECOGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/ECOGAN_restore_data"}

paths = {"EBGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN",
         "BEGAN_pre-trained_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN_pre-trained",
         # "ECOGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/ECOGAN"
}

class Classification_Model(pl.LightningModule):
    def __init__(self, img_dim, num_classes, lr, **kwargs):
        super(Classification_Model, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = resnet18(pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conf = ConfusionMatrix(num_classes=num_classes)
        self.best_acc = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        img, label = batch
        logit = self(img)
        loss = self.ce_loss(logit, label)

        self.conf.update(preds=logit.argmax(1), target=label)
        cm = self.conf.compute()
        acc = cm.trace() / cm.sum()
        acc_per_cls = cm.diagonal() / cm.sum(1)

        log_dict = {'train/acc': acc}
        log_dict.update({f'train/acc_cls{idx}':acc_ for idx, acc_ in enumerate(acc_per_cls)})
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    # def training_epoch_end(self, outputs):
    #     cm = self.conf.compute()
    #     acc = cm.trace() / cm.sum()
    #     acc_per_cls = cm.diagonal() / cm.sum(1)
    #     self.log_dict({'train/acc': acc,
    #                    'train/acc_per_cls': acc_per_cls},
    #                   prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        img, label = batch
        logit = self(img)
        loss = self.ce_loss(logit, label)

        self.conf.update(preds=logit.argmax(1), target=label)
        self.log('val/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'logit':logit}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs])
        logit = torch.cat([output['logit'] for output in outputs])

        cm = self.conf.compute()
        acc = cm.trace() / cm.sum()
        acc_per_cls = cm.diagonal() / cm.sum(1)

        log_dict = {'val/acc': acc}
        log_dict.update({f'val/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.best_acc < acc:
            self.best_acc = acc
            wandb.run.summary['best/epoch'] = self.current_epoch + 1
            wandb.run.summary['best/acc'] = acc
            for idx, acc_ in enumerate(acc_per_cls):
                wandb.run.summary[f'best/acc_cls{idx}'] = acc_


    def ce_loss(self, logit, label):
        return F.cross_entropy(logit, label)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)


# dm = DataModule_(data_name, batch_size=128)
# model = Classification_Model(num_classes=10, lr=0.01)



# trainer = pl.Trainer(
#     default_root_dir='/shared_hdd/sin/save_files/cls_tset/',
#     fast_dev_run=False,
#     max_epochs=100,
#     # callbacks=[pl.callbacks.ModelCheckpoint(monitor="train_loss", mode='min')],
#     logger=wandb_logger,
#     strategy='ddp',
#     accelerator='gpu',
#     gpus=1,
# )

parser = ArgumentParser()
parser.add_argument("--num_classes", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=0.01, required=False)
parser.add_argument("--img_dim", type=int, default=3, required=True)
parser.add_argument("--data_name", type=str, default='imb_CIFAR10',
                    choices=['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST'], required=True)


args = parser.parse_args()
dm = DataModule_.from_argparse_args(args)

model = Classification_Model(**vars(args))
wandb_logger = WandbLogger(project="eval_cls", name=f"{args.data_name}", log_model=True)
# wandb.define_metric('val/acc', summary='max')
wandb_logger.watch(model, log='all')

trainer = pl.Trainer.from_argparse_args(args,
    default_root_dir='/shared_hdd/sin/save_files/cls_tset/',
    fast_dev_run=False,
    max_epochs=100,
    # callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/acc", mode='max')],
    logger=wandb_logger,
    # strategy='ddp',
    strategy='ddp_find_unused_parameters_false',
    accelerator='gpu',
    gpus=1,
    # num_sanity_val_steps=0
)
trainer.fit(model, datamodule=dm)





