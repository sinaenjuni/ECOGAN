# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from torch.optim import Adam, SGD

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.dataset import DataModule_
from utils.confusion_matrix import ConfusionMatrix
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Lambda
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py



class imbalanced_dataset(Dataset):
    def __init__(self, data_info, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        base_path = Path('/shared_hdd/sin/gen/')

        self.h5_ori = h5py.File(base_path / 'ori.h5py', 'r')

        if self.is_train:
            self.data_ori = self.h5_ori[data_info]['train']['data']
            self.targets_ori = self.h5_ori[data_info]['train']['targets']
        else:
            self.data_ori = self.h5_ori[data_info]['test']['data']
            self.targets_ori = self.h5_ori[data_info]['test']['targets']

    def __len__(self):
        return len(self.data_ori)
    def __getitem__(self, idx):
        img = self.data_ori[idx]
        target = self.targets_ori[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class ClsTestDM(pl.LightningDataModule):
    def __init__(self, data_name, img_size, batch_size):
        super(ClsTestDM, self).__init__()
        self.data_name = data_name
        self.batch_size = batch_size
        self.img_size = img_size

        self.transform = Compose([ToTensor(),
                                 Resize(self.img_size),
                                 Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                                 Normalize([0.5, 0.5, 0.5],
                                           [0.5, 0.5, 0.5])
                                 ])
    def setup(self, stage):
        self.dataset_train = imbalanced_dataset(data_info=self.data_name, is_train=True, transform=self.transform)
        self.dataset_test = imbalanced_dataset(data_info=self.data_name, is_train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_test, shuffle=False, batch_size=self.batch_size, num_workers=4, pin_memory=True)


class Eval_cls_model(pl.LightningModule):
    def __init__(self, num_classes, lr, **kwargs):
        super(Eval_cls_model, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = resnet34(pretrained=False, num_classes=num_classes)
        # self.model.conv1 = nn.Conv2d(img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.cm_train = ConfusionMatrix(num_classes=num_classes)
        self.cm_val = ConfusionMatrix(num_classes=num_classes)
        self.best_acc = 0
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch):
        img, label = batch
        logit = self(img)
        loss = self.ce_loss(logit, label)
        self.cm_train.update(target=label.cpu().numpy(), pred=logit.argmax(1).cpu().numpy())
        # cm = self.cm.compute()
        # acc = cm.trace() / cm.sum()
        # acc_per_cls = cm.diagonal() / cm.sum(1)

        # log_dict = {'train/acc': acc}
        # log_dict.update({f'train/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})
        # self.log_dict(log_dict, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    def training_epoch_end(self, outputs):
        acc = self.cm_train.getAccuracy()
        acc_per_cls = self.cm_train.getAccuracyPerClass()
        log_dict = {'train/accuracy': acc}
        log_dict.update({f'train/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})
        self.log_dict(log_dict,
                      prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.cm_train.reset()
    def validation_step(self, batch, batch_idx):
        img, label = batch
        logit = self(img)
        loss = self.ce_loss(logit, label)
        self.cm_val.update(target=label.cpu().numpy(), pred=logit.argmax(1).cpu().numpy())
        self.log('val/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}
    def validation_epoch_end(self, outputs):
        acc = self.cm_val.getAccuracy()
        acc_per_cls = self.cm_val.getAccuracyPerClass()
        log_dict = {'val/accuracy': acc}
        log_dict.update({f'val/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.best_acc < acc:
            self.best_acc = acc
            self.logger.log_metrics({'best/epoch': self.current_epoch})
            self.logger.log_metrics({'best/steps': self.global_step})
            self.logger.log_metrics({'best/accuracy': acc})
            for idx, acc_ in enumerate(acc_per_cls):
                self.logger.log_metrics({f'best/acc_cls{idx}':acc_})
        self.cm_val.reset()

    def ce_loss(self, logit, label):
        return F.cross_entropy(logit, label)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)




parser = ArgumentParser()
parser.add_argument("--num_classes", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=0.01, required=False)
parser.add_argument("--img_size", type=int, default=32, required=False)
parser.add_argument("--batch_size", type=int, default=128, required=False)
parser.add_argument("--data_name", type=str, default='CIFAR10_LT',
                    choices=['CIFAR10_LT', 'FashionMNIST_LT', 'Places_LT'], required=False)
parser.add_argument("--gpus", nargs='+', type=int, default=7, required=False)

args = parser.parse_args()
# dm = Eval_gen_cls_dataset.from_argparse_args(args)
dm = ClsTestDM.from_argparse_args(args)
# parser.add_argument("--img_dim", type=int, default=dm.img_dim, required=False)
# args = parser.parse_args()

model = Eval_cls_model(**vars(args))
wandb_logger = WandbLogger(project="eval_cls", name=f"resnet34", log_model=True)
# wandb.define_metric('val/acc', summary='max')
# wandb_logger.watch(model, log='all')

trainer = pl.Trainer.from_argparse_args(args,
    default_root_dir='/shared_hdd/sin/save_files/cls_tset/',
    max_epochs=-1,
    max_steps=100000,
    # fast_dev_run=True,
    # max_epochs=200,
    # callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/acc", mode='max')],
    logger=wandb_logger,
    # strategy='ddp',
    strategy='ddp_find_unused_parameters_false',
    accelerator='gpu',
    # gpus=[6, 7],
    num_sanity_val_steps=0
)
trainer.fit(model, datamodule=dm)





