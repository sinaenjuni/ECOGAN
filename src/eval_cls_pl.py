# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.optim import Adam, SGD

from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.dataset import DataModule_, Eval_gen_cls_dataset
from utils.confusion_matrix import ConfusionMatrix
# from torchmetrics import ConfusionMatrix

# ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
# # paths = {"EBGAN_gened_data_path" : "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data",
# #          "ContraGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/contraGAN_restore_data",
# #          "ECOGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/ECOGAN_restore_data"}
#
# paths = {"EBGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN",
#          "BEGAN_pre-trained_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN_pre-trained",
#          # "ECOGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/ECOGAN"
# }

# gen_path = Path('/shared_hdd/sin/gen/')
# gen_path = Path('/shared_hdd/sin/gen/2000/ECOGAN(imb_CIFAR10_128)')
#
# for i in gen_path.glob('*/*'):
#     # print(i)
#     print(int(i.parts[-2][-1]))
# t_path = list(gen_path.glob('*/*'))
#
# [i for i in gen_path.glob('*/*/*/*')]



# for path_ in gen_path.glob('*/*'):
#     print(path_)
#     for ori_name, ori_path in paths_train.items():
#         if ori_name in str(path_):
#             print(path_, ori_name)
#     # if str(path_) in paths_train.keys():
#     #     print(path_)
#
#     # dataset = ImageFolder(type)
#     # print(dataset, dataset.classes)
#
# data_names = ['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST']
# sel_name = [data_name for data_name in data_names if data_name in str(path_)][0]
#


# paths_train = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
#                'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/train',
#                'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/train'}
#
# paths_test = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val',
#               'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/val',
#               'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/val'}


class Eval_cls_model(pl.LightningModule):
    def __init__(self, img_dim, num_classes, lr, **kwargs):
        super(Eval_cls_model, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.model = resnet18(pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(img_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        log_dict = {'train/acc': acc}
        log_dict.update({f'train/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})
        self.log_dict(log_dict,
                      prog_bar=True, logger=True, on_step=False, on_epoch=True)
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
        log_dict = {'val/acc': acc}
        log_dict.update({f'val/acc_cls{idx}': acc_ for idx, acc_ in enumerate(acc_per_cls)})

        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        if self.best_acc < acc:
            self.best_acc = acc
            self.logger.log_metrics({'best/epoch': self.current_epoch + 1})
            self.logger.log_metrics({'best/acc': acc})
            for idx, acc_ in enumerate(acc_per_cls):
                self.logger.log_metrics({f'best/acc_cls{idx}':acc_})
        self.cm_val.reset()

    def ce_loss(self, logit, label):
        return F.cross_entropy(logit, label)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)


# dm = DataModule_(data_name, batch_size=128)
# model = Eval_cls_model(num_classes=10, lr=0.01)



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
# parser.add_argument("--gen_path", type=str, required=True)
parser.add_argument("--num_classes", type=int, default=10, required=False)
parser.add_argument("--max_epochs", type=int, default=2, required=False)
# parser.add_argument("--max_steps", type=int, default=1000, required=False)
parser.add_argument("--lr", type=float, default=0.01, required=False)
parser.add_argument("--img_size", type=int, default=32, required=False)
parser.add_argument("--is_sampling", type=bool, default=False, required=False)
parser.add_argument("--img_dim", type=int, default=3, required=False)
parser.add_argument("--data_name", type=str, default='imb_CIFAR10',
                    choices=['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST'], required=False)
parser.add_argument("--gpus", nargs='+', type=int, default=7, required=False)

args = parser.parse_args()
# dm = Eval_gen_cls_dataset.from_argparse_args(args)
dm = DataModule_.from_argparse_args(args)
# parser.add_argument("--img_dim", type=int, default=dm.img_dim, required=False)
# args = parser.parse_args()

model = Eval_cls_model(**vars(args))
wandb_logger = WandbLogger(project="eval_cls", name=f"original", log_model=True)
# wandb.define_metric('val/acc', summary='max')
# wandb_logger.watch(model, log='all')ß

trainer = pl.Trainer.from_argparse_args(args,
    default_root_dir='/shared_hdd/sin/save_files/cls_tset/',
    # fast_dev_run=True,
    max_epochs=200,
    # callbacks=[pl.callbacks.ModelCheckpoint(monitor="val/acc", mode='max')],
    logger=wandb_logger,
    # strategy='ddp',
    strategy='ddp_find_unused_parameters_false',
    accelerator='gpu',
    # gpus=[6, 7],
    num_sanity_val_steps=0
)
trainer.fit(model, datamodule=dm)





