import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from torch.optim import Adam, SGD
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
import numpy as np
from pathlib import Path
from PIL import Image
from utils import MergeDataset

import pytorch_lightning as pl

ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
# paths = {"EBGAN_gened_data_path" : "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data",
#          "ContraGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/contraGAN_restore_data",
#          "ECOGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/ECOGAN_restore_data"}

paths = {"EBGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN",
         "BEGAN_pre-trained_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN_pre-trained",
         # "ECOGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/ECOGAN"
        }


class Classification_Model(pl.LightningModule):
    def __init__(self, num_classes, lr=0.01):
        super(Classification_Model, self).__init__()
        self.lr = lr
        self.model = resnet18(num_classes=num_classes)
        self.conf = ConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        img, label = batch

        logit = self(img)
        loss = self.ce_loss(logit, label)

        self.conf.update(preds=logit.argmax(1), target=label)

        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        cm = self.conf.compute()
        acc = cm.trace() / cm.sum()
        acc_per_cls = cm.diagonal() / cm.sum(1)
        self.log_dict({'train/acc': acc,
                       'train/acc_per_cls': acc_per_cls},
                      prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch):
        img, label = batch
        logit = self(img)
        loss = self.ce_loss(logit, label)
        self.log('val/loss', loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

    def validation_epoch_end(self, outputs):

    def ce_loss(self, logit, label):
        return F.cross_entropy(logit, label)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)


for data_name, data_path in paths.items():
    print(data_name)
    merge_dataset = MergeDataset(ori_imb_data_path, data_path,
                                 transform=Compose([Resize(32),
                                                    ToTensor(),
                                                    Normalize((0.5, 0.5, 0.5),
                                                              (0.5, 0.5, 0.5))]))
    train_loader = DataLoader(merge_dataset, shuffle=True, batch_size=64)



    test_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val"
    test_dataset = ImageFolder(test_data_path, transform=Compose([ ToTensor(),
                                                                         Normalize((0.5, 0.5, 0.5),
                                                                                  (0.5, 0.5, 0.5))]))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

    print(np.unique(train_loader.dataset.targets, return_counts=True))



    cls_model = resnet18(num_classes=10).cuda()
    optimizer = SGD(cls_model.parameters(), lr=0.01)

    class Metric:
        def __init__(self, num_classes):
            self.pred = []
            self.label = []
            self.loss = []
            self.conf = ConfusionMatrix(num_classes=num_classes)

        def get_loss(self):
            return torch.stack(self.loss).mean()

        def get_acc(self):
            pred = torch.cat(self.pred)
            label = torch.cat(self.label)
            matrix = self.conf(pred, label)
            acc = matrix.trace() / matrix.sum()
            acc_per_cls = matrix.diagonal() / matrix.sum(1)
            return acc, acc_per_cls

        def clear(self):
            self.pred = []
            self.label = []
            self.loss = []

    best_epoch = 0
    best_acc = 0
    best_acc_per_cls = []
    best_loss = 0

    metric_train = Metric(10)
    metric_test = Metric(10)
    for epoch in range(100):
        for i, batch in enumerate(train_loader):
            img, label = batch[0].cuda(), batch[1].cuda()
            cls_model.train()

            optimizer.zero_grad()
            y_hat = cls_model(img)
            loss = F.cross_entropy(y_hat, label)

            loss.backward()
            optimizer.step()
            metric_train.loss.append(loss.cpu())

            pred = y_hat.argmax(1)
            metric_train.pred.append(pred.cpu())
            metric_train.label.append(label.cpu())

        for i, batch in enumerate(test_loader):
            img, label = batch[0].cuda(), batch[1].cuda()
            cls_model.eval()

            y_hat = cls_model(img)
            loss = F.cross_entropy(y_hat, label)

            metric_test.loss.append(loss.cpu())

            pred = y_hat.argmax(1)
            metric_test.pred.append(pred.cpu())
            metric_test.label.append(label.cpu())


        train_loss = metric_train.get_loss()
        test_loss = metric_test.get_loss()
        train_acc, train_acc_per_cls = metric_train.get_acc()
        test_acc, test_acc_per_cls = metric_test.get_acc()

        if test_acc > best_acc:
            best_epoch = epoch
            best_acc = test_acc
            best_loss = test_loss
            best_acc_per_cls = test_acc_per_cls

        print(f"epoch:\t{epoch}\t\t\t\t\t\t\t\t{best_epoch}")
        print(f"loss:\t{train_loss:.4f}(train)\t{test_loss:.4f}(test)\t{best_loss:.4f}(best)")
        print(f"acc:\t{train_acc:.2f}(train)\t\t{test_acc:.2f}(test)\t\t{best_acc:.2f}(best)")
        for idx, (train_acc_per_cls_, test_acc_per_cls_, best_acc_per_cls_) in enumerate(zip(train_acc_per_cls, test_acc_per_cls, best_acc_per_cls)):
            print(f"acc {idx}:\t{train_acc_per_cls_:.2f}(train)\t\t{test_acc_per_cls_:.2f}(test)\t\t{best_acc_per_cls_:.2f}(best)")

        metric_train.clear()
        metric_test.clear()

