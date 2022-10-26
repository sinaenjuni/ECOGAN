import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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


class MergeDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        self.path1 = Path(path1)
        self.path2 = Path(path2)

        self.images = []
        self.targets = []

        self.get_data_paths(path1)
        self.get_data_paths(path2)

        self.transform = transform

    def get_data_paths(self, path):
        for path_ in Path(path).glob("*/*.JPEG"):
            self.images.append(path_)
            self.targets.append(int(path_.parts[-2][-1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
# paths = {"EBGAN_gened_data_path" : "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data",
#          "ContraGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/contraGAN_restore_data",
#          "ECOGAN_gened_data_path" : "/home/dblab/git/ECOGNA/save_files/ECOGAN_restore_data"}

paths = {"EBGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN",
         "BEGAN_pre-trained_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/EBGAN_pre-trained",
         # "ECOGAN_gened_data_path" : "/shared_hdd/sin/save_files/gened_img/ECOGAN"
        }


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

