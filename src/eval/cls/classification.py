import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18
from torch.optim import Adam, SGD
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
import numpy as np



ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
ori_img_dataset = ImageFolder(ori_imb_data_path, transform=Compose([ ToTensor(),
                                                                     Normalize((0.5, 0.5, 0.5),
                                                                              (0.5, 0.5, 0.5))]))

test_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val"
test_dataset = ImageFolder(test_data_path, transform=Compose([ ToTensor(),
                                                                     Normalize((0.5, 0.5, 0.5),
                                                                              (0.5, 0.5, 0.5))]))


EBGAN_gened_data_path = "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data"
EBGAN_gened_data = ImageFolder(EBGAN_gened_data_path, transform=Compose([ ToTensor(),
                                                                     Normalize((0.5, 0.5, 0.5),
                                                                              (0.5, 0.5, 0.5))]))


ori_img_loader = DataLoader(ori_img_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# batch = iter(ori_img_dataset).__next__()
# concat_dataset = ConcatDataset([ori_img_dataset, EBGAN_gened_data])


cls_model = resnet18(num_classes=10).cuda()
optimizer = SGD(cls_model.parameters(), lr=0.02)

train_loader = tqdm(ori_img_loader, desc='Training model')
test_loader = tqdm(test_loader, desc='Test model')

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

metric_train = Metric(10)
metric_test = Metric(10)


for epoch in range(2):
    for i, batch in enumerate(ori_img_loader):
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
    print(f"loss:  {train_loss:.4f}(train) {test_loss:.4f}(test)")
    print(f"acc:   {train_acc:.2f}(train) {test_acc:.2f}(test)")
    for idx, (train_acc_per_cls_, test_acc_per_cls_) in enumerate(zip(train_acc_per_cls, test_acc_per_cls)):
        print(f"acc {idx}: {train_acc_per_cls_:.2f}(train) {test_acc_per_cls_:.2f}(test)")
