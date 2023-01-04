import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose
from utils.misc import print_h5py, elapsed_time
import h5py
from utils.confusion_matrix import ConfusionMatrix
import importlib
from datetime import datetime
from argparse import ArgumentParser
import wandb


class H5py_dataset(Dataset):
    def __init__(self, path, query, transforms=None):
        file = h5py.File(path, 'r')        
        self.data = file[f'{query}/data'].astype(np.uint8)
        self.targets = file[f'{query}/targets'].astype(np.uint8)
        self.transforms = transforms
        self.classes, self.num_classes = np.unique(self.targets, return_counts=True)
        self.num_class = len(self.classes)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        
        if img.ndim == 2:
            img = img[..., None].repeat(3, 2)
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    


def worker(args):
    transforms = Compose([ToTensor(),
                        Resize(64),
                        Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

    # train_dataset = H5py_dataset(path=paths['ori'], query=f'ori/{data_names[args.data_seq]}/train', transforms=transforms)
    
    if args.data_seq != -1:
        train_dataset = H5py_dataset(path=paths['ori'], query=f'ori/{args.data_name}/train', transforms=transforms)
        print(train_dataset.num_classes)
        
    else:    
        # gen_dataset = H5py_dataset(path=paths['gen'], query=f'gen/{ids[args.data_seq]}/{data_names[args.data_seq]}', transforms=transforms)
        ori_dataset = H5py_dataset(path=paths['ori'], query=f'ori/{args.data_name}/train', transforms=transforms)
        gen_dataset = H5py_dataset(path=paths['gen'], query=f'gen/{ids[args.data_seq]}/{args.data_name}', transforms=transforms)
        train_dataset = ConcatDataset([ori_dataset, gen_dataset])
        print(ori_dataset.num_classes)

    # test_dataset = H5py_dataset(path=paths['ori'], query=f'ori/{data_names[args.data_seq]}/test', transforms=transforms)
    test_dataset = H5py_dataset(path=paths['ori'], query=f'ori/{args.data_name}/test', transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(test_dataset.num_classes)


    

    # device = torch.device(args.gpus)
    device = args.gpus

    module_model = importlib.import_module('torchvision.models')
    model = getattr(module_model, args.model)(num_classes=train_dataset.num_class).to(device)

    optimizer = SGD(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    cm_train = ConfusionMatrix(train_dataset.num_class)
    cm_test = ConfusionMatrix(train_dataset.num_class)
    global_epoch = 0
    losses_train = 0
    losses_test = 0

    best_step = 0
    best_acc = 0
    best_acc_cls = 0

    model.train()
    start_time = datetime.now()
    train_iter = iter(train_loader)
    for step in range(args.steps):
        try:
            img, label = next(train_iter)
        except StopIteration:
            global_epoch += 1
            train_iter = iter(train_loader)
            img, label = next(train_iter)
        img, label = img.to(device), label.to(device)
        
        optimizer.zero_grad()
        y_hat = model(img)
        loss = loss_fn(y_hat, label)
        loss.backward()
        optimizer.step()
        
        pred = F.softmax(y_hat, 1).argmax(1)
        losses_train += loss.item()
        cm_train.update(label.cpu().numpy(), pred.cpu().numpy())
        
        if (step+1) % 100 == 0:
            model.eval()
            for img, label in test_loader:
                img, label = img.to(device), label.to(device) 

                y_hat = model(img)
                loss = loss_fn(y_hat, label)

                pred = F.softmax(y_hat, 1).argmax(1)
                losses_test += loss.item()
                cm_test.update(label.cpu().numpy(), pred.cpu().numpy())
            
            
            losses_train = losses_train / 100
            losses_test = losses_test / len(test_loader)
            acc_train = cm_train.getAccuracy()
            acc_test = cm_test.getAccuracy()
            acc_train_cls = cm_train.getAccuracyPerClass()
            acc_test_cls = cm_test.getAccuracyPerClass()
            
            acc_train_cls = {f'train/acc/{idx}': round(acc, 4) for idx, acc in enumerate(acc_train_cls)}
            acc_test_cls = {f'test/acc/{idx}': round(acc, 4) for idx, acc in enumerate(acc_test_cls)}
            
            if best_acc < acc_test:
                best_step = step+1
                best_acc = acc_test
                best_acc_cls = acc_test_cls
                wandb.summary['best/acc'] = acc_test
                wandb.summary['best/step'] = best_step
                for idx, acc in enumerate(acc_test_cls.values()):
                    wandb.summary[f'best/acc/cls_{idx}'] = acc
                
            
            wandb.log({'train/loss': losses_train,
                    'test/loss': losses_test,
                    'train/acc': acc_train,
                    'test/acc': acc_test}, step=(step+1))
            
            wandb.log(acc_train_cls, step=(step+1))
            wandb.log(acc_test_cls, step=(step+1))
            
            
            print(f'step: {step+1}/{args.steps}({((step+1) / args.steps)*100:.2f}%), '
                        f'time: {elapsed_time(start_time)}, '
                        f'loss train: {losses_train}, '
                        f'loss test: {losses_test}, '
                        f'acc train: {acc_train}, '
                        f'acc test: {acc_test}, '
                        f'acc best: {best_acc}')
            print('acc_train_cls', acc_train_cls)
            print('acc_test_cls', acc_test_cls)
            print('best_acc_cls', best_acc_cls)
            
            model.train()
            losses_train = 0
            losses_test = 0
            cm_train.reset()
            cm_test.reset()
            
            
            
            # for idx, (acc_train, acc_test) in enumerate(zip(acc_train_cls, acc_test_cls)):
            #     print(f'cls {idx}: {acc_train}, {acc_test}')
            
            
if __name__ == '__main__':
    
    paths = {'ori':'/shared_hdd/sin/data.h5py',
         'gen':'/shared_hdd/sin/gen.h5py'}


    ids=["1r08m7xe",
    "3rczol1e",
    "3u337fao",
    "6gem8lca",
    "3p2vbr49",
    "i2mijyme",
    "iitr6wnc",
    "vpgfgriu",
    "bagan_cifar10",
    "bagan_fashionmnist"]

    data_names=["CIFAR10_LT",
    "CIFAR10_LT",
    "CIFAR10_LT",
    "CIFAR10_LT",
    "FashionMNIST_LT",
    "FashionMNIST_LT",
    "FashionMNIST_LT",
    "FashionMNIST_LT",
    "CIFAR10_LT",
    "FashionMNIST_LT"]
    
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet18', required=True)
    parser.add_argument("--gpus", type=int, default=1, required=True)
    parser.add_argument("--data_seq", type=int, default=0, required=True)
    parser.add_argument("--steps", type=int, default=1000, required=True)
    parser.add_argument("--lr", type=float, default=0.001, required=True)
    parser.add_argument("--batch_size", type=int, default=128, required=True)
    
    parser.add_argument("--data_name", type=str, required=False)
    parser.add_argument("--gen_id", type=str, required=False)


    args = parser.parse_args()
    if args.data_seq == -1:
        args.gen_id = -1
    else:
        args.data_name = data_names[args.data_seq]
        args.gen_id = ids[args.data_seq]
    
    print(args)
    wandb.init(project="cls_eval", entity="sinaenjuni", config=args)
    worker(args)
