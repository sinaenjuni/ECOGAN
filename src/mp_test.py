# from tqdm import tqdm
# import torch.multiprocessing as mp
#
#
# def get_train_batch():
#     try:
#         img, label = next(iter_train)
#     except StopIteration:
#         iter_train = iter(loader_train)
#         img, label = next(iter_train)
#
#     return img, label
#
#
# def worker():
#     transforms = Compose([ToTensor(),
#                           Resize(32),
#                           Normalize(mean=[0.5, 0.5, 0.5],
#                                     std=[0.5, 0.5, 0.5])])
#
    # dataset_module = importlib.import_module('utils.datasets')
    # dataset_train = getattr(dataset_module, 'CIFAR10_LT')(is_train=True, is_extension=False, transform=transforms)
    # dataset_test = getattr(dataset_module, 'CIFAR10_LT')(is_train=False, is_extension=False, transform=transforms)
    #
    # loader_train = DataLoader(dataset=dataset_train, batch_size=128, shuffle=True)
#
#     loader_train = loader_train
#     iter_train = iter(loader_train)
#
#     for i in tqdm(range(1000)):
#         worker.get_train_batch()
#
#
# worker()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import importlib

from tqdm import tqdm
import numpy as np
import wandb
from argparse import ArgumentParser

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confmat = np.zeros((num_classes, num_classes))
        self.minlength = num_classes**2
    def update(self, target, pred):
        assert len(target) == len(pred)
        unique_mapping = target.reshape(-1) * self.num_classes + pred.reshape(-1)
        bins = np.bincount(unique_mapping, minlength=self.minlength)
        self.confmat += bins.reshape(self.num_classes, self.num_classes)
    def getConfusionMactrix(self):
        return self.confmat
    def getAccuracy(self):
        return np.nan_to_num(self.confmat.trace() / self.confmat.sum())
    def getAccuracyPerClass(self):
        return np.nan_to_num(self.confmat.diagonal() / self.confmat.sum(1))
    def reset(self):
        self.confmat = np.zeros((self.num_classes, self.num_classes))



def worker(rank, world_size):
    print(f"rank: {rank}, world_size: {world_size}")
    setup(rank, world_size)

    if args.local_rank == 0:
        wandb.init(project="eval_cls", entity="sinaenjuni")


    transforms = Compose([ToTensor(),
                          Resize(32),
                          Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

    dataset_module = importlib.import_module('utils.datasets')
    dataset_train = getattr(dataset_module, 'CIFAR10_LT')(is_train=True, is_extension=False, transform=transforms)
    dataset_test = getattr(dataset_module, 'CIFAR10_LT')(is_train=False, is_extension=False, transform=transforms)


    if world_size > 1:
        sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
        sampler_test = DistributedSampler(dataset_test, rank=rank, num_replicas=world_size, shuffle=True)

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=128,
                              sampler=sampler_train if world_size > 1 else None,
                              shuffle= not (world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)
    iter_train = iter(loader_train)

    loader_train = DataLoader(dataset=dataset_test,
                              batch_size=128,
                              sampler=sampler_test if world_size > 1 else None,
                              shuffle=not(world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)


    model_module = importlib.import_module('torchvision.models.resnet')
    model = getattr(model_module, 'resnet18')(num_classes=10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)


    loss_train = []
    loss_test = []
    # loss_train_append = loss_train.append
    # loss_test_append = loss_test.append



    conf_train = ConfusionMatrix(10)
    conf_test = ConfusionMatrix(10)


    for step in range(100):
        try:
            img, labels = next(iter_train)
            img, labels = img.to(rank), labels.to(rank)
        except StopIteration:
            iter_train = iter(loader_train)
            img, labels = next(iter_train)
            img, labels = img.to(rank), labels.to(rank)

        ddp_model.train()
        optimizer.zero_grad()
        outputs = ddp_model(img)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_train.append(loss)
        conf_train.update(labels.cpu(), F.softmax(outputs, dim=1).argmax(1).cpu())


        print(step, torch.stack(loss_train).mean())

        if (step+1) % 10 == 0:
            acc = conf_train.getAccuracy()
            acc_per_cls = conf_train.getAccuracyPerClass()

            print(acc)
            wandb.log({f'acc/train{rank}': acc}, step=step+1)
            # print({cls: f"{val:.2f}" for cls, val in zip(range(len(acc_per_cls)), acc_per_cls)})
            conf_train.reset()
            loss_train = []

            for img, labels in loader_train:
                img, labels = img.to(rank), labels.to(rank)
                ddp_model.eval()
                outputs = ddp_model(img)
                loss = loss_fn(outputs, labels)
                loss_test.append(loss)
                conf_test.update(labels.cpu(), F.softmax(outputs, dim=1).argmax(1).cpu())

            acc = conf_test.getAccuracy()
            acc_per_cls = conf_test.getAccuracyPerClass()


            print(step, torch.stack(loss_test).mean())
            print(acc)
            # print({cls: f"{val:.2f}" for cls, val in zip(range(len(acc_per_cls)), acc_per_cls)})
            conf_test.reset()
            loss_test = []

        # print(outputs.argmax(1))



    cleanup()



def load_configs_init():
    parser = ArgumentParser()
    parser.add_argument("--gpus", nargs='+', default=[1], type=int, required=False)

    args = parser.parse_args()
    # cfgs = vars(args)

    return args

def setup(rank, world_size, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend=backend,
                            rank=rank,
                            # init_method='env://',
                            world_size=world_size)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # setup_for_distributed(rank == 0) # 특정 rank에서만 print 허용

def cleanup():
    dist.destroy_process_group()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


if __name__ == "__main__":
    #
    # print(gpus_per_node, rank)

    args = load_configs_init()
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, args.gpus))
    world_size = len(args.gpus)

    # gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
    # print(gpus_per_node, rank)

    # wandb.require('service')
    # run = wandb.init(project="eval_cls", entity="sinaenjuni")

    if len(args.gpus) > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        ctx = torch.multiprocessing.spawn(fn=worker,
                                          args=(world_size,),
                                          nprocs=world_size,
                                          join=False)

        # print(ctx)
        ctx.join()
        for process in ctx.processes:
            process.kill()

    else:
        print("Train the models through Single GPU mode.")
        worker(rank=0, world_size=world_size)





