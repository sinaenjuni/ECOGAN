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





class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def spmd_main(rank, world_size):
    print(f"rank: {rank}, world_size: {world_size}")
    setup(rank, world_size)
    current_steps = 0

    transforms = Compose([ToTensor(),
                          Resize(32),
                          Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

    dataset_module = importlib.import_module('utils.datasets')
    dataset_train = getattr(dataset_module, 'CIFAR10_LT')(is_train=True, is_extension=False, transform=transforms)
    # dataset_test = getattr(dataset_module, 'CIFAR10_LT')(is_train=False, is_extension=False, transform=transforms)

    sampler = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=128,
                              sampler=sampler,
                              num_workers=1,
                              pin_memory=True,
                              persistent_workers=True)
    iter_train = iter(loader_train)

    # print(len(loader_train))

    model_module = importlib.import_module('torchvision.models.resnet')
    model = getattr(model_module, 'resnet18')(num_classes=10).to(rank)
    # model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in tqdm(range(1000)):
        try:
            img, labels = next(iter_train)
            img, labels = img.to(rank), labels.to(rank)
        except StopIteration:
            iter_train = iter(loader_train)
            img, labels = next(iter_train)
            img, labels = img.to(rank), labels.to(rank)

        # print(img.size())
        optimizer.zero_grad()
        # outputs = ddp_model(torch.randn(20, 10))
        # labels = torch.randn(20, 5).to(rank)
        outputs = ddp_model(img)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(epoch, loss)

        # current_steps += 1
        # print(current_steps)
    cleanup()








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
    gpus = [1,2,3,4]
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, gpus))


    mp.set_start_method("spawn", force=True)
    world_size = len(gpus)
    print("Train the models through DistributedDataParallel (DDP) mode.")
    ctx = torch.multiprocessing.spawn(fn=spmd_main,
                                      args=(world_size,),
                                      nprocs=world_size,
                                      join=False)
    # print(ctx)
    ctx.join()
    for process in ctx.processes:
        process.kill()







