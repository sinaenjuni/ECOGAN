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
from utils.misc import GatherLayer



def worker(rank, world_size):
    print(f"rank: {rank}, world_size: {world_size}")
    setup(rank, world_size)

    if rank == 0:
        logger = wandb.init(project="eval_cls", entity="sinaenjuni")


    transforms = Compose([ToTensor(),
                          Resize(64),
                          Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

    dataset_module = importlib.import_module('utils.datasets')
    dataset_train = getattr(dataset_module, 'CIFAR10_LT')(is_train=True, is_extension=False, transform=transforms)


    if world_size > 1:
        sampler_train = DistributedSampler(dataset_train, rank=rank, num_replicas=world_size, shuffle=True)

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=128,
                              sampler=sampler_train if world_size > 1 else None,
                              shuffle=not (world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)

    model_module = importlib.import_module('models.bagan')
    encoder, decoder = getattr(model_module, 'pre_training')(loader_train, logger, world_size, rank, args)






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





