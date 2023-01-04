import os
import wandb
import importlib
import numpy as np
from utils import misc
from datetime import datetime
from argparse import ArgumentParser
from utils.misc import load_configs_init

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda



def worker(rank, world_size, args):
    print(f"rank: {rank}, world_size: {world_size}")
    if world_size > 1:
        misc.setup(rank, world_size)
        misc.fix_seed(rank)
        
    if args.logger and rank==0:
        wandb.init(project="bagan", entity="sinaenjuni", config=args)

 
    transforms = Compose([ToTensor(),
                        Resize(64),
                        Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

    dataset_module = importlib.import_module('utils.datasets')
    dataset_train = getattr(dataset_module, args.data_name)(is_train=True, is_extension=False, transform=transforms)
    
    if world_size > 1:
        sampler_train = DistributedSampler(dataset_train, 
                                           rank=rank, 
                                           num_replicas=world_size,
                                           shuffle=True)

    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=args.batch_size,
                              sampler=sampler_train if world_size > 1 else None,
                              shuffle=not (world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)



    model_module = importlib.import_module(f'models.{args.model}')
    # getattr(model_module, 'pre_training')(loader_train, logger, world_size, rank, args)
    
    getattr(model_module, 'training')(loader_train, world_size=world_size, rank=rank, args=args)

    # print("@@@@@", model_module)
 
 
    misc.cleanup()



if __name__ == "__main__":
    
    args = load_configs_init()
    world_size = len(args.gpus)
    args.batch_size = args.batch_size//world_size
    args.path = os.path.join(args.path, args.model, args.data_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, args.gpus))
    # os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.makedirs(args.path, exist_ok=True)
    print(args)
    
    if world_size > 1:
        mp.set_start_method("spawn", force=True)
        print("Train the models through DistributedDataParallel (DDP) mode.")
        ctx = torch.multiprocessing.spawn(fn=worker,
                                          args=(world_size, args),
                                          nprocs=world_size,
                                          join=False)

        # print(ctx)
        ctx.join()
        for process in ctx.processes:
            process.kill()

    else:
        print("Train the models through Single GPU mode.")
        worker(rank=0, world_size=world_size, args=args)