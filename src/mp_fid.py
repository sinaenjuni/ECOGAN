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

from metric.inception_net_V2 import EvalModel
from metric.fid import calculate_mu_sigma, frechet_inception_distance
from metric.ins import calculate_inception_score

from metric.ins import calculate_kl_div
from metric.fid import calculate_mu_sigma, frechet_inception_distance


def worker(rank, world_size):
    print(f"rank: {rank}, world_size: {world_size}")
    setup(rank, world_size)

    # if rank == 0:
    #     wandb.init(project="eval_cls", entity="sinaenjuni")


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
                              shuffle=not (world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)
    iter_train = iter(loader_train)

    dataset_test = DataLoader(dataset=dataset_test,
                              batch_size=128,
                              sampler=sampler_test if world_size > 1 else None,
                              shuffle=not(world_size > 1),
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True)

    eval_model = EvalModel(world_size=world_size, device=rank)
    eval_model.eval()
    
    real_feature, real_logit = eval_model.stacking_feature(loader_train)
    mu_real, sigma_real = calculate_mu_sigma(real_feature.cpu().numpy())
    
    

    fid = frechet_inception_distance(mu_real, sigma_real, mu_real, sigma_real)
    ins = calculate_kl_div(ps_list=real_logit)[0]
    print(fid, ins)


    #
    # for step in range(100):
    #     try:
    #         img, labels = next(iter_train)
    #         img, labels = img.to(rank), labels.to(rank)
    #     except StopIteration:
    #         iter_train = iter(loader_train)
    #         img, labels = next(iter_train)
    #         img, labels = img.to(rank), labels.to(rank)
    #
    #     ddp_model.train()
    #     optimizer.zero_grad()
    #     outputs = ddp_model(img)
    #     loss = loss_fn(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #     loss_train.append(loss)
    #     conf_train.update(labels.cpu(), F.softmax(outputs, dim=1).argmax(1).cpu())
    #
    #     feat.append(outputs)
    #
    #     print(step, torch.stack(loss_train).mean())
    #
    #     if (step+1) % 10 == 0:
    #         acc = conf_train.getAccuracy()
    #         acc_per_cls = conf_train.getAccuracyPerClass()
    #
    #         print(acc)
    #         # wandb.log({f'acc/train{rank}': acc}, step=step+1)
    #         # print({cls: f"{val:.2f}" for cls, val in zip(range(len(acc_per_cls)), acc_per_cls)})
    #         conf_train.reset()
    #         loss_train = []
    #
    #         for idx, (img, labels) in enumerate(loader_train):
    #             print(idx)
    #             img, labels = img.to(rank), labels.to(rank)
    #             ddp_model.eval()
    #             outputs = ddp_model(img)
    #             loss = loss_fn(outputs, labels)
    #             loss_test.append(loss)
    #             conf_test.update(labels.cpu(), F.softmax(outputs, dim=1).argmax(1).cpu())
    #
    #         acc = conf_test.getAccuracy()
    #         acc_per_cls = conf_test.getAccuracyPerClass()
    #
    #
    #         print(step, torch.stack(loss_test).mean())
    #         print(acc)
    #         # print({cls: f"{val:.2f}" for cls, val in zip(range(len(acc_per_cls)), acc_per_cls)})
    #         conf_test.reset()
    #         loss_test = []
    #
    #     # print(outputs.argmax(1))
    #
    # feat = torch.cat(feat)
    # feat = torch.cat(GatherLayer.apply(feat), dim=0)
    # print(feat.size())

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





