import os
import random
import numpy as np
from argparse import ArgumentTypeError
from datetime import datetime
import torch
import torch.distributed as dist
from argparse import ArgumentParser


def load_configs_init():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gpus", nargs='+', default=[1, 2], type=int, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--epoch_ae", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    
    parser.add_argument("--logger", type=str2bool, required=True)
    parser.add_argument("--img_dim", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--beta1", type=float, required=True)
    parser.add_argument("--beta2", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--is_save", type=str2bool, required=True)
    
    args = parser.parse_args()
    # cfgs = vars(args)

    return args



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def elapsed_time(start_time):
    now = datetime.now()
    elapsed = now - start_time
    return str(elapsed).split(".")[0]  # remove milliseconds


class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
    
    
def cleanup():
    dist.destroy_process_group()
    
def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)    
    

def setup(rank, world_size, backend="nccl"):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'

    # initialize the process group
    # dist.init_process_group(backend=backend,
    #                         rank=rank,
    #                         # init_method='env://',
    #                         world_size=world_size)
    dist.init_process_group(backend=backend,
                                init_method="env://",
                                rank=rank,
                                world_size=world_size)

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup_for_distributed(rank == 0) # 특정 rank에서만 print 허용
    
    

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
    
    
def print_h5py(f, p=0):
    for n, v in f.items():
        print('\t'*p+n)
        if hasattr(v, 'items'):
            print_h5py(v, p+1)
        else:
            print('\t'*p+str(v.shape))
