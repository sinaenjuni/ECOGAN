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