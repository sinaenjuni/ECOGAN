import torch
torch.arange(0, 10).repeat(10).view(10,10)



from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--num_classes", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=0.0002, required=False)
args = parser.parse_args("")
args.lr