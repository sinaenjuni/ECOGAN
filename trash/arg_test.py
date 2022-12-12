from argparse import ArgumentParser
import wandb


parser = ArgumentParser()
parser.add_argument("--gpus", nargs='+', type=int, default=7, required=True)

args = parser.parse_args()

print(parser)
print(args.gpus)

wandb.init(project="eval_cls", entity="sinaenjuni")
wandb.config.update(args)