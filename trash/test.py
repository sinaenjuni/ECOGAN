import torch
torch.arange(0, 10).repeat(10).view(10,10)



from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--num_classes", type=int, default=10, required=False)
parser.add_argument("--lr", type=float, default=0.0002, required=False)
args = parser.parse_args("")
args.lr


import numpy as np

np.mean([552.1089396500687,
544.1794644656591,
547.8769674228432,
544.8030220446703])


inputs = torch.randn(10, 2)

LSTM = torch.nn.LSTM(input_size=2, hidden_size=1)
outputs, hidden = LSTM(inputs)
outputs.size()
hidden