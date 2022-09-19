# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/config.py

from itertools import chain
import json
import os
import random
import sys
import yaml

import torch
import torch.nn as nn
import ops





class Configurations(object):
    def __init__(self):

        self.define_modules()


    def define_modules(self):


        return self.MODULES


