# import torch
# torch.arange(0, 10).repeat(10).view(10,10)
import matplotlib.pyplot as plt
#
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("--num_classes", type=int, default=10, required=False)
# parser.add_argument("--lr", type=float, default=0.0002, required=False)
# args = parser.parse_args("")
# args.lr
#
#
# import numpy as np
#
# np.mean([552.1089396500687,
# 544.1794644656591,
# 547.8769674228432,
# 544.8030220446703])
#
#
# inputs = torch.randn(10, 2)
#
# LSTM = torch.nn.LSTM(input_size=2, hidden_size=1)
# outputs, hidden = LSTM(inputs)
# outputs.size()
# hidden
#
# dataset = (torch.rand(100) * 10).to(torch.long)
# unique, counts = np.unique(dataset, return_counts=True)
# n_sample = len(dataset)
# weight = [n_sample / count for count in counts]
# weights = [weight[label] for label in dataset]
# sampler = WeightedRandomSampler(weights, 128 * 6 * 2000)
# loader = DataLoader(dataset, batch_size=128, sampler=sampler)
#
# torch.multinomial(torch.as_tensor(weights, dtype=torch.double), 2, replacement=True, generator=None).tolist()
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
from torchvision.datasets import ImageFolder

base_dir = Path('/shared_hdd/sin/dataset/Places/train_256_places365standard')
target_dir = Path('/shared_hdd/sin/dataset/MyPlacesLT/')
with open('/shared_hdd/sin/dataset/Places_LT/Places_LT_train.txt', 'r') as f:
        data_paths, targets = zip(*[[row.split(' ')[0], int(row.split(' ')[1])]
                                        for row in f.readlines()])

for data_path in tqdm(data_paths):
    source_path = base_dir / data_path
    img = Image.open(source_path)
    target_path = target_dir / '/'.join((source_path.parts[-2:]))
    if not target_path.exists():
        target_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(source_path, target_path)

dataset = ImageFolder('/shared_hdd/sin/dataset/MyPlacesLT/')


path = list(Path('/shared_hdd/sin/dataset/ImageNet_LT/').glob('*.txt'))

file = path[3].open()
print(file)
dpath, dlabel = zip(*[[row.split(' ')[0], int(row.split(' ')[1])] for row in file.readlines()])


path = list(Path('/shared_hdd/sin/dataset/ImageNet_LT/ImageNet_LT_open').glob('*'))
len(path)

num = np.unique(dlabel, return_counts=True)[1]
num.sort()



plt.bar(range(len(classes)), np.flip(np.sort(num_classes)))
plt.show()


for path in tqdm(dpath):
    img = Image.open(img_path)
    # plt.imshow(img)
    # plt.show()
    # file = img_path.open()
    # print(file.)

for i in files:
    print(i)

import pickle
with open('/shared_hdd/sin/dataset/ImageNet64/train_data_batch_1', 'rb') as fo:
    dict = pickle.load(fo)





from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
class PlacesDataset(Dataset):
    def __init__(self, is_train=True, batch_size=128, steps=2000, transform=None):
        self.base_path = Path('/shared_hdd/sin/dataset/Places/train_256_places365standard')
        with open('/shared_hdd/sin/dataset/Places_LT/Places_LT_train.txt', 'r') as f:
            self.data, self.targets = zip(*[[row.split(' ')[0], int(row.split(' ')[1])]
                                            for row in f.readlines()])
        self.classes, self.num_classes = np.unique(self.targets, return_counts=True)
        self.num_ori = len(self.targets)
        self.batch_size = batch_size
        self.steps = steps
        self.is_train = is_train
        # print(len(dlabel), len(classes), num_classes.min(), num_classes.max())
        self.transform = transform
    def __len__(self):
        if self.is_train:
            return self.batch_size * 6 * self.steps
        else:
            return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx % self.num_ori)
        img_path = self.base_path / self.data[idx]
        img = Image.open(img_path)
        label = self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


dataset = PlacesDataset(is_train=False, batch_size=128, steps=2000, transform=Compose(
                                                            [ToTensor(),
                                                            Resize(64),
                                                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
loader = DataLoader(dataset, batch_size=128)
img, label = iter(loader).next()

from torchvision.datasets import ImageFolder
dataset = Im


import importlib
m = importlib.import_module('utils/dataset', 'PlacesLTDataset')
m.PlacesLTDataset