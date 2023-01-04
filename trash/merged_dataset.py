import h5py
from pathlib import Path
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Resize, Lambda


data_info=[
('CIFAR10_LT', '1r08m7xe'),
('CIFAR10_LT', '3rczol1e'),
('CIFAR10_LT', '3u337fao'),
('CIFAR10_LT', '6gem8lca'),
('FashionMNIST_LT', '3p2vbr49'),
('FashionMNIST_LT', 'i2mijyme'),
('FashionMNIST_LT', 'iitr6wnc'),
('FashionMNIST_LT', 'vpgfgriu'),
('Places_LT', '1qlor8tk'),
('Places_LT', '1vdnnadg'),
('Places_LT', '23rk0cyl'),
('Places_LT', '2bi04amy')]



# base_path = Path('/shared_hdd/sin/gen')
# h5_gen = h5py.File(base_path / 'gen.h5py', 'r')
# h5_ori = h5py.File(base_path / 'ori.h5py', 'r')
#
# data_ori = h5_ori['CIFAR10_LT']['train']['data']
# data_gen = h5_gen['gened_data']['CIFAR10_LT']['3rczol1e']['data'].shape
#
#
# for data_name, data in h5_ori.items():
#     print(data_name)
#     for data_type, data2 in data.items():
#         print(data_type, data2['data'].shape, data2['targets'].shape)
#
#
# for data_name, data_set in h5_gen.items():
#     # print(data_name)
#     for data_type, data in data_set.items():
#         print(f"('{data_name}', '{data_type}')")
#         print(data['data'].shape)
#         print(np.transpose(data['data'], (0,2,3,1)).shape)



class imbalanced_dataset(Dataset):
    def __init__(self, data_info, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        base_path = Path('/shared_hdd/sin/gen/')

        self.h5_ori = h5py.File(base_path / 'ori.h5py', 'r')

        if self.is_train:
            self.data_ori = self.h5_ori[data_info]['train']['data']
            self.targets_ori = self.h5_ori[data_info]['train']['targets']
        else:
            self.data_ori = self.h5_ori[data_info]['test']['data']
            self.targets_ori = self.h5_ori[data_info]['test']['targets']

    def __len__(self):
        if self.is_train:
            return 100000
        else:
            return len(self.data_ori)
    def __getitem__(self, idx):
        idx = int(idx % len(self.data_ori))
        img = self.data_ori[idx]
        target = self.targets_ori[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class merged_dataset(Dataset):
    def __init__(self, data_info, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        data_name, data_id = data_info
        base_path = Path('/shared_hdd/sin/gen/')

        self.h5_ori = h5py.File(base_path / 'ori.h5py', 'r')
        self.h5_gen = h5py.File(base_path / 'gen.h5py', 'r')

        if self.is_train:
            self.data_ori = self.h5_ori[data_name]['train']['data']
            self.targets_ori = self.h5_ori[data_name]['train']['targets']

            self.data_gen = self.h5_gen[data_name][data_id]['data']
            self.targets_gen = self.h5_gen[data_name][data_id]['targets']
        else:
            self.data_ori = self.h5_ori[data_name]['test']['data']
            self.targets_ori = self.h5_ori[data_name]['test']['targets']

    def __len__(self):
        if self.is_train:
            return 128*100000
        else:
            return len(self.data_ori)

    def __getitem__(self, idx):
        if idx < len(self.data_ori):
            img = self.data_ori[idx]
            target = self.targets_ori[idx]
        else:
            idx = idx - len(self.data_ori)
            img = self.data_gen[idx]
            target = self.targets_gen[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target



transform = Compose([ToTensor(),
                         Resize(64),
                         Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                         Normalize([0.5, 0.5, 0.5],
                                   [0.5, 0.5, 0.5])
                         ])

dataset = merged_dataset(data_info=data_info[10], is_train=True, transform=transform)
dataset = imbalanced_dataset(data_info='CIFAR10_LT', is_train=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

print(len(loader))

temp = []
ap = temp.append
for img, target in tqdm(loader):
    ap(target)



import matplotlib.pyplot as plt
plt.imshow((img[0].permute(1,2,0)  + 0.5 ) * 0.5)
plt.show()

for data_info_ in data_info:
    print(data_info_)
    dataset = merged_dataset(data_info=data_info_, is_train=True, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)


    for i in tqdm(loader):
        pass



h5_ori.keys()

data_gen = h5_gen['gened_data']['CIFAR10_LT'].keys()
classes, num_classes = np.unique(cifar10_gen, return_counts=True)
num_classes



