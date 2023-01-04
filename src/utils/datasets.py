import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, PILToTensor, Normalize, ToTensor, Resize, Grayscale, Lambda
from torch.utils.data import Dataset, WeightedRandomSampler, DistributedSampler
import numpy as np
from pathlib import Path
from PIL import Image
from utils.sampler import DistributedWeightedSampler
from tqdm import tqdm
import importlib

class MyDataset(Dataset):
    def __init__(self, is_extension:bool, batch_size:int, steps:int, transform=None):
        self.is_extension:bool = is_extension
        self.batch_size:int = batch_size
        self.steps:int = steps
        self.transform = transform

        self.data:list
        self.targets:list

        self.num_ori:int
        self.num_classes:int
        self.classes:list

    def parse_data_path(self, data:list, targets:list):
        assert len(data) == len(targets)
        self.data = data
        self.targets = targets
        self.classes, self.num_classes = np.unique(targets, return_counts=True)
        self.num_ori = len(targets)

    def __len__(self):
        if self.is_extension:
            return self.batch_size * 6 * self.steps
        else:
            return self.num_ori

    def __getitem__(self, idx):
        assert self.num_ori != 0, "num_ori is zero"
        idx = idx % self.num_ori
        image = Image.open(self.data[idx]).convert('RGB')
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
            
        # if image.size(0) == 1:
            # image = image.repeat(3,1,1)
        # Lambda(lambda x : x.repeat(3, 1, 1) if x.size(0) == 1 else x),
        
        return image, target

class CIFAR10_LT(MyDataset):
    def __init__(self, is_train=True, is_extension=True, batch_size=128, steps=2000, transform=None):
        super(CIFAR10_LT, self).__init__(is_extension=is_extension, batch_size=batch_size, steps=steps, transform=transform)
        if is_train:
            path = '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train'
        else:
            path = '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val'

        dataset = ImageFolder(path)
        data, targets = zip(*[(row[0], row[1]) for row in dataset.imgs])
        self.parse_data_path(data, targets)

class MNIST_LT(MyDataset):
    def __init__(self, is_train=True, is_extension=True, batch_size=128, steps=2000, transform=None):
        super(MNIST_LT, self).__init__(is_extension=is_extension, batch_size=batch_size, steps=steps, transform=transform)
        if is_train:
            path = '/shared_hdd/sin/save_files/imb_MNIST/train'
        else:
            path = '/shared_hdd/sin/save_files/imb_MNIST/val'
        dataset = ImageFolder(path)
        data, targets = zip(*[(row[0], row[1]) for row in dataset.imgs])
        self.parse_data_path(data, targets)

class FashionMNIST_LT(MyDataset):
    def __init__(self, is_train=True, is_extension=True, batch_size=128, steps=2000, transform=None):
        super(FashionMNIST_LT, self).__init__(is_extension=is_extension, batch_size=batch_size, steps=steps, transform=transform)
        if is_train:
            path = '/shared_hdd/sin/save_files/imb_FashionMNIST/train'
        else:
            path = '/shared_hdd/sin/save_files/imb_FashionMNIST/val'
        dataset = ImageFolder(path)
        data, targets = zip(*[(row[0], row[1]) for row in dataset.imgs])
        self.parse_data_path(data, targets)

class Places_LT(MyDataset):
    def __init__(self, is_train=True, is_extension=True, batch_size=128, steps=2000, transform=None):
        super(Places_LT, self).__init__(is_extension=is_extension, batch_size=batch_size, steps=steps, transform=transform)
        if is_train:
            base_path = Path('/shared_hdd/sin/dataset/Places/train_256_places365standard')
            with open('/shared_hdd/sin/dataset/Places_LT/Places_LT_train.txt', 'r') as f:
                data, targets = zip(*[(base_path / row.split(' ')[0], int(row.split(' ')[1])) for row in f.readlines()])
        else:
            base_path = Path('/shared_hdd/sin/dataset/Places/val_256/')
            with open('/shared_hdd/sin/dataset/Places_LT/Places_LT_test.txt', 'r') as f:
                data, targets = zip(*[(base_path / row.split(' ')[0], int(row.split(' ')[1])) for row in f.readlines()])

        self.parse_data_path(data, targets)





base_path = Path('/shared_hdd/sin/dataset/Places/train_256_places365standard')
with open('/shared_hdd/sin/dataset/ImageNet_LT/ImageNet_LT_train.txt', 'r') as f:
    data, targets = zip(*[(base_path / row.split(' ')[0], int(row.split(' ')[1])) for row in f.readlines()])


# dataset = Places_LT(is_train=True, is_sampling=False, transform=Compose(
#                                                                 [ToTensor(),
#                                                                  Resize(64),
#                                                                  Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]))
#
# dataset.num_classes
# loader = DataLoader(dataset, batch_size=128, shuffle=True)
# for i in tqdm(loader):
#     pass
# img, label = iter(loader).next()
#
# import importlib
# m = importlib.import_module('utils.dataset')
# cls = getattr(m, 'PlacesLTDataset')()
# cls.num_ori

# dataset = getattr(m ,data_name)(is_train=True, is_sampling=True, transform=Compose(
#                                                                 [ToTensor(),
#                                                                  Resize(64),
#                                                                  Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]))
# loader = DataLoader(dataset, batch_size=128, shuffle=True)
# for i in tqdm(loader):
#     pass
# img, label = iter(loader).next()
class DataModule_(pl.LightningDataModule):
    def __init__(self,
                 data_name:str,
                 img_size:int,
                 img_dim:int,
                 is_extension:bool,
                 is_sampling:bool,
                 batch_size:int=128,
                 steps:int=2000,
                 num_workers=4,
                 pin_memory=True):
        super(DataModule_, self).__init__()
        self.batch_size = batch_size
        self.steps = steps

        self.data_name = data_name
        self.img_size = img_size
        self.img_dim = img_dim
        self.is_sampling = is_sampling
        self.is_extension = is_extension

        self.transforms = Compose([ToTensor(),
                                   # Grayscale() if self.img_dim == 1 else Lambda(lambda x: x),
                                   Resize(self.img_size),
                                   Normalize(mean=[0.5] * self.img_dim,
                                             std=[0.5] * self.img_dim)
                                    ])
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        # self.dataset_train = ImageFolder(self.path_train, transform=self.transforms)
        m = importlib.import_module('utils.datasets')
        self.dataset_train = getattr(m, self.data_name)(
                                     is_train=True,
                                     is_extension=self.is_extension,
                                     batch_size=self.batch_size,
                                     steps=self.steps,
                                     transform=self.transforms)

        if self.is_sampling:
            # unique, counts = np.unique(self.dataset_train.targets, return_counts=True)
            unique, counts = self.dataset_train.classes, self.dataset_train.num_classes
            # n_sample = len(self.dataset_train.targets)
            # n_sample = len(self.dataset_train)
            n_sample = self.dataset_train.num_ori
            weight = [n_sample / count for count in counts]
            weights = [weight[label] for label in self.dataset_train.targets]
            self.sampler = WeightedRandomSampler(weights, n_sample)
            # self.sampler = DistributedWeightedSampler(
            #     dataset=self.dataset_train, weights=weights)

        self.dataset_val = getattr(m, self.data_name)(
                                     is_train=True,
                                     is_extension=False,
                                     batch_size=self.batch_size,
                                     steps=self.steps,
                                     transform=self.transforms)


    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          sampler=self.sampler if self.is_sampling else None,
                          shuffle=False if self.is_sampling else True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

# class Eval_gen_cls_dataset(pl.LightningDataModule):
#     def __init__(self, gen_path, img_size, batch_size=128, num_workers=4, pin_memory=True, *args, **kwargs):
#         super(Eval_gen_cls_dataset, self).__init__()
#
#         data_names = ['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST']
#         self.sel_name = [data_name for data_name in data_names if data_name in str(gen_path)][0]
#
#
#         self.data = []
#         self.targets = []
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.img_dim = img_dims[self.sel_name]
#         self.path_train0 = paths_train[self.sel_name]
#         self.path_train1 = gen_path
#         self.path_test = paths_test[self.sel_name]
#
#         self.transforms = Compose([ToTensor(),
#                                    Grayscale() if self.img_dim == 1 else Lambda(lambda x: x),
#                                    Resize(self.img_size),
#                                    Normalize(mean=[0.5] * self.img_dim,
#                                              std=[0.5] * self.img_dim)
#                                     ])
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#
#     def setup(self, stage):
#         self.dataset_train = MergeDataset(self.path_train0, self.path_train1, transform=self.transforms)
#         self.dataset_test = ImageFolder(self.path_test, transform=self.transforms)
#
#     def train_dataloader(self):
#         return DataLoader(self.dataset_train,
#                           batch_size=self.batch_size,
#                           shuffle=True,
#                           num_workers=self.num_workers,
#                           pin_memory=True)
#
#     def val_dataloader(self):
#         return DataLoader(self.dataset_test,
#                           batch_size=self.batch_size,
#                           shuffle=False,
#                           num_workers=self.num_workers,
#                           pin_memory=True)
# class MergeDataset(Dataset):
#     def __init__(self, path1, path2, transform=None):
#         self.path1 = Path(path1)
#         self.path2 = Path(path2)
#         self.images = []
#         self.targets = []
#         self.get_data(path1)
#         self.get_data(path2)
#         self.transform = transform
#
#     def get_data(self, path:str):
#         path = Path(path)
#         for img_path in path.glob("*/*.JPEG"):
#             self.images.append(img_path)
#             self.targets.append(int(img_path.parts[-2][-1]))
#         for img_path in path.glob("*/*.png"):
#             self.images.append(img_path)
#             self.targets.append(int(img_path.parts[-2][-1]))
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.images[idx])
#         if self.transform is not None:
#             img = self.transform(img)
#
#         label = self.targets[idx]
#         return img, label
# class MyImageFolderDataset(Dataset):
#     def __init__(self, path, batch_size=128, steps=2000, transform=None):
#         self.path = Path(path)
#         self.data = []
#         self.targets = []
#         self.batch_size = batch_size
#         self.steps = steps
#         self.get_data()
#         self.num_ori = len(self.data)
#         self.transform = transform
#     def get_data(self):
#         for path_img in self.path.glob("*/*"):
#             if path_img.suffix.lower() in IMG_EXTENSIONS:
#                 self.data.append(path_img)
#                 self.targets.append(int(path_img.parts[-2][-1]))
#         # print(f"Fine number of {len(self.data)}")
#     def __len__(self):
#         return self.batch_size * 6 * self.steps
#     def __getitem__(self, idx):
#         idx = int(idx % self.num_ori)
#         image = Image.open(self.data[idx])
#         target = self.targets[idx]
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, target




# dataset = MyImageFolderDataset(paths_test['imb_CIFAR10'], transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
# dataset = MyImageFolderDataset(paths_train['imb_CIFAR10'], transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

# dist = []
# dist_sm = []
# from tqdm import tqdm
# for img, label in tqdm(loader):
#     dist.append(label)
    # dist_sm.append(label)
# np.unique(np.concatenate(dist), return_counts=True)
# np.unique(np.concatenate(dist_sm), return_counts=True)
# loader.dataset.num_ori
# img, label = iter(loader).next()
# len(dataset)
# len(loader)
# # dataset[1]
#
# #
# for idx, (image, label) in enumerate(loader):
#     print(idx, label.size())
#


# if __name__ == "__main__":
    # dm = DataModule_(data_name='imb_CIFAR10', is_sampling=False, img_size = 64, img_dim=3)
    # dm.setup('fit')
    # np.unique(dm.dataset_train.targets, return_counts=True)
    #
    # for i, (img, label) in enumerate(dm.train_dataloader()):
    #     print(i, label.size())
    #
    # dm = DataModule_(data_name='imb_MNIST', is_sampling=False, img_size = 64, img_dim=1)
    # dm.setup('fit')
    # np.unique(dm.dataset_train.targets, return_counts=True)
    #
    # dm = DataModule_(data_name='imb_FashionMNIST', is_sampling=False, img_size = 64, img_dim=1)
    # dm.setup('fit')
    # np.unique(dm.dataset_train.targets, return_counts=True)
    #
    # dm.setup('fit')
    # dm.setup('val')
    #
    # batch = iter(dm.train_dataloader()).next()
    # img, label = batch
    # print(img.shape, label.shape, img.min(), img.max())

    # print(dm.train_dataloader())
    # print(dm.val_dataloader())
    #
    #
    # dataset = ImageFolder('/home/dblab/sin/save_files/refer/ebgan_cifar10', Compose([ToTensor(),
    #                                                                        Normalize(mean=(0.5, 0.5, 0.5),
    #                                                                                  std=(0.5, 0.5, 0.5))]))

    # count = []
    # for epoch in range(100):
    #     for batch in dm.train_dataloader():
    #         data, label = batch
    #         # print(label)
    #         count.append(label.numpy())
    #
    # np.unique(np.concatenate(count), return_counts=True)

    # import torch.distributed as dist
    # import os
    #
    # img_dim = 3
    # dataset = MyImageFolderDataset(paths_train['imb_CIFAR10'],
    #                                transform=Compose([ToTensor(),
    #                                                   Resize(64),
    #                                                   Normalize(mean=[0.5] * img_dim,
    #                                                             std=[0.5] * img_dim)
    #                                                   ]))
    #
    # unique, counts = np.unique(dataset.targets, return_counts=True)
    # n_sample = len(dataset.targets)
    # weight = [n_sample / count for count in counts]
    # weights = [weight[label] for label in dataset.targets]
    # # sampler = WeightedRandomSampler(weights, 128 * 6 * 2000)
    #
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    # dist.init_process_group("gloo", rank=0, world_size=2)
    #
    # sampler = DistributedWeightedSampler(dataset=dataset, weights=weights)
    # loader = DataLoader(dataset, batch_size=128, sampler=sampler)
    #
    # dist_sm = []
    # from tqdm import tqdm
    # for img, label in tqdm(loader):
    #     dist.append(label)
    # dist_sm.append(label)
    # print(np.unique(np.concatenate(dist), return_counts=True))

