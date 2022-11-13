import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, PILToTensor, Normalize, ToTensor, Resize, Grayscale, Lambda
from torch.utils.data import Dataset, WeightedRandomSampler, DistributedSampler
import numpy as np
from pathlib import Path
from PIL import Image
from utils.sampler import DistributedWeightedSampler

paths_train = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
               'imb_MNIST': '/shared_hdd/sin/save_files/imb_MNIST/train',
               'imb_FashionMNIST': '/shared_hdd/sin/save_files/imb_FashionMNIST/train'}
paths_test = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val',
              'imb_MNIST': '/shared_hdd/sin/save_files/imb_MNIST/val',
              'imb_FashionMNIST': '/shared_hdd/sin/save_files/imb_FashionMNIST/val'}
img_dims = {'imb_CIFAR10': 3, 'imb_MNIST': 1, 'imb_FashionMNIST': 1}
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class DataModule_(pl.LightningDataModule):
    def __init__(self, data_name, img_size, img_dim, is_sampling, batch_size=128, steps=2000, num_workers=4, pin_memory=True):
        super(DataModule_, self).__init__()
        self.batch_size = batch_size
        self.steps = steps

        self.data_name = data_name
        self.img_size = img_size
        self.img_dim = img_dim
        self.is_sampling = is_sampling
        self.path_train = paths_train[self.data_name]
        self.path_test = paths_test[self.data_name]
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
        self.dataset_train = MyImageFolderDataset(self.path_train, batch_size=self.batch_size, steps=self.steps, transform=self.transforms)
        if self.is_sampling:
            unique, counts = np.unique(self.dataset_train.targets, return_counts=True)
            # n_sample = len(self.dataset_train.targets)
            n_sample = len(self.dataset_train)
            weight = [n_sample / count for count in counts]
            weights = [weight[label] for label in self.dataset_train.targets]
            # self.sampler = WeightedRandomSampler(weights, n_sample)
            self.sampler = DistributedWeightedSampler(
                dataset=self.dataset_train, weights=weights)

        self.dataset_val = ImageFolder(self.path_test, transform=self.transforms)


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

class Eval_gen_cls_dataset(pl.LightningDataModule):
    def __init__(self, gen_path, img_size, batch_size=128, num_workers=4, pin_memory=True, *args, **kwargs):
        super(Eval_gen_cls_dataset, self).__init__()

        data_names = ['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST']
        self.sel_name = [data_name for data_name in data_names if data_name in str(gen_path)][0]


        self.data = []
        self.targets = []
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dim = img_dims[self.sel_name]
        self.path_train0 = paths_train[self.sel_name]
        self.path_train1 = gen_path
        self.path_test = paths_test[self.sel_name]

        self.transforms = Compose([ToTensor(),
                                   Grayscale() if self.img_dim == 1 else Lambda(lambda x: x),
                                   Resize(self.img_size),
                                   Normalize(mean=[0.5] * self.img_dim,
                                             std=[0.5] * self.img_dim)
                                    ])
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.dataset_train = MergeDataset(self.path_train0, self.path_train1, transform=self.transforms)
        self.dataset_test = ImageFolder(self.path_test, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
class MergeDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        self.path1 = Path(path1)
        self.path2 = Path(path2)
        self.images = []
        self.targets = []
        self.get_data(path1)
        self.get_data(path2)
        self.transform = transform

    def get_data(self, path:str):
        path = Path(path)
        for img_path in path.glob("*/*.JPEG"):
            self.images.append(img_path)
            self.targets.append(int(img_path.parts[-2][-1]))
        for img_path in path.glob("*/*.png"):
            self.images.append(img_path)
            self.targets.append(int(img_path.parts[-2][-1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label
class MyImageFolderDataset(Dataset):
    def __init__(self, path, batch_size=128, steps=2000, transform=None):
        self.path = Path(path)
        self.data = []
        self.targets = []
        self.batch_size = batch_size
        self.steps = steps
        self.get_data()
        self.num_ori = len(self.data)
        self.transform = transform
    def get_data(self):
        for path_img in self.path.glob("*/*"):
            if path_img.suffix.lower() in IMG_EXTENSIONS:
                self.data.append(path_img)
                self.targets.append(int(path_img.parts[-2][-1]))
        # print(f"Fine number of {len(self.data)}")
    def __len__(self):
        return self.batch_size * 6 * self.steps
    def __getitem__(self, idx):
        idx = int(idx % self.num_ori)
        image = Image.open(self.data[idx])
        target = self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, target


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