import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, PILToTensor, Normalize, ToTensor, Resize
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image


class DataModule_(pl.LightningDataModule):
    def __init__(self, data_name, batch_size=128, num_workers=4, pin_memory=True):
        super(DataModule_, self).__init__()
        self.batch_size = batch_size

        paths_train = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
                       'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/train',
                       'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/train'}
        paths_test = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val',
                      'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/val',
                      'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/val'}

        self.path_train = paths_train[data_name]
        self.path_test = paths_test[data_name]
        self.transforms = Compose([ToTensor(),
                                   Resize(64),
                                   Normalize(mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))])
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.dataset_train = ImageFolder(self.path_train, transform=self.transforms)
        if self.path_test is None:
            self.dataset_test = ImageFolder(self.path_train, transform=self.transforms)
        else:
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
    #
    # def test_dataloader(self):
    #     return DataLoader(self.valid_dataset, batch_size=self.batch_size)

class MergeDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        self.path1 = Path(path1)
        self.path2 = Path(path2)

        self.images = []
        self.targets = []

        self.get_data_paths(path1)
        self.get_data_paths(path2)

        self.transform = transform

    def get_data_paths(self, path):
        for path_ in Path(path).glob("*/*.JPEG"):
            self.images.append(path_)
            self.targets.append(int(path_.parts[-2][-1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label

if __name__ == "__main__":
    dm = DataModule_(data_name='imb_CIFAR10')
    dm.setup('fit')
    dm.setup('val')

    batch = iter(dm.train_dataloader()).next()
    img, label = batch
    print(img.shape, label.shape, img.min(), img.max())

    print(dm.train_dataloader())
    print(dm.val_dataloader())


    dataset = ImageFolder('/home/dblab/sin/save_files/refer/ebgan_cifar10', Compose([ToTensor(),
                                                                           Normalize(mean=(0.5, 0.5, 0.5),
                                                                                     std=(0.5, 0.5, 0.5))]))
