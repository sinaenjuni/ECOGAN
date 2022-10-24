import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, PILToTensor, Normalize, ToTensor

class DataModule_(pl.LightningDataModule):
    def __init__(self, path_train, batch_size, num_workers=4, pin_memory=True):
        super(DataModule_, self).__init__()
        self.batch_size = batch_size
        self.path_train = path_train
        self.transforms = Compose([ToTensor(),
                                   Normalize(mean=(0.5, 0.5, 0.5),
                                             std=(0.5, 0.5, 0.5))])
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.dataset_train = ImageFolder(self.path_train, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.valid_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dm = DataModule_(path_train='/home/dblab/sin/save_files/refer/ebgan_cifar10', batch_size=128)
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

    type(dataset.targets)
    type(dataset.imgs)
    torch.where(dataset.targets == 2)



