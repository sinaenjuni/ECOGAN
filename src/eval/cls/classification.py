
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

EBGAN_gened_data_path = "/home/dblab/git/EBGAN/save_files/EBGAN_restore_data"
EBGAN_gened_data = ImageFolder(EBGAN_gened_data_path, transform=Compose([ ToTensor(),
                                                                     Normalize((0.5, 0.5, 0.5),
                                                                              (0.5, 0.5, 0.5))]))

ori_imb_data_path = "/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train"
ori_img_dataset = ImageFolder(ori_imb_data_path, transform=Compose([ ToTensor(),
                                                                     Normalize((0.5, 0.5, 0.5),
                                                                              (0.5, 0.5, 0.5))]))

concat_dataset = ConcatDataset([ori_img_dataset, EBGAN_gened_data])

ori_img_loader = DataLoader(ori_img_dataset, batch_size=64)

batch = iter(ori_img_dataset).__next__()


for i, batch in enumerate(concat_dataset):
    print(i)


