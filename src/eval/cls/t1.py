import wandb
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
# from models import Generator
from big_resnet import Generator
from torchvision.transforms import ToTensor, Compose, Normalize

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np



class GenDataset(Dataset):
    def __init__(self, gne_dict, img_dim, latent_dim=128):
        api = wandb.Api()
        artifact = api.artifact(name=f'sinaenjuni/MYTEST1/ECOGAN(imb_FashionMNIST_512):v0', type='model')
        artifact = api.artifact(name=f'sinaenjuni/MYTEST1/model-2ni8hs6n:v0', type='model')
        artifact_dir = artifact.download()
        ch = torch.load(Path(artifact_dir) / "model.ckpt")
        g_ch = {'.'.join(k.split('.')[1:]): w for k, w in ch['state_dict'].items() if 'G' in k}
        G = Generator(**ch['hyper_parameters'])
        G.load_state_dict(g_ch)

        gen_dict = {1: 1000, 2: 100, 3: 1000, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000}
        label = np.concatenate([[k] * v for k, v in gen_dict.items()])

    def __len__(self):
        return len(label)

    def __getitem__(self, idx):


z = torch.rand(100, 128)
label = torch.arange(0, 10).repeat(10)
img = G(z, label)
print(img.size(), img.min(), img.max(), img.mean())

grid = make_grid(img, nrow=10, normalize=True)
plt.imshow(grid.permute(1,2,0).detach().cpu().numpy())
plt.show()

model.on_load_checkpoint(ch)


for i in api.artifact_type(type_name='model', project='MYTEST1').collections():
    # print(i, i.load())
    name = i.load()['name']
    print(name)
    artifact = api.artifact(name=f'sinaenjuni/MYTEST1/{name}:v0', type='model')
    print(artifact)
    artifact_dir = artifact.download()
    ch = torch.load(Path(artifact_dir) / "model.ckpt")
    {'.'.join(k.split('.')[1:]): w for k, w in ch['state_dict'].items() if 'G' in k}.keys()

dataset = ImageFolder('/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train', transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
loader = DataLoader(dataset, batch_size=128, shuffle=True)
img, label = iter(loader).next()
print(img.min(), img.max(), img.mean())



datasets = [["imb_MNIST", "ECOGAN(imb_MNIST_128)"],
["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_128)"],
["imb_CIFAR10", "ECOGAN(imb_CIFAR10_128)"],
["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_512)"],
["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_256)"],
["imb_MNIST", "ECOGAN(imb_MNIST_256)"],
["imb_CIFAR10", "ECOGAN(imb_CIFAR10_256)"],
["imb_MNIST", "ECOGAN(imb_MNIST_512)"],
["imb_CIFAR10", "ECOGAN(imb_CIFAR10_512)"],
["trained_imb_FashionMNIST", "BEGAN-GAN(pre-trained_imb_FashionMNIST)"],
["trained_imb_MNIST", "BEGAN-GAN(pre-trained_imb_MNIST)"],
["trained_imb_CIFAR10", "BEGAN-GAN(pre-trained_imb_CIFAR10)"],
["imb_FashionMNIST", "BEGAN-GAN(imb_FashionMNIST)"],
["imb_MNIST", "BEGAN-GAN(imb_MNIST)"],
["imb_CIFAR10", "BEGAN-GAN(imb_CIFAR10)"]]

class AugmentDataset(Dataset):
    def __init__(self):
        datasets = [["imb_MNIST", "ECOGAN(imb_MNIST_128)"],
                    ["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_128)"],
                    ["imb_CIFAR10", "ECOGAN(imb_CIFAR10_128)"],
                    ["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_512)"],
                    ["imb_FashionMNIST", "ECOGAN(imb_FashionMNIST_256)"],
                    ["imb_MNIST", "ECOGAN(imb_MNIST_256)"],
                    ["imb_CIFAR10", "ECOGAN(imb_CIFAR10_256)"],
                    ["imb_MNIST", "ECOGAN(imb_MNIST_512)"],
                    ["imb_CIFAR10", "ECOGAN(imb_CIFAR10_512)"],
                    ["trained_imb_FashionMNIST", "BEGAN-GAN(pre-trained_imb_FashionMNIST)"],
                    ["trained_imb_MNIST", "BEGAN-GAN(pre-trained_imb_MNIST)"],
                    ["trained_imb_CIFAR10", "BEGAN-GAN(pre-trained_imb_CIFAR10)"],
                    ["imb_FashionMNIST", "BEGAN-GAN(imb_FashionMNIST)"],
                    ["imb_MNIST", "BEGAN-GAN(imb_MNIST)"],
                    ["imb_CIFAR10", "BEGAN-GAN(imb_CIFAR10)"]]

        paths_train = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
                       'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/train',
                       'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/train'}

        api = wandb.Api()
        artifact = api.artifact(name=f'sinaenjuni/MYTEST1/ECOGAN(imb_MNIST_128):v0', type='model')
        project = api.project(name='MYTEST1')
        artifact_dir = artifact.download()


        num_aug = 1000
        dataset = ImageFolder('/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train')
        classes, num_per_classes = torch.unique(torch.tensor(dataset.targets), return_counts=True)
        num_original_data = len(dataset)

        num_aug_per_classes = torch.where(num_aug - num_per_classes > 0, num_aug - num_per_classes, 0)


        for i, v in enumerate(num_aug_per_classes):
            print(i, v.item())

        [[i] * v for i, v in enumerate(num_aug_per_classes)]