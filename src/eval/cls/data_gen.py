import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import wandb
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from models import Generator
# from big_resnet import Generator
from torchvision.transforms import ToTensor, Compose, Normalize

import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm import  tqdm
from PIL import Image

batch_size = 128
rate = 4000
gen_dict = {1:rate, 2:rate, 3:rate, 4:rate, 5:rate, 6:rate, 7:rate, 8:rate, 9:rate}
label = torch.tensor(np.concatenate([[k] * v for k, v in gen_dict.items()]))
base_path = Path('/shared_hdd/sin/gen/')

api = wandb.Api()
target_name = []
for i in api.artifact_type(type_name='model', project='MYTEST1').collections():
    if 'GAN' in i.name:
        print(i.name)
        target_name.append(i.name)

for name in target_name:
    # print(name)
    ## load model from weight
    artifact = api.artifact(name=f'sinaenjuni/MYTEST1/{name}:v0', type='model')
    artifact_dir = artifact.download()
    ch = torch.load(Path(artifact_dir) / "model.ckpt", map_location=torch.device('cuda'))
    g_ch = {'.'.join(k.split('.')[1:]): w for k, w in ch['state_dict'].items() if 'G' in k}
    G = Generator(**ch['hyper_parameters']).cuda()
    ret = G.load_state_dict(g_ch)
    print(ret)

    pbar = tqdm(desc=name, iterable=range(len(label)))
    # gen dataset
    for idx in range(0, (len(label)//batch_size)+1):
        batch_label = label[idx * batch_size: (idx+1) * batch_size]

        z = torch.randn(batch_label.size(0), 128).cuda()
        imgs = G(z, batch_label.cuda())
        imgs = imgs.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)

        for idx_, (img, cls) in enumerate(zip(imgs, batch_label)):
            # print(idx * batch_size + idx_)
            save_path = base_path / str(rate) / name / str(cls.item())
            if not save_path.exists():
                save_path.mkdir(exist_ok=True, parents=True)
            if img.size(0) == 1:
                img = img.squeeze()
                img = Image.fromarray(img.numpy())
            else:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())

            img.save(save_path/f'{idx * batch_size + idx_}.png')
            pbar.update()


# save_path.mkdir(exist_ok=True, parents=True)
# save_image(img, save_path/'img.png')

class GenDataset(Dataset):
    def __init__(self, gen_dict = {1: 1000, 2: 100, 3: 1000, 4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000}, transform=None):
        # artifact = api.artifact(name=f'sinaenjuni/MYTEST1/ECOGAN(imb_FashionMNIST_512):v0', type='model')
        artifact = api.artifact(name=f'sinaenjuni/MYTEST1/model-3r7a3cr2:v0', type='model')
        artifact_dir = artifact.download()
        ch = torch.load(Path(artifact_dir) / "model.ckpt")
        g_ch = {'.'.join(k.split('.')[1:]): w for k, w in ch['state_dict'].items() if 'G' in k}
        self.G = Generator(**ch['hyper_parameters']).cuda()
        self.G.load_state_dict(g_ch)

        self.data = []
        self.label = torch.tensor(np.concatenate([[k] * v for k, v in gen_dict.items()]))
        for l_ in tqdm(range(0, len(self.label)//128, 1)):
            with torch.no_grad():
                label = self.label[l_:(l_+1) * 128].cuda()
                z = torch.randn(len(label), 128).cuda()
                self.data.append(self.G(z, label).cpu().numpy())
        self.transform = transform
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.label[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

dataset = GenDataset()
loader = DataLoader(dataset, batch_size=128, shuffle=True)
img, label = iter(loader).next()


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