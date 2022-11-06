import wandb
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder



api = wandb.Api()
artifact = api.artifact(name=f'sinaenjuni/MYTEST1/ECOGAN(imb_MNIST_128):v0', type='model')
project = api.project(name='MYTEST1')
artifact_dir = artifact.download()

ch = Path(artifact_dir) / "model.ckpt"
ch = torch.load(Path(artifact_dir) / "model.ckpt")
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



class AugmentDataset(Dataset):
    def __init__(self):
        paths_train = {'imb_CIFAR10': '/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
                       'imb_MNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_MNIST/train',
                       'imb_FashionMNIST': '/home/shared_hdd/shared_hdd/sin/save_files/imb_FashionMNIST/train'}

        num_aug = 1000
        dataset = ImageFolder('/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train')
        classes, num_per_classes = torch.unique(torch.tensor(dataset.targets), return_counts=True)
        num_original_data = len(dataset)

        num_aug_per_classes = torch.where(num_aug - num_per_classes > 0, num_aug - num_per_classes, 0)


        for i, v in enumerate(num_aug_per_classes):
            print(i, v.item())

        [[i] * v for i, v in enumerate(num_aug_per_classes)]