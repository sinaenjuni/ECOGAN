import torch
from torchvision.datasets import ImageFolder

from src.model import Generator
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def get_target_weight(state_dict, keyword):
    return {'.'.join(k.split('.')[1:]): v for k, v in state_dict['state_dict'].items() if keyword in k}

def denorm(img):
    img = img * 0.5 + 0.5
    img = img * 255.
    img = img.astype(np.uint8)
    return img


dataset = ImageFolder('/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train')
classes, counts = torch.unique(torch.tensor(dataset.targets), return_counts=True)
num_make_data = counts.max() - counts


paths = {'EBGAN': '/shared_hdd/sin/save_files/EBGAN/MYGAN/EBGAN/checkpoints/EBGAN-epoch=91-fid=350.3489733048592.ckpt',
         'EBGAN_pre-trained': '/shared_hdd/sin/save_files/EBGAN/MYGAN/EBGAN_pre-trained/checkpoints/EBGAN-epoch=18-fid=308.3734377469981.ckpt',
         'ECOGAN': '/shared_hdd/sin/save_files/EBGAN/MYGAN/ECOGAN/checkpoints/EBGAN-epoch=59-fid=304.000065739339.ckpt'}
save_path = Path('/shared_hdd/sin/save_files/gened_img/')

batch_size= 128
latent_dim= 128
img_dim= 3
num_class= 10
model = Generator(img_dim=img_dim, latent_dim=latent_dim, num_class=num_class).cuda()
model.eval()


EBGAN = torch.load(paths['EBGAN'])
EBGAN = get_target_weight(EBGAN, 'G')
model.load_state_dict(EBGAN)

EBGAN_pre_trained = torch.load(paths['EBGAN_pre-trained'])
EBGAN_pre_trained = get_target_weight(EBGAN_pre_trained, 'G')
model.load_state_dict(EBGAN_pre_trained)

ECOGAN = torch.load(paths['ECOGAN'])
ECOGAN = get_target_weight(ECOGAN, 'G')
model.load_state_dict(ECOGAN)

num_make_data % batch_size
num_make_data // batch_size * batch_size


for cls, num_max in zip(classes, num_make_data):
    path = save_path / 'ECOGAN' / str(cls.item())
    path.mkdir(parents=True, exist_ok=True)

    for iter in tqdm(range(0, num_max, 1), desc=f'{cls.item()}'):
        z = torch.randn(1, latent_dim).cuda()
        label = (torch.ones(1, dtype=torch.long) * cls).cuda()
        img = model(z, label)

        img = img.squeeze().permute(1,2,0)
        img = denorm(img.detach().cpu().numpy())
        img = Image.fromarray(img)
        img.save(path / f"{cls}_{iter}.JPEG")



