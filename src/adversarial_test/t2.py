import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
# from src.adversarial_test.model import Encoder, Decoder
from torch.nn.utils import spectral_norm
from tqdm import tqdm
from src.metric import fid, ins
from src.metric.inception_net import EvalModel
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from src.EBGAN.model_gan import Generator, Discriminator

img_dim = 3
num_classes = 10
latent_dim = 128
batch_size = 128
lr_g = 0.0002
lr_d = 0.0002
num_epoch = 50

# dataset = FashionMNIST(root='/shared_hdd/sin/dataset', download=False, train=True,
#                 transform=Compose([ToTensor(),
#                                    # Resize(64),
#                                    Normalize([0.5],
#                                              [0.5])]))

dataset = CIFAR10(root='/shared_hdd/sin/dataset', download=False, train=True,
                transform=Compose([ToTensor(),
                                   Resize(64),
                                   Normalize([0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.5])]))

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
img, label = iter(data_loader).__next__()

grid = make_grid(img, normalize=True)
plt.imshow(grid.permute(1,2,0).cpu())
plt.show()

# eval_model = EvalModel(device=torch.device('cuda'))

# mu, sigma, label_list = fid.calculate_mu_sigma(data_loader, eval_model, True)

# np.savez('/shared_hdd/sin/save_files/fashionMNIST.npz', mu=mu, sigma=sigma, label_list=label_list)
# a = np.load('/shared_hdd/sin/save_files/fashionMNIST.npz')
# fid.frechet_inception_distance()

# m_scroe, std, label_list = ins.inception_score(data_loader, eval_model, quantize=True)

G = Generator(img_dim, latent_dim, num_classes).cuda()
D = Discriminator(img_dim, latent_dim, num_classes).cuda()


# latent = torch.randn(batch_size, latent_dim).cuda()
# img_fake = decoder(latent).detach()
# adv_output = encoder(img.cuda())
# print(img_fake.size())

optimizer_g = torch.optim.Adam(params=G.parameters(), lr=lr_g)
optimizer_d = torch.optim.Adam(params=D.parameters(), lr=lr_d)

bce_loss = nn.BCELoss()

for epoch in range(num_epoch):
    loader_tqdm = tqdm(data_loader)
    for img, label in loader_tqdm:
        img_real, label = img.cuda(), label.cuda()
        latent = torch.randn(img_real.size(0), latent_dim).cuda()

        y_real_, y_fake_ = torch.ones(img_real.size(0), 1).cuda(), torch.zeros(img_real.size(0), 1).cuda()

        optimizer_d.zero_grad()

        img_fake = G(latent, label).detach()
        adv_real = D(img_real, label)
        adv_fake = D(img_fake, label)

        # d_loss = get_loss(adv_real, 1).mean() + get_loss(adv_fake, 0).mean()
        # d_loss = -1 * torch.log(adv_real).mean() + -1 * torch.log(1-adv_fake).mean()
        d_loss = bce_loss(adv_real, y_real_) + bce_loss(adv_fake, y_fake_)

        d_loss.backward()
        optimizer_d.step()



        optimizer_g.zero_grad()

        img_fake = G(latent, label)
        adv_fake = D(img_fake, label)

        # g_loss = get_loss(adv_fake, 1).mean()
        # g_loss = -1 * torch.log(adv_fake).mean()
        g_loss = bce_loss(adv_fake, y_real_)
        g_loss.backward()
        optimizer_g.step()

        loader_tqdm.set_description(f'epoch {epoch}')
        loader_tqdm.set_postfix({'d_loss':d_loss.item(),
                                 'g_loss':g_loss.item()})

    with torch.no_grad():
        grid = make_grid(img_fake, normalize=True)
        plt.imshow(grid.permute(1,2,0).cpu())
        plt.show()
    # print(d_loss, g_loss)









