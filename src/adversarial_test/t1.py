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


img_dim = 3
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
                                   # Resize(64),
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

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias),
                         # eps=1e-6
                         )

def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias),
                         # eps=1e-6
                         )

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias),
                         # eps=1e-6
                         )

g_deconv = sndeconv2d
g_conv = snconv2d
g_linear = snlinear
g_act_fn = nn.ReLU(True)

d_conv = snconv2d
d_linear = snlinear
d_act_fn = nn.ReLU(True)



class Decoder(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def __init__(self, img_dim, latent_dim):
        super(Decoder, self).__init__()
        # self.dims = [256, 128, 128, 64, img_dim]
        d = 16
        self.dims = [int(d*16), int(d*8), int(d*4), int(d*2), int(d), img_dim]

        self.linear0 = g_linear(in_features=latent_dim, out_features= self.dims[0] * (4 * 4))

        self.deconv0 = nn.Sequential(g_deconv(in_channels=self.dims[0], out_channels=self.dims[1],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[1]),
                                     g_act_fn)

        self.deconv1 = nn.Sequential(g_deconv(in_channels=self.dims[1], out_channels=self.dims[2],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[2]),
                                     g_act_fn)

        self.deconv2 = nn.Sequential(g_deconv(in_channels=self.dims[2], out_channels=self.dims[3],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[3]),
                                     g_act_fn)

        self.deconv3 = nn.Sequential(g_deconv(in_channels=self.dims[3], out_channels=self.dims[4], kernel_size=4,
                                          stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[4]),
                                     g_act_fn)

        self.conv4 = g_conv(in_channels=self.dims[4], out_channels=self.dims[5], kernel_size=3,
                                          stride=1, padding=1)

        self.tanh = nn.Tanh()
        self.initialize_weights()

    def forward(self, x):
        x = self.linear0(x)
        x = x.view(-1, self.dims[0], 4, 4)
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.conv4(x)
        x = self.tanh(x)
        return x

class Encoder(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def __init__(self, img_dim, latent_dim):
        super(Encoder, self).__init__()
        # self.dims = [64, 128, 128, 256]
        d = 16
        self.dims = [int(d), int(d*2), int(d*4), int(d*8)]

        self.conv0 = nn.Sequential(d_conv(in_channels=img_dim, out_channels=self.dims[0], kernel_size=4, stride=2, padding=1),
                                   d_act_fn)

        self.conv1 = nn.Sequential(d_conv(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=4, stride=2, padding=1),
                                   d_act_fn)

        self.conv2 = nn.Sequential(d_conv(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=4, stride=2, padding=1),
                                   d_act_fn)

        self.conv3 = nn.Sequential(d_conv(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=4, stride=2, padding=1),
                                   d_act_fn)

        self.linear0 = d_linear(in_features=self.dims[3], out_features=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.sum(x, dim=[2, 3])
        adv_output = self.linear0(x)
        adv_output = torch.sigmoid(adv_output)
        return adv_output



decoder = Decoder(img_dim=img_dim, latent_dim=latent_dim).cuda()
encoder = Encoder(img_dim=img_dim, latent_dim=img_dim).cuda()

# latent = torch.randn(batch_size, latent_dim).cuda()
# img_fake = decoder(latent).detach()
# adv_output = encoder(img.cuda())
# print(img_fake.size())

optimizer_g = torch.optim.Adam(params=decoder.parameters(), lr=lr_g)
optimizer_d = torch.optim.Adam(params=encoder.parameters(), lr=lr_d)

bce_loss = nn.BCELoss()

for epoch in range(num_epoch):
    loader_tqdm = tqdm(data_loader)
    for img, label in loader_tqdm:
        img_real, label = img.cuda(), label.cuda()
        latent = torch.randn(img_real.size(0), latent_dim).cuda()

        y_real_, y_fake_ = torch.ones(img_real.size(0), 1).cuda(), torch.zeros(img_real.size(0), 1).cuda()

        optimizer_d.zero_grad()

        img_fake = decoder(latent).detach()
        adv_real = encoder(img_real)
        adv_fake = encoder(img_fake)

        # d_loss = get_loss(adv_real, 1).mean() + get_loss(adv_fake, 0).mean()
        # d_loss = -1 * torch.log(adv_real).mean() + -1 * torch.log(1-adv_fake).mean()
        d_loss = bce_loss(adv_real, y_real_) + bce_loss(adv_fake, y_fake_)

        d_loss.backward()
        optimizer_d.step()



        optimizer_g.zero_grad()

        img_fake = decoder(latent)
        adv_fake = encoder(img_fake)

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









