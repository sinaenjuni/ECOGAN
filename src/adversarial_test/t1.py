import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
# from src.adversarial_test.model import Encoder, Decoder


img_dim = 1
latent_dim = 128
batch_size = 64
lr_g = 0.0002
lr_d = 0.0002
num_epoch = 1

dataset = FashionMNIST(root='/shared_hdd/sin/dataset', download=False, train=True,
                transform=Compose([ToTensor(),
                                   Resize(64),
                                   Normalize([0.5],
                                             [0.5])]))

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
img, label = iter(data_loader).__next__()

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
        self.dims = [256//d, 128//d, 128//d, 64//d, img_dim]

        self.linear0 = nn.Sequential(nn.Linear(in_features=latent_dim, out_features= self.dims[0] * (4 * 4)),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.deconv0 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[0], out_channels=self.dims[1],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[1]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[1], out_channels=self.dims[2],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[2]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=self.dims[2], out_channels=self.dims[3],
                                                        kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(self.dims[3]),
                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.deconv3 = nn.ConvTranspose2d(in_channels=self.dims[3], out_channels=self.dims[4], kernel_size=4,
                                          stride=2, padding=1)
        self.tanh = nn.Tanh()
        self.initialize_weights()

    def forward(self, x):
        x = self.linear0(x)
        x = x.view(-1, self.dims[0], 4, 4)
        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
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
        self.dims = [64//d, 128//d, 128//d, 256//d]

        self.conv0 = nn.Sequential(nn.Conv2d(in_channels=img_dim, out_channels=self.dims[0], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.dims[0], out_channels=self.dims[1], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.dims[1], out_channels=self.dims[2], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=self.dims[2], out_channels=self.dims[3], kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.linear0 = nn.Linear(in_features=self.dims[3], out_features=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.sum(x, dim=[2, 3])
        adv_output = self.linear0(x)
        return adv_output



decoder = Decoder(img_dim=img_dim, latent_dim=latent_dim).cuda()
encoder = Encoder(img_dim=img_dim, latent_dim=img_dim).cuda()




optimizer_g = torch.optim.Adam(params=decoder.parameters(), lr=lr_g)
optimizer_d = torch.optim.Adam(params=encoder.parameters(), lr=lr_d)

def get_loss(a, y):
    return -1 * (y * torch.log(a) +
               (1-y) * torch.log(1-a))

for epoch in range(num_epoch):
    for img, label in data_loader:
        img_real, label = img.cuda(), label.cuda()
        latent = torch.randn(batch_size, latent_dim).cuda()

        optimizer_d.zero_grad()

        img_fake = decoder(latent).detach()
        adv_real = torch.sigmoid(encoder(img_real))
        adv_fake = torch.sigmoid(encoder(img_fake))

        d_loss = get_loss(adv_real, 1).mean() + get_loss(adv_fake, 0).mean()

        d_loss.backward()
        optimizer_d.step()



        optimizer_g.zero_grad()

        img_fake = decoder(latent)
        adv_fake = torch.sigmoid(encoder(img_fake))

        g_loss = get_loss(adv_fake, 1).mean()

        g_loss.backward()
        optimizer_g.step()

        print(d_loss, g_loss)









