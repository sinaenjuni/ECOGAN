import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import *
from torch.distributions.multivariate_normal import MultivariateNormal


class BaganEncoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(BaganEncoder, self).__init__()
        self.encoder = Encoder(img_dim, latent_dim)
        self.linear = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.encoder.dims[3] * (4 * 4), out_features=latent_dim))
    def forward(self, img):
        out = self.encoder(img)
        out = self.linear(out)
        return out
class BaganDecoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(BaganDecoder, self).__init__()
        self.decoder = Decoder(img_dim, latent_dim)

    def forward(self, img):
        return self.decoder(img)

class BaganGenerator(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(BaganGenerator, self).__init__()
        self.decoder = Decoder(img_dim, latent_dim)
    def forward(self, latent):
        return self.decoder(latent)

class BaganDiscriminator(nn.Module):
    def __init__(self, img_dim, latent_dim, num_classes):
        super(BaganDiscriminator, self).__init__()
        self.encoder = Encoder(img_dim, latent_dim, num_classes)
        self.adv = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.encoder.dims[3] * (4 * 4),
                                               out_features=num_classes+1))
    def forward(self, img):
        out = self.encoder(img)
        out = self.adv(out)
        return out

class ClassCondLatentGen:
    def __init__(self):
        self.latents=[]
        self.targets=[]
        self.mean={}
        self.cov={}
        self.num_classes:list
        self.classes:list
    def stacking(self, data_loader, model):
        print("Stacking BAGAN AE features")
        model.eval()
        for img, labels in data_loader:
            img, labels = img.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(img)

                self.latents.append(outputs.cpu().numpy())
                self.targets.append(labels.cpu().numpy())

        latents_np = np.concatenate(self.latents)
        targets_np = np.concatenate(self.targets)
        self.classes, self.num_classes = np.unique(targets_np, return_counts=True)
        latents_np_cls = {c: latents_np[np.where(targets_np == c)] for c in self.classes}
        self.mean = {c: np.mean(e, axis=0) for c, e in latents_np_cls.items()}
        self.cov = {c: np.cov(e.T) for c, e in latents_np_cls.items()}
        print("Finish stacking BAGAN AE features")

    def sampling(self, labels):
        sample_latents = np.stack([np.random.multivariate_normal(self.mean[c], self.cov[c]).astype(np.float32)
                                   for c in labels])
        sample_latents = torch.from_numpy(sample_latents)
        return sample_latents


import importlib
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda
from torch.utils.data import DataLoader

transforms = Compose([ToTensor(),
                      Resize(64),
                      Lambda(lambda x : x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                      Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])

dataset_module = importlib.import_module('utils.datasets')
dataset_train = getattr(dataset_module, 'FashionMNIST_LT')(is_train=True, is_extension=False, transform=transforms)

loader_train = DataLoader(dataset=dataset_train,
                          batch_size=128,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True,
                          persistent_workers=True)

num_classes = len(loader_train.dataset.classes)
latent_dim = 128
img_dim = 3

bagan_encoder = BaganEncoder(img_dim=img_dim, latent_dim=latent_dim).cuda()
bagan_decoder = BaganDecoder(img_dim=img_dim, latent_dim=latent_dim).cuda()
optimizer = torch.optim.Adam([{'params': bagan_encoder.parameters(),
                               'params': bagan_decoder.parameters()}], lr =0.00005, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()
calc_mean_cov = ClassCondLatentGen()

for epoch in range(150):
    print(epoch)
    loss_epoch = 0
    for img, labels in loader_train:
        img, labels = img.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = bagan_encoder(img)
        outputs = bagan_decoder(outputs)
        loss = loss_fn(outputs, img)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        loss_epoch += loss.item()

    print(loss_epoch/len(loader_train))

calc_mean_cov.stacking(loader_train, bagan_encoder)

bagan_generator = BaganGenerator(img_dim=img_dim, latent_dim=latent_dim).cuda()
bagan_discriminator = BaganDiscriminator(img_dim=img_dim, latent_dim=latent_dim, num_classes=num_classes).cuda()

bagan_generator.load_state_dict(bagan_decoder.state_dict(), strict=False)
bagan_discriminator.load_state_dict(bagan_encoder.state_dict(), strict=False)

optimizer_g = torch.optim.Adam(bagan_generator.parameters(), lr =0.00005, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(bagan_discriminator.parameters(), lr =0.00005, betas=(0.5, 0.999))
loss_fn = nn.CrossEntropyLoss()

iter_train = iter(loader_train)
steps=100
for step in range(steps):
    try:
        img_real, label_real = next(iter_train)
    except StopIteration:
        iter_train = iter(iter_train)
        img_real, label_real = next(iter_train)

    img_real, label_real = img_real.cuda(), label_real.cuda()

    sample_labels = np.random.randint(0, 10, label_real.size(0))
    cond_latent = calc_mean_cov.sampling(sample_labels).cuda()
    fake_label = (torch.ones_like(label_real) * num_classes).cuda()

    optimizer_d.zero_grad()
    gen_img = bagan.get_generator(cond_latent).detach()
    output_real = bagan.get_discriminator(img_real)
    output_fake = bagan.get_discriminator(gen_img)

    loss_d = loss_fn(output_real, label_real) + loss_fn(output_fake, fake_label)
    loss_d.backward()
    optimizer_d.step()

    ###########################
    sample_labels = np.random.randint(0, 10, label_real.size(0))
    cond_latent = calc_mean_cov.sampling(sample_labels).cuda()
    fake_label = torch.from_numpy(sample_labels).cuda()

    optimizer_g.zero_grad()
    gen_img = bagan.get_generator(cond_latent)
    output_fake = bagan.get_discriminator(gen_img)
    loss_g = loss_fn(output_fake, fake_label)
    loss_g.backward()
    optimizer_g.step()

    print(loss_d.item(), loss_g.item())





# sample_labels = np.concatenate([np.ones(10)*i for i in range(10)])
# cond_latent = calc_mean_cov.sampling(sample_labels)





# la = np.stack([np.random.multivariate_normal(mean[0], cov[0]) for i in range(100)] )
#
# from torch.distributions.multivariate_normal import MultivariateNormal
# distrib = MultivariateNormal(loc=torch.from_numpy(mean[0]).to(torch.float32),
#                              covariance_matrix=torch.from_numpy(cov[0]).to(torch.float32))
#
#
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# with torch.no_grad():
#     gen = bagan_ae.get_decoder(torch.from_numpy(la).to(torch.float32).cuda())
#     gen = torch.cat([gen, bagan_ae.get_decoder(distrib.rsample((100,)).cuda())])
#     # gen = bagan_ae.get_decoder(distrib.rsample((100,)).cuda())
#
#     gen = make_grid(gen, nrow=10, normalize=True)
# plt.imshow(gen.cpu().permute(1,2,0))
# plt.show()
#
# latents = []
# targets = []
# for img, labels in loader_train:
#     img, labels = img.cuda(), labels.cuda()
#     with torch.no_grad():
#         outputs = bagan_ae.get_encoder(img)
#
#         latents.append(outputs.cpu().numpy())
#         targets.append(labels.cpu().numpy())
# #
# latents_np = np.concatenate(latents)
# targets_np = np.concatenate(targets)
# classes, num_classes = np.unique(targets_np, return_counts=True)
# latents_np_cls = {c: latents_np[np.where(targets_np == c)] for c in classes}
# mean = {c: np.mean(e, axis=0) for c, e in latents_np_cls.items()}
# cov = {c: np.cov(e.T) for c, e in latents_np_cls.items()}



# sample_latents = (torch.rand(128,) * 10).to(torch.int)
# sample_target = np.random.randint(0, 10, 128)
# la = np.stack([np.random.multivariate_normal(mean[c], cov[c]).astype(np.float32) for c in sample_target] )
# la = np.random.multivariate_normal(mean[sample_target], cov[sample_target])
#
#
#
# distrib = MultivariateNormal(loc=mean[sample_latents],
#                              covariance_matrix=cov[sample_latents])
# return distrib.rsample((batch_size,))
#
#
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
with torch.no_grad():
    # gen = bagan_ae.get_decoder(cond_latent.cuda())
    gen = bagan.get_generator(cond_latent)
    # gen = torch.cat([gen, bagan_ae.get_decoder(distrib.rsample((100,)).cuda())])
    # gen = bagan_ae.get_decoder(distrib.rsample((100,)).cuda())

    gen = make_grid(gen, nrow=10, normalize=True)
plt.imshow(gen.cpu().permute(1,2,0))
plt.show()


