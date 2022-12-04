import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import *



class BaganAE(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(BaganAE, self).__init__()
        self.encoder = Encoder(img_dim, latent_dim)
        self.decoder = Decoder(img_dim, latent_dim)

    def forward(self, img):
        return self.decoder(self.encoder(img))

    def get_encoder(self, img):
        return self.encoder(img)

    def get_decoder(self, img):
        return self.decoder(img)


class CalcMeanCovar:
    def __init__(self):
        self.latents = []
        self.targets = []

    def update(self, latent, target):
        self.latents.append(latent)
        self.targets.append(target)

    def calc(self):
        latents = np.concatenate(self.latents)
        targets = np.concatenate(self.targets)

        classes, num_classes = np.unique(targets, return_counts=True)
        target_latents = [latents[np.where(targets == c)] for c in range(len(classes))]
        target_mean = [np.mean(e) for e in target_latents]
        target_cov = [np.cov(e) for e in target_latents]



import importlib
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader

transforms = Compose([ToTensor(),
                      Resize(64),
                      Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])

dataset_module = importlib.import_module('utils.datasets')
dataset_train = getattr(dataset_module, 'CIFAR10_LT')(is_train=True, is_extension=False, transform=transforms)

loader_train = DataLoader(dataset=dataset_train,
                          batch_size=128,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True,
                          persistent_workers=True)


bagan_ae = BaganAE(3, 128).cuda()
optimizer = torch.optim.Adam(bagan_ae.parameters(), lr=0.00005, betas=(0.5, 0.999))
loss_fn = nn.MSELoss()


for epoch in range(150):
    loss_epoch = 0
    for img, labels in loader_train:
        img, labels = img.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = bagan_ae(img)
        loss = loss_fn(outputs, img)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        loss_epoch += loss.item()

    print(loss_epoch/len(loader_train))


latents = []
targets = []
for img, labels in loader_train:
    img, labels = img.cuda(), labels.cuda()
    with torch.no_grad():
        outputs = bagan_ae.get_encoder(img)

        latents.append(outputs.cpu().numpy())
        targets.append(labels.cpu().numpy())


latents_np = np.concatenate(latents)
targets_np = np.concatenate(targets)

np.mean(np.concatenate(latents))
np.cov(np.concatenate(latents).T).shape
torch.concat(targets)

torch.concat(latents).mean()
torch.concat(latents).T.cov().size()
torch.concat(targets)
