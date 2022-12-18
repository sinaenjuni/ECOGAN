import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import Decoder as BaseDecoder, Encoder as BaseEncoder
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import datetime
from utils import misc



class Encoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = BaseEncoder(img_dim, latent_dim)
        self.linear = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.encoder.dims[3] * (4 * 4), out_features=latent_dim))
    def forward(self, img):
        out = self.encoder(img)
        out = self.linear(out)
        return out
class Decoder(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = BaseDecoder(img_dim, latent_dim)

    def forward(self, img):
        return self.decoder(img)
class Generator(nn.Module):
    def __init__(self, img_dim, latent_dim):
        super(Generator, self).__init__()
        self.decoder = BaseDecoder(img_dim, latent_dim)
    def forward(self, latent):
        return self.decoder(latent)
class Discriminator(nn.Module):
    def __init__(self, img_dim, latent_dim, num_classes):
        super(Discriminator, self).__init__()
        self.encoder = BaseEncoder(img_dim, latent_dim, num_classes)
        self.adv = nn.Sequential(nn.Flatten(1),
                                     nn.Linear(in_features=self.encoder.dims[3] * (4 * 4),
                                               out_features=num_classes+1))
    def forward(self, img):
        out = self.encoder(img)
        out = self.adv(out)
        return out
class ClassCondLatentGen:
    def __init__(self):
        self.latents = list()
        self.targets = list()
        self.mean: dict
        self.cov: dict
        self.num_classes: list
        self.classes: list
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


def pre_training(data_loader, logger, world_size, device, args):
    encoder = Encoder(img_dim=args.img_dim, latent_dim=args.latent_dim).to(device)
    decoder = Decoder(img_dim=args.img_dim, latent_dim=args.latent_dim).to(device)
    
    if world_size > 1:
       encoder = DDP(encoder, device_ids=[device])
       decoder = DDP(decoder, device_dis=[device]) 
        
    optimizer = torch.optim.Adam([{'params': encoder.parameters(),
                                   'params': decoder.parameters()}], 
                                 lr=args.lr, 
                                 betas=(args.beta1, args.beta2))
    loss_fn = nn.MSELoss()
    
    start_time = datetime.now()
    losses = 0
    for epoch in range(args.epoch_ae):
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
           
            optimizer.zero_grad()
            outputs = encoder(img)
            outputs = decoder(outputs)
            loss = loss_fn(outputs, img)
            loss.backward()
            optimizer.step()
            losses += loss.item()


        losses = losses / len(data_loader)
        print(f'epoch: {epoch+1}/{args.epoch_ae}({((epoch+1) / args.epoch_ae)*100:.2f}%), '
                f'time: {misc.elapsed_time(start_time)}, '
                f'loss: {losses:.4f}')
        logger.log({'loss/ae':losses})
        losses = 0

    return encoder, decoder


def gan_training(data_loader, logger, world_size, device, args):
    
    if args.pre_traning:
        pre_training(data_loader, logger, world_size, device, args)