import os
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
        self.encoder = BaseEncoder(img_dim, latent_dim)
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


def pre_training(data_loader, logger, world_size, rank, args):
    encoder = Encoder(img_dim=args.img_dim, latent_dim=args.latent_dim).to(rank)
    decoder = Decoder(img_dim=args.img_dim, latent_dim=args.latent_dim).to(rank)
    
    encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    decoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(decoder)
    
    if world_size > 1:
       encoder = DDP(encoder, device_ids=[rank])
       decoder = DDP(decoder, device_ids=[rank]) 
        
    optimizer = torch.optim.Adam([{'params': encoder.parameters(),
                                   'params': decoder.parameters()}], 
                                 lr=args.lr, 
                                 betas=(args.beta1, args.beta2))
    loss_fn = nn.MSELoss()
    
    start_time = datetime.now()
    losses = 0
    for epoch in range(args.epoch_ae):
        for img, label in data_loader:
            img, label = img.to(rank), label.to(rank)
           
            optimizer.zero_grad()
            outputs = encoder(img)
            outputs = decoder(outputs)
            loss = loss_fn(outputs, img)
            loss.backward()
            optimizer.step()
            losses += loss.item()

        losses = losses / len(data_loader)
        if rank == 0:
            print(f'epoch: {epoch+1}/{args.epoch_ae}({((epoch+1) / args.epoch_ae)*100:.2f}%), '
                    f'time: {misc.elapsed_time(start_time)}, '
                    f'loss: {losses:.4f}')
        if logger is not None:
            logger.log({'loss/ae':losses})
        losses = 0

    return encoder, decoder


def gan_training(data_loader, logger, world_size, rank, args):
    encoder, decoder = pre_training(data_loader, logger, world_size, rank, args)
    
    num_classes = len(data_loader.dataset.classes)
    G = Generator(args.img_dim, latent_dim=args.latent_dim).to(rank)
    D = Discriminator(img_dim=args.img_dim, latent_dim=args.latent_dim, num_classes=num_classes).to(rank)
    
    ret_G = G.load_state_dict(decoder.module.state_dict(), strict=False)
    ret_D = D.load_state_dict(encoder.module.state_dict(), strict=False)
    print(ret_G)
    print(ret_D)
    
    calc_mean_cov = ClassCondLatentGen()
    calc_mean_cov.stacking(data_loader, encoder)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    loss_fn = nn.CrossEntropyLoss()

    start_time = datetime.now()
    data_iter = iter(data_loader)
    for step in range(args.steps):
        try:
            img_real, label_real = data_iter.next()
        except StopIteration:
            data_iter = iter(data_loader)
            img_real, label_real = data_iter.next()
        img_real, label_real = img_real.to(rank), label_real.to(rank)

        sample_labels = np.random.randint(0, 10, label_real.size(0))
        cond_latent = calc_mean_cov.sampling(sample_labels).to(rank)
        fake_label = (torch.ones_like(label_real) * num_classes).to(rank)

        optimizer_d.zero_grad()
        gen_img = G(cond_latent).detach()
        output_real = D(img_real)
        output_fake = D(gen_img)

        loss_d = loss_fn(output_real, label_real) + loss_fn(output_fake, fake_label)
        loss_d.backward()
        optimizer_d.step()

        ###########################
        sample_labels = np.random.randint(0, 10, label_real.size(0))
        cond_latent = calc_mean_cov.sampling(sample_labels).to(rank)
        fake_label = torch.from_numpy(sample_labels).to(rank)

        optimizer_g.zero_grad()
        gen_img = G(cond_latent)
        output_fake = D(gen_img)
        loss_g = loss_fn(output_fake, fake_label)
        loss_g.backward()
        optimizer_g.step()

        if rank == 0:
            print(f'step: {step+1}/{args.steps}({((step+1) / args.steps)*100:.2f}%), '
                  f'time: {misc.elapsed_time(start_time)}, '
                  f'loss D: {loss_d.item():.4f} '
                  f'loss G: {loss_g.item():.4f}')
        if logger is not None:
            logger.log({'loss/d':loss_d.item()})
            logger.log({'loss/g':loss_g.item()})


    if rank==0:
        print('Save the weights.')
        torch.save(encoder.state_dict(), os.path.join(args.path, 'encoder.pth'))
        torch.save(decoder.state_dict(), os.path.join(args.path, 'decoder.pth'))
        torch.save(G.state_dict(), os.path.join(args.path, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(args.path, 'D.pth'))



#

# vis_label = torch.arange(0, 10).repeat(10)
# vis_z = calc_mean_cov.sampling(vis_label.numpy())


# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
# with torch.no_grad():
#     # gen = bagan_ae.get_decoder(cond_latent.cuda())
#     gen = generator(vis_z.cuda())
#     # gen = torch.cat([gen, bagan_ae.get_decoder(distrib.rsample((100,)).cuda())])
#     # gen = bagan_ae.get_decoder(distrib.rsample((100,)).cuda())

#     gen = make_grid(gen, nrow=10, normalize=True)
# plt.imshow(gen.cpu().permute(1,2,0))
# plt.show()


# parser = ArgumentParser()
# parser.add_argument("--gpus", nargs='+', type=int, default=7, required=True)

# args = parser.parse_args()

# wandb.init(project="eval_cls", entity="sinaenjuni")
# wandb.config.update(args)
