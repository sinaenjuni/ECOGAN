import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from collections import OrderedDict

from EBGAN.dataset import DataModule_
from torchvision.utils import make_grid
import wandb
from pytorch_lightning.loggers import WandbLogger
from EBGAN.models import Encoder, Decoder, Embedding_labeled_latent
from torchmetrics.image.fid import FrechetInceptionDistance



class Generator(nn.Module):
    def __init__(self, img_dim, latent_dim, num_class):
        super(Generator, self).__init__()

        self.embedding = Embedding_labeled_latent(latent_dim=latent_dim, num_class=num_class)
        self.decoder = Decoder(img_dim=img_dim, latent_dim=latent_dim)


    def forward(self, z, label):
        latent = self.embedding(z, label)
        gened_img = self.decoder(latent)

        return gened_img

class Discriminator(nn.Module):
    def __init__(self, img_dim, latent_dim, num_class):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(img_dim, latent_dim)

        self.embedding = nn.Sequential(nn.Embedding(num_embeddings=num_class, embedding_dim=512),
                                       nn.Flatten(),
                                       nn.Linear(512, 256 * (4 * 4)),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.discriminator = nn.Linear(256 * (4*4), 1)


    def forward(self, img, label):
        x = self.encoder.getFeatures(img)
        x = torch.flatten(x, 1)

        le = self.embedding(label)

        out = x * le
        out = self.discriminator(out)
        return out


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, img_dim, num_class):
        super(GAN, self).__init__()

        # path = '/home/dblab/git/VAE-GAN/src/EBGAN/GAN/1gqzkv1e/checkpoints/epoch=28-step=1508.ckpt'
        # weights = torch.load(path)

        self.latent_dim = latent_dim
        self.fid = FrechetInceptionDistance()

        self.G = Generator(img_dim=img_dim, latent_dim=latent_dim, num_class=num_class)
        self.D = Discriminator(img_dim=img_dim, latent_dim=latent_dim, num_class=num_class)

        # self.G.load_state_dict(weights['state_dict'], strict=False)
        # self.D.load_state_dict(weights['state_dict'], strict=False)


    def forward(self, z, label):
        return self.G(z, label)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, real_labels = batch
        batch_size = real_imgs.size(0)


        if optimizer_idx == 0:
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_labels = (torch.rand((batch_size,)) * 10).to(torch.long).to(self.device)
            wrong_labels = (torch.rand((batch_size,)) * 10).to(torch.long).to(self.device)

            fake_imgs = self(z, real_labels).detach()
            fake_logits = self.D(fake_imgs, fake_labels)
            real_logits = self.D(real_imgs, real_labels)
            wrong_logits = self.D(real_imgs, wrong_labels)
            d_cost = self.d_loss(real_logits, fake_logits, wrong_logits)

            gp = self.compute_gradient_penalty(real_imgs, fake_imgs, real_labels)
            d_loss = d_cost + gp * 10.0

            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_epoch=True)
            return d_loss

        if optimizer_idx == 1:
            z = torch.randn(real_imgs.size(0), self.latent_dim).to(self.device)
            fake_labels = (torch.rand((batch_size,)) * 10).to(torch.long).to(self.device)

            fake_imgs = self(z, fake_labels)
            fake_logits = self.D(fake_imgs, fake_labels)
            g_loss = self.g_loss(fake_logits)

            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_epoch=True)
            return g_loss


    # def training_epoch_end(self, outputs):
    #     z = torch.randn((100, self.latent_dim)).to(self.device)
    #     label = torch.arange(0, 9, dtype=torch.long).repeat(10).to(self.device)
    #     gened_imgs = self(z, label)
    #     self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        z = torch.randn((imgs.size(0), self.latent_dim)).to(self.device)
        # label = torch.arange(0, 9, dtype=torch.long).repeat(100).to(self.device)
        gend_imgs = self(z, labels)

        imgs = (((imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        self.fid.update(imgs, real=True)
        self.fid.update(gend_imgs, real=False)

    def validation_epoch_end(self, outputs):
        # print('valid_fid_epoch', self.fid.compute())
        self.log('fid', self.fid.compute(), logger=True, prog_bar=True, on_epoch=True)
        self.fid.reset()

        z = torch.randn((100, self.latent_dim)).to(self.device)
        label = torch.arange(0, 10, dtype=torch.long).repeat(10).to(self.device)
        gened_imgs = self(z, label)
        self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)


    def configure_optimizers(self):
        optimizer_g = Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.9))
        optimizer_d = Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.9))

        return [{'optimizer': optimizer_d, 'frequency': 5},
                {'optimizer': optimizer_g, 'frequency': 1}]


    def mes_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)


    def d_loss(self, real_logits, fake_logits, wrong_logits):
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        wrong_loss = F.binary_cross_entropy_with_logits(wrong_logits, torch.zeros_like(wrong_logits))

        return real_loss + fake_loss + wrong_loss


    def compute_gradient_penalty(self, real_samples, fake_samples, real_labels):
        # real_samples = real_samples.reshape(real_samples.size(0), 1, 28, 28).to(device)
        # fake_samples = fake_samples.reshape(fake_samples.size(0), 1, 28, 28).to(device)

        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        # alpha = torch.rand(real_samples.size(0), 1, 1, 1)
        alpha = torch.randn(real_samples.size(0), 1, 1, 1).to(self.device)
        # Get random interpolation between real and fake samples
        # interpolates = (alpha * real_samples.data + ((1 - alpha) * fake_samples.data)).requires_grad_(True)
        diff = fake_samples - real_samples
        interpolates = (real_samples + alpha * diff).requires_grad_(True)

        d_interpolates = self.D(interpolates, real_labels)

        weights = torch.ones(d_interpolates.size()).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=weights,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradients2L2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        gradient_penalty = torch.mean(( gradients2L2norm - 1 ) ** 2)
        return gradient_penalty


    def g_loss(self, fake_logits):
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
        return fake_loss





if __name__ == "__main__":
    # decoder = Decoder(3, 128)
    # z = torch.randn(100, 128)
    # output = decoder(z)
    # print(output.shape)
    #
    # encoder = Encoder(3, 128)
    # img = torch.randn(100, 3, 64, 64)
    # output = encoder(img)
    # print(output.shape)
    #
    #
    # label = torch.randint(0,10, (100, ))
    # le = Embedding_labeled_latent(128, 10)
    # output = le(z, label)

    dm = DataModule_(path_train='/home/dblab/sin/save_files/refer/ebgan_cifar10', batch_size=128)
    model = GAN(latent_dim=128, img_dim=3, num_class=10)

    # model

    # wandb.login(key='6afc6fd83ea84bf316238272eb71ef5a18efd445')
    wandb.init(project='GAN', name='no-pre-train-4g')

    wandb_logger = WandbLogger(project="GAN")
    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=500,
        # callbacks=[EarlyStopping(monitor='val_loss')],
        callbacks=[pl.callbacks.ModelCheckpoint(filename="EBGAN-{epoch:02d}-{fid}",
                                                monitor="fid", mode='min')],
        logger=wandb_logger,
        # logger=False,
        strategy='ddp',
        accelerator='gpu',
        gpus=[4,5,6,7],
        check_val_every_n_epoch=10
    )
    trainer.fit(model, datamodule=dm)


    # img = torch.randn(100, 3, 64, 64)
    # label = torch.randint(0,10, (100, ))
    # ae =
    # output = ae(img, label)
