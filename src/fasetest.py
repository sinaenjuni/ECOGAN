
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from utils.dataset import DataModule_
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from models import Generator, Discriminator
from argparse import ArgumentParser
from metric.img_metrics import Fid_and_is
import wandb
from pathlib import Path


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, img_dim, num_classes, lr, betas, *args, **kwargs):
        super(GAN, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.betas = betas
        self.img_dim = img_dim
        self.latent_dim = latent_dim
        self.img_metric = Fid_and_is()
        self.G = Generator(img_dim=img_dim, latent_dim=latent_dim, num_classes=num_classes)
        self.D = Discriminator(img_dim=img_dim, latent_dim=latent_dim, num_classes=num_classes)


    def on_load_checkpoint(self, checkpoint):
        self.G.decoder.load_state_dict(checkpoint['decoder'])
        self.G.embedding.load_state_dict(checkpoint['embedding'])
        self.D.encoder.load_state_dict(checkpoint['encoder'])


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

            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            return d_loss

        if optimizer_idx == 1:
            z = torch.randn(real_imgs.size(0), self.latent_dim).to(self.device)
            fake_labels = (torch.rand((batch_size,)) * 10).to(torch.long).to(self.device)

            fake_imgs = self(z, fake_labels)
            fake_logits = self.D(fake_imgs, fake_labels)
            g_loss = self.g_loss(fake_logits)

            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
            return g_loss


    def validation_step(self, batch, batch_idx):
        real_imgs, labels = batch
        if self.current_epoch == 0:
            self.img_metric.update(real_imgs, real=True)

        # z = torch.randn((imgs.size(0), self.latent_dim)).to(self.device)
        # label = torch.arange(0, 9, dtype=torch.long).repeat(100).to(self.device)
        # gend_imgs = self(z, labels)
        #
        # imgs = (((imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        # gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        # self.fid.update(imgs, real=True)
        # self.fid.update(gend_imgs, real=False)

        # with torch.no_grad():
            # label = targets[i*batch_size : batch_size * (i + 1)].cuda()
            # z = torch.randn(label.size(0), latent_dim).cuda()
        z = torch.randn((real_imgs.size(0), self.latent_dim)).to(self.device)
        img_fake = self(z, labels)
        self.img_metric.update(img_fake, real=False)
            # img_fake = ((0.5*img_fake+0.5)*255.).to(torch.uint8)
            # if self.img_dim == 1:
            #     img_fake = img_fake.repeat(1,3,1,1)
            # self.fid.update(img_fake, real=False)
            # self.ins.update(img_fake)
            # embeddings, logits = self.eval_model(img_fake, quantize=True)
            # ps = torch.nn.functional.softmax(logits, dim=1)
            # ps_list.append(ps)
            # em_list.append(embeddings)

        # return {'ps': ps, 'embedding': embeddings}
        # return {"img_fake": img_fake, "label": labels}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # ps = torch.cat([output['ps'] for output in outputs])
        # embedding = torch.cat([output['embedding'] for output in outputs])
        # print(ps.size())
        # print(embedding.size())

        # ins_score, ins_std = calculate_kl_div(ps, 10)
        # mu_target, sigma_target = calculate_mu_sigma(embedding.cpu().numpy())
        # fid_score = frechet_inception_distance(self.mu_original, self.sigma_original, mu_target, sigma_target)


        # ins_score = self.ins.compute()[0]
        # fid_score = self.fid.compute()
        ins_score = self.img_metric.compute_ins()[0]
        fid_score = self.img_metric.compute_fid()

        # print('ins_score', ins_score)
        # print('fid_score', fid_score)
        self.log('fid', fid_score, logger=True, prog_bar=True, on_epoch=True, on_step=False)
        self.log('ins', ins_score, logger=True, prog_bar=True, on_epoch=True, on_step=False)
        self.img_metric.reset(real=False)
        # self.ins.reset()
        # self.fid.reset()
        # print('valid_fid_epoch', self.fid.compute())
        # self.log('fid', self.fid.compute(), logger=True, prog_bar=True, on_epoch=True)
        # self.fid.reset()

        # z = torch.randn((100, self.latent_dim)).to(self.device)
        # label = torch.arange(0, 10, dtype=torch.long).repeat(10).to(self.device)
        # gened_imgs = self(z, label)
        # self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)


    def configure_optimizers(self):
        optimizer_g = Adam(self.G.parameters(), lr=self.lr, betas=self.betas)
        optimizer_d = Adam(self.D.parameters(), lr=self.lr, betas=self.betas)

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

    parser = ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10, required=False)
    parser.add_argument("--lr", type=float, default=0.0002, required=False)
    parser.add_argument("--betas", type=tuple, default=(0.5, 0.9), required=False)
    parser.add_argument("--img_dim", type=int, default=1, required=False)
    parser.add_argument("--latent_dim", type=int, default=128, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--gpus", nargs='+', type=int, default=7, required=False)
    parser.add_argument("--data_name", type=str, default='imb_FashionMNIST',
                        choices=['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST'], required=False)

    args = parser.parse_args()
    dm = DataModule_.from_argparse_args(args)
    model = GAN(**vars(args))

    api = wandb.Api()
    artifact = api.artifact(name=f'sinaenjuni/EBGAN-AE/{args.data_name}:v0', type='model')
    # artifact = api.artifact(name=f'sinaenjuni/EBGAN-AE/imb_CIFAR10:v0', type='model')
    artifact_dir = artifact.download()
    ch = torch.load(Path(artifact_dir) / "model.ckpt")
    model.on_load_checkpoint(ch)

    # dm = DataModule_(path_train='ß/home/dblab/sin/save_files/refer/ebgan_cifar10', batch_size=128)
    # dm = DataModule_(path_train='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train', batch_size=128, num_workers=4)
    # model = GAN(latent_dim=128, img_dim=3, num_class=10)
    # wandb.login(key='6afc6fd83ea84bf316238272eb71ef5a18efd445')
    # wandb.init(project='MYGAN', name='BEGAN-GAN')

    wandb_logger = WandbLogger(project='MYTEST1', name=f'BEGAN-GAN(pre-trained_{args.data_name})', log_model=True)
    wandb.define_metric('fid', summary='min')

    trainer = pl.Trainer.from_argparse_args(args,
        fast_dev_run=False,
        default_root_dir='/shared_hdd/sin/save_files/EBGAN/',
        max_epochs=100,
        # callbacks=[EarlyStopping(monitor='val_loss')],
        callbacks=[pl.callbacks.ModelCheckpoint(filename="EBGAN-{epoch:02d}-{fid}",
                                                monitor="fid", mode='min')],
        logger=wandb_logger,
        # logger=False,
        strategy='ddp',
        accelerator='gpu',
        # gpus=[5],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=dm)


    # img = torch.randn(100, 3, 64, 64)
    # label = torch.randint(0,10, (100, ))
    # ae =
    # output = ae(img, label)
