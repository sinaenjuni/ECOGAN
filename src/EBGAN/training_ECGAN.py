import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from collections import OrderedDict

from dataset import DataModule_
from torchvision.utils import make_grid
import wandb
from pytorch_lightning.loggers import WandbLogger
from models import Encoder, Decoder, Embedding_labeled_latent
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from metric.inception_net import EvalModel
from metric.ins import calculate_kl_div
from metric.fid import calculate_mu_sigma, frechet_inception_distance
import numpy as np

class Generator(nn.Module):
    def __init__(self, img_dim, latent_dim, num_class):
        super(Generator, self).__init__()

        self.embedding = Embedding_labeled_latent(latent_dim=latent_dim, num_class=num_class)
        self.decoder = Decoder(img_dim=img_dim, latent_dim=latent_dim)


    def forward(self, z, label):
        latent = self.embedding(z, label)
        gened_img = self.decoder(latent)

        return gened_img


class Discriminator_EC(nn.Module):
    def __init__(self, img_dim, latent_dim, num_class, d_embed_dim):
        super(Discriminator_EC, self).__init__()

        self.encoder = Encoder(img_dim, latent_dim)
        self.linear1 = nn.Linear(in_features=self.encoder.dims[3], out_features=1, bias=True)
        self.linear2 = nn.Linear(in_features=self.encoder.dims[3], out_features=d_embed_dim, bias=True)
        self.embedding = nn.Embedding(num_embeddings=num_class, embedding_dim=d_embed_dim)

        # self.embedding = nn.Sequential(nn.Embedding(num_embeddings=num_class, embedding_dim=512),
        #                                nn.Flatten(),
        #                                nn.Linear(512, 256 * (4 * 4)),
        #                                nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # self.discriminator = nn.Linear(256 * (4*4), 1)


    def forward(self, img, label):
        x = self.encoder.getFeatures(img)
        x = torch.sum(x, dim=[2,3])
        adv_output = self.linear1(x)

        embed_data = self.linear2(x)
        embed_label = self.embedding(label)

        embed_data = F.normalize(embed_data, dim=1)
        embed_label = F.normalize(embed_label, dim=1)

        return adv_output, embed_data, embed_label



class ExhustiveContrastiveLoss(nn.Module):
    def __init__(self, num_classes, temperature):
        super(ExhustiveContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long)

    # def make_index_matrix(self, labels):
    #     labels = labels.detach().cpu().numpy()
    #     num_samples = labels.shape[0]
    #     mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0
    #
    #     for c in range(self.num_classes):
    #         c_indices = np.where(labels==c)
    #         mask_multi[c, c_indices] = target
    #     return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed_data, embed_label, label, **_):
        device = torch.device(embed_data.device)
        f2f_sim = self.calculate_similarity_matrix(embed_data, embed_data)
        f2f_sim = self._remove_diag(f2f_sim)
        f2f_max, _ = torch.max(f2f_sim, dim=1, keepdim=True)

        f2f_logits = f2f_sim - f2f_max.detach()
        f2f_logits = torch.exp(f2f_logits / self.temperature)

        pos_mask_redia = self._remove_diag(self._make_neg_removal_mask(label)[label]).to(device)
        f2f_logits_pos_only = pos_mask_redia * f2f_logits

        # emb2proxy = torch.exp(self.cosine_similarity(embed_data, embed_label) / self.temperature)

        e2p_sim = self.calculate_similarity_matrix(embed_data, embed_label)
        e2p_max, _ = torch.max(e2p_sim, dim=1, keepdim=True)

        e2p_logits = e2p_sim - e2p_max.detach()
        e2p_logits = torch.exp(e2p_logits / self.temperature)

        pos_mask = self._make_neg_removal_mask(label)[label].to(device)
        e2p_logits_pos_only = pos_mask * e2p_logits


        # numerator = emb2proxy + sim_pos_only.sum(dim=1)
        numerator = e2p_logits_pos_only.sum(dim=1) + f2f_logits_pos_only.sum(dim=1)
        denomerator = e2p_logits.sum(dim=1) + f2f_logits.sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, img_dim, num_class, pre_train_path=None):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        # self.fid = FrechetInceptionDistance()

        self.eval_model = EvalModel()
        self.G = Generator(img_dim=img_dim, latent_dim=latent_dim, num_class=num_class)
        self.D = Discriminator_EC(img_dim=img_dim, latent_dim=latent_dim, num_class=num_class, d_embed_dim=256)

        mu_sigma_train = np.load('/shared_hdd/sin/save_files/img_cifar10.npz')
        self.mu_original, self.sigma_original = mu_sigma_train['mu'][-1], mu_sigma_train['sigma'][-1]

        if pre_train_path is not None:
            # path = '/home/dblab/git/VAE-GAN/src/EBGAN/GAN/1gqzkv1e/checkpoints/epoch=28-step=1508.ckpt'
            weights = torch.load(pre_train_path)

            self.G.load_state_dict(weights['state_dict'], strict=False)
            self.D.load_state_dict(weights['state_dict'], strict=False)

        self.eco_loss = ExhustiveContrastiveLoss(num_classes=num_class, temperature=1.0)

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
            fake_logits, _, _ = self.D(fake_imgs, fake_labels)
            real_logits, embed_data, embed_label = self.D(real_imgs, real_labels)


            d_adv_loss = self.d_loss(real_logits, fake_logits)
            d_cond_loss = self.eco_loss(embed_data, embed_label, real_labels)
            gp = self.compute_gradient_penalty(real_imgs, fake_imgs, real_labels)
            d_loss = d_adv_loss + gp * 10.0 + d_cond_loss

            self.log('d_loss', d_loss, prog_bar=True, logger=True, on_epoch=True)
            return d_loss

        if optimizer_idx == 1:
            z = torch.randn(real_imgs.size(0), self.latent_dim).to(self.device)
            fake_labels = (torch.rand((batch_size,)) * 10).to(torch.long).to(self.device)

            gen_imgs = self(z, fake_labels)
            gen_logits, gen_embed_data, gen_embed_label = self.D(gen_imgs, fake_labels)
            g_adv_loss = self.g_loss(gen_logits)
            g_cond_loss = self.eco_loss(gen_embed_data, gen_embed_label, fake_labels)

            g_loss = g_adv_loss + g_cond_loss
            self.log('g_loss', g_loss, prog_bar=True, logger=True, on_epoch=True)
            return g_loss


    # def training_epoch_end(self, outputs):
    #     z = torch.randn((100, self.latent_dim)).to(self.device)
    #     label = torch.arange(0, 9, dtype=torch.long).repeat(10).to(self.device)
    #     gened_imgs = self(z, label)
    #     self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        # z = torch.randn((imgs.size(0), self.latent_dim)).to(self.device)
        # label = torch.arange(0, 9, dtype=torch.long).repeat(100).to(self.device)
        # gend_imgs = self(z, labels)
        #
        # imgs = (((imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        # gend_imgs = (((gend_imgs * 0.5) + 0.5) * 255.).to(torch.uint8)
        # self.fid.update(imgs, real=True)
        # self.fid.update(gend_imgs, real=False)

        with torch.no_grad():
            # label = targets[i*batch_size : batch_size * (i + 1)].cuda()
            # z = torch.randn(label.size(0), latent_dim).cuda()
            z = torch.randn((imgs.size(0), self.latent_dim)).to(self.device)
            img_fake = self(z, labels)

            embeddings, logits = self.eval_model(img_fake, quantize=True)
            ps = torch.nn.functional.softmax(logits, dim=1)
            # ps_list.append(ps)
            # em_list.append(embeddings)

        return {'ps': ps, 'embedding': embeddings}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        ps = torch.cat([output['ps'] for output in outputs])
        embedding = torch.cat([output['embedding'] for output in outputs])
        # print(ps.size())
        # print(embedding.size())

        ins_score, ins_std = calculate_kl_div(ps, 10)
        mu_target, sigma_target = calculate_mu_sigma(embedding.cpu().numpy())
        fid_score = frechet_inception_distance(self.mu_original, self.sigma_original, mu_target, sigma_target)

        # print('ins_score', ins_score)
        # print('fid_score', fid_score)
        self.log_dict({'fid': fid_score, 'ins_score': ins_score}, logger=True, prog_bar=True, on_epoch=True)
        # print('valid_fid_epoch', self.fid.compute())
        # self.log('fid', self.fid.compute(), logger=True, prog_bar=True, on_epoch=True)
        # self.fid.reset()

        # z = torch.randn((100, self.latent_dim)).to(self.device)
        # label = torch.arange(0, 10, dtype=torch.long).repeat(10).to(self.device)
        # gened_imgs = self(z, label)
        # self.logger.log_image("img", [gened_imgs], self.trainer.current_epoch)


    def configure_optimizers(self):
        optimizer_g = Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.9))
        optimizer_d = Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.9))

        return [{'optimizer': optimizer_d, 'frequency': 5},
                {'optimizer': optimizer_g, 'frequency': 1}]


    def mes_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def d_loss(self, real_logits, fake_logits):
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return real_loss + fake_loss

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

        d_interpolates, _, _ = self.D(interpolates, real_labels)

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

    # dm = DataModule_(path_train='/home/dblab/sin/save_files/refer/ebgan_cifar10', batch_size=128)
    dm = DataModule_(path_train='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train', batch_size=128, num_workers=4)
    model = GAN(latent_dim=128, img_dim=3, num_class=10)

    # model

    # wandb.login(key='6afc6fd83ea84bf316238272eb71ef5a18efd445')
    # wandb.init(project='MYGAN', name='BEGAN-GAN')

    wandb_logger = WandbLogger(project='MYGAN', name='ECOGAN')
    trainer = pl.Trainer(
        # fast_dev_run=True,
        default_root_dir='/shared_hdd/sin/save_files/EBGAN/',
        max_epochs=100,
        # callbacks=[EarlyStopping(monitor='val_loss')],
        callbacks=[pl.callbacks.ModelCheckpoint(filename="EBGAN-{epoch:02d}-{fid}",
                                                monitor="fid", mode='min')],
        logger=wandb_logger,
        # logger=False,
        strategy='ddp',
        accelerator='gpu',
        gpus=[4,5,6],
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=dm)


    # img = torch.randn(100, 3, 64, 64)
    # label = torch.randint(0,10, (100, ))
    # ae =
    # output = ae(img, label)