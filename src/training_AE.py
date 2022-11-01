import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from utils.dataset import DataModule_
from pytorch_lightning.loggers import WandbLogger
from models import Encoder, Decoder, Embedding_labeled_latent
from argparse import ArgumentParser
import wandb
# wandb.login(key = '6afc6fd83ea84bf316238272eb71ef5a18efd445')
# wandb.init(project='MYGAN', name='BEGAN-AE')


class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder, embedding, lr, betas, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()
        # self.encoder = Encoder(img_dim=img_dim, latent_dim=latent_dim)
        # self.decoder = Decoder(img_dim=img_dim, latent_dim=latent_dim)
        # self.embedding = Embedding_labeled_latent(latent_dim=latent_dim, num_class=num_class)
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.lr = lr
        self.betas = betas

    def forward(self, img, label):
        x = self.encoder(img)
        x = self.embedding(x, label)
        x = self.decoder(x)
        return x

    def training_step(self, batch):
        img, label = batch

        y_hat = self(img, label)
        loss = self.mes_loss(y_hat, img)

        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': loss, 'y_hat': y_hat}

    def training_epoch_end(self, outputs):
        y_hat = torch.cat([out['y_hat'] for out in outputs])
        sample_imgs = [y_hat[-40:]]
        # grid = make_grid(sample_imgs).permute(1,2,0)
        self.logger.log_image("img", sample_imgs, self.trainer.current_epoch)
        # wandb_logger.log_image("img", sample_imgs, self.trainer.current_epoch)
        # for out in outputs:
        #     print(out['y_hat'].shape)
        # y_hat = outputs[0]['y_hat']
        # sample_imgs = y_hat[:10]
        # print(type(y_hat))
        # y_hat = torch.stack(y_hat)
        # print(y_hat.shape)
        # print(y_hat[:10].shape )
        # print(torch.tensor(outputs['y_hat']).shape)
        # sample_imgs = outputs['y_hat'])
        # print(sample_imgs.shape)
        # sample_imgs = [:10]
        # grid = make_grid(sample_imgs)
        # self.logger.experiment.add_image('imgs', grid)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, betas=self.betas)

    def mes_loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['encoder'] = encoder.state_dict()
        checkpoint['decoder'] = decoder.state_dict()
        checkpoint['embedding'] = embedding.state_dict()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AE")
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--betas', type=tuple, default=(0.5, 0.9))
        return parent_parser


if __name__ == "__main__":
    # decoder = Decoder(3, 128)
    # z = torch.randn(100, 128)
    # output = decoder(z)
    # print(output.shape)

    # encoder = Encoder(3, 128)
    # img = torch.randn(100, 3, 64, 64)
    # output = encoder(img)
    # print(output.shape)
    #
    #
    # z = torch.randn(100, 128)
    # img = torch.randn(100, 3, 64, 64)
    # label = torch.randint(0,10, (100))
    # le = Embedding_labeled_latent(128, 10)
    # output = le(z, label)

    # dm = DataModule_(path_train='/home/dblab/sin/save_files/refer/ebgan_cifar10', batch_size=128)
    # dm = DataModule_(path_train='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train', batch_size=128)

    parser = ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=10, required=False)
    parser.add_argument("--img_dim", type=int, default=3, required=False)
    parser.add_argument("--latent_dim", type=int, default=128, required=False)
    parser.add_argument("--data_name", type=str, default='imb_CIFAR10',
                        choices=['imb_CIFAR10', 'imb_MNIST', 'imb_FashionMNIST'], required=False)
    parser = Autoencoder.add_model_specific_args(parser)

    args = parser.parse_args("")
    dm = DataModule_.from_argparse_args(args)

    encoder = Encoder(**vars(args))
    decoder = Decoder(**vars(args))
    embedding = Embedding_labeled_latent(**vars(args))
    model = Autoencoder(encoder, decoder, embedding, **vars(args))

    # decoder_weight = {'.'.join(k.split('.')[1:]):v for k, v in model.state_dict().items() if 'decoder' in k}
    # decoder_weight.keys()
    # for i, _ in model.state_dict().items():
    #     if 'decoder' in i:
    #         print('.'.join(i.split('.')[1:]))
    #
    # if 'encoder' in model.state_dict():
    #     print('True')

    # model

    wandb_logger = WandbLogger(project='MYGAN', name=f'BEGAN-AE-{args.data_name}', log_model=True)
    wandb_logger.watch(model, log='all')
    wandb.define_metric('train/loss', summary='min')
    trainer = pl.Trainer(
        default_root_dir=f'/shared_hdd/sin/save_files/EBGAN/AE/{args.data_name}',
        fast_dev_run=False,
        max_epochs=30,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="train/loss", mode='min')],
        logger=wandb_logger,
        strategy='ddp_find_unused_parameters_false',
        accelerator='gpu',
        gpus=1,
        # check_val_every_n_epoch=10
    )
    trainer.fit(model, datamodule=dm)


    # img = torch.randn(100, 3, 64, 64)
    # label = torch.randint(0,10, (100, ))
    # ae =
    # output = ae(img, label)

    # weight = torch.load("/shared_hdd/sin/save_files/EBGAN/MYGAN/308uyxwa/checkpoints/epoch=24-step=1875.ckpt")


# import wandb
# from pathlib import Path
# run = wandb.init()
# artifact = run.use_artifact('sinaenjuni/MYGAN/model-3vkiuonh:v0', type='model')
# artifact_dir = artifact.download()
#
# torch.load(Path(artifact_dir) / "model.ckpt")