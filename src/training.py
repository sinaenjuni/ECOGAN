import importlib
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda
from torch.utils.data import DataLoader
import wandb
from argparse import ArgumentParser
import numpy as np
from utils import misc
from datetime import datetime
from utils.misc import load_configs_init


args = load_configs_init()
logger = wandb.init(project="eval_cls", entity="sinaenjuni")

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
                          num_workers=8,
                          pin_memory=True,
                          persistent_workers=True)


model_module = importlib.import_module('models.bagan')
encoder, decoder = getattr(model_module, 'pre_training')(loader_train, logger, world_size=1, device=1, args=args)

print(model_module)


# models_module = importlib.import_module('models.bagan')
# encoder = getattr(models_module, 'Encoder')(img_dim=img_dim, latent_dim=latent_dim).cuda()
# decoder = getattr(models_module, 'Decoder')(img_dim=img_dim, latent_dim=latent_dim).cuda()

# optimizer = torch.optim.Adam([{'params': encoder.parameters(),
#                                'params': decoder.parameters()}], lr=0.00005, betas=(0.5, 0.999))
# loss_fn = nn.MSELoss()
# calc_mean_cov = getattr(models_module, 'ClassCondLatentGen')()

# start_time = datetime.now()
# steps_ae = 15000
# vis_loss_ae = 0

# iter_train = iter(loader_train)
# for steps in range(steps_ae):
#     # print(steps)
#     try:
#         img, labels = iter_train.next()
#     except StopIteration:
#         iter_train = iter(loader_train)
#         img, labels = iter_train.next()

#     img, labels = img.cuda(), labels.cuda()

#     optimizer.zero_grad()
#     outputs = encoder(img)
#     outputs = decoder(outputs)
#     loss = loss_fn(outputs, img)
#     loss.backward()
#     optimizer.step()
#     # print(loss.item())
#     vis_loss_ae += loss.item()

#     if (steps+1) % 100 == 0:
#         print(f'steps: {steps+1}/{steps_ae}({((steps+1) / steps_ae)*100:.2f}%), '
#               f'time: {misc.elapsed_time(start_time)} '
#               f'loss: {vis_loss_ae / 100:.4f}')
#         vis_loss_ae = 0

# calc_mean_cov.stacking(loader_train, encoder)


# generator = getattr(models_module, 'Generator')(img_dim=img_dim, latent_dim=latent_dim).cuda()
# discriminator = getattr(models_module, 'Discriminator')(img_dim=img_dim, latent_dim=latent_dim, num_classes=num_classes).cuda()

# generator.load_state_dict(decoder.state_dict(), strict=False)
# discriminator.load_state_dict(encoder.state_dict(), strict=False)

# optimizer_g = torch.optim.Adam(generator.parameters(),     lr=0.00005, betas=(0.5, 0.999))
# optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
# loss_fn = nn.CrossEntropyLoss()

# start_time = datetime.now()
# vis_loss_g = 0
# vis_loss_d = 0

# iter_train = iter(loader_train)
# steps_gan = 1000
# for steps in range(steps_gan):
#     print(steps)
#     try:
#         img_real, label_real = next(iter_train)
#     except StopIteration:
#         iter_train = iter(loader_train)
#         img_real, label_real = next(iter_train)

#     img_real, label_real = img_real.cuda(), label_real.cuda()

#     sample_labels = np.random.randint(0, 10, label_real.size(0))
#     cond_latent = calc_mean_cov.sampling(sample_labels).cuda()
#     fake_label = (torch.ones_like(label_real) * num_classes).cuda()

#     optimizer_d.zero_grad()
#     gen_img = generator(cond_latent).detach()
#     output_real = discriminator(img_real)
#     output_fake = discriminator(gen_img)

#     loss_d = loss_fn(output_real, label_real) + loss_fn(output_fake, fake_label)
#     loss_d.backward()
#     optimizer_d.step()
#     vis_loss_d += loss_d

#     ###########################
#     sample_labels = np.random.randint(0, 10, label_real.size(0))
#     cond_latent = calc_mean_cov.sampling(sample_labels).cuda()
#     fake_label = torch.from_numpy(sample_labels).cuda()

#     optimizer_g.zero_grad()
#     gen_img = generator(cond_latent)
#     output_fake = discriminator(gen_img)
#     loss_g = loss_fn(output_fake, fake_label)
#     loss_g.backward()
#     optimizer_g.step()
#     vis_loss_g += loss_g

#     if (steps + 1) % 100 == 0:
#         print(f'steps: {steps+1}/{steps_gan}({((steps+1) / steps_gan)*100:.2f}%), '
#               f'time: {misc.elapsed_time(start_time)}, '
#               f'loss_d: {vis_loss_d / 100:.4f}, '
#               f'loss_g: {vis_loss_g / 100:.4f}')
#         vis_loss_d = 0
#         vis_loss_g = 0



# from metric.inception_net_V2 import EvalModel


# eval_model = EvalModel()
# eval_model.eval()


# for img, labels in loader_train:





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
