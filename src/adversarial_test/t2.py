import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
# from src.adversarial_test.model import Encoder, Decoder
from torch.nn.utils import spectral_norm
from tqdm import tqdm
from src.metric import fid, ins
from src.metric.inception_net import EvalModel
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from src.EBGAN.training_EBGAN import Generator, Discriminator

img_dim = 3
num_classes = 10
latent_dim = 128
batch_size = 128
lr_g = 0.0002
lr_d = 0.0002
num_epoch = 100


# class MyFashionMNIST(FashionMNIST):
#     def __init__(self, cls_target, **args):
#         super(MyFashionMNIST, self).__init__(**args)
#         if cls_target is not None:
#             label_target = []
#             image_target = []
#             for cls, count in cls_target.items():
#                 idx_target = torch.where(self.targets == cls)[0]
#                 label_target.append(self.targets[idx_target][:count])
#                 image_target.append(self.data[idx_target][:count])
#
#             self.data = torch.cat(image_target)
#             self.targets = torch.cat(label_target)
#
#
# dataset = MyFashionMNIST(root='/shared_hdd/sin/dataset',
#                          cls_target={0: 6000,
#                                      1: 6000},
#                          download=True,
#                          train=True,
#                         transform=Compose([ToTensor(),
#                                             Resize(64),
#                                             Normalize([0.5],
#                                                      [0.5])]))
#

# dataset = FashionMNIST(root='/shared_hdd/sin/dataset', download=False, train=True,
#                 transform=Compose([ToTensor(),
#                                    # Resize(64),
#                                    Normalize([0.5],
#                                              [0.5])]))

# dataset = CIFAR10(root='/shared_hdd/sin/dataset', download=False, train=True,
#                 transform=Compose([ToTensor(),
#                                    Resize(64),
#                                    Normalize([0.5, 0.5, 0.5],
#                                              [0.5, 0.5, 0.5])]))

dataset = ImageFolder(root='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
                      transform=Compose([ToTensor(),
                                         Resize(64),
                                         Normalize([0.5, 0.5, 0.5],
                                                   [0.5, 0.5, 0.5])]))

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# img, label = iter(data_loader).__next__()

targets = torch.tensor(dataset.targets)
classes, class_count = torch.unique(targets, return_counts=True)

idx = torch.cat([torch.where(targets == cls_)[0][:10] for cls_ in classes])
target_img = torch.stack([dataset[idx_][0] for idx_ in idx])
grid = make_grid(target_img, normalize=True, nrow=10)
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.show()


eval_model = EvalModel(device=torch.device('cuda'))

mu_sigma_train = np.load('/shared_hdd/sin/save_files/img_cifar10.npz')
mu_train, sigma_train = mu_sigma_train['mu'], mu_sigma_train['sigma']


# def save_train_data_mu_sigma():
#     mu, sigma, label_list = fid.calculate_mu_sigma(data_loader, eval_model, True)
#     np.savez('/shared_hdd/sin/save_files/img_cifar10.npz', mu=mu, sigma=sigma, label_list=label_list)

# fid.frechet_inception_distance()

# m_scroe, std, label_list = ins.inception_score(data_loader, eval_model, quantize=True)



G = Generator(img_dim, latent_dim, num_classes).cuda()
D = Discriminator(img_dim, latent_dim, num_classes).cuda()


# latent = torch.randn(batch_size, latent_dim).cuda()
# img_fake = decoder(latent).detach()
# adv_output = encoder(img.cuda())
# print(img_fake.size())

optimizer_g = torch.optim.Adam(params=G.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(params=D.parameters(), lr=lr_d, betas=(0.5, 0.999))


def d_loss_fn(real_logits, fake_logits, wrong_logits):
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    wrong_loss = F.binary_cross_entropy_with_logits(wrong_logits, torch.zeros_like(wrong_logits))

    return real_loss + fake_loss + wrong_loss

def g_loss_fn(fake_logits):
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return fake_loss

def compute_gradient_penalty(real_samples, fake_samples, real_labels):
    # real_samples = real_samples.reshape(real_samples.size(0), 1, 28, 28).to(device)
    # fake_samples = fake_samples.reshape(fake_samples.size(0), 1, 28, 28).to(device)

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    alpha = torch.randn(real_samples.size(0), 1, 1, 1).cuda()
    # Get random interpolation between real and fake samples
    # interpolates = (alpha * real_samples.data + ((1 - alpha) * fake_samples.data)).requires_grad_(True)
    diff = fake_samples - real_samples
    interpolates = (real_samples + alpha * diff).requires_grad_(True)

    d_interpolates = D(interpolates, real_labels)

    weights = torch.ones(d_interpolates.size()).cuda()
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


for epoch in range(num_epoch):
    loader_tqdm = tqdm(data_loader)
    for img, label in loader_tqdm:
        img_real = img.cuda()
        label_real = label.cuda()
        z = torch.randn(img_real.size(0), latent_dim).cuda()
        label_fake = (torch.rand( (label_real.size(0),)) * num_classes).to(torch.long).cuda()
        label_wrong = (torch.rand( (label_real.size(0),)) * num_classes).to(torch.long).cuda()

        for i in range(5):
            optimizer_d.zero_grad()
            img_fake = G(z, label_fake).detach()
            logit_real = D(img_real, label_real)
            logit_fake = D(img_fake, label_fake)
            logit_wrong = D(img_real, label_wrong)
            # y_real_, y_fake_ = torch.ones_like(adv_real).cuda(), torch.zeros_like(adv_fake).cuda()
            gp  = compute_gradient_penalty(img_real, img_fake, label_real)
            d_loss = d_loss_fn(logit_real, logit_fake, logit_wrong) + gp * 10.0
            d_loss.backward()
            optimizer_d.step()


        optimizer_g.zero_grad()
        img_fake = G(z, label_fake)
        gen_logit = D(img_fake, label_fake)
        g_loss = g_loss_fn(gen_logit)
        g_loss.backward()
        optimizer_g.step()

        loader_tqdm.set_description(f'epoch {epoch}')
        loader_tqdm.set_postfix({'d_loss': d_loss.item(),
                                 'g_loss': g_loss.item()})

    with torch.no_grad():
        z = torch.randn(10, latent_dim).repeat(num_classes, 1).cuda()
        label = torch.arange(0, num_classes).view(-1, 1).repeat(1, 10).flatten().cuda()
        grid = make_grid(G(z, label), normalize=True, nrow=10)
        plt.imshow(grid.permute(1,2,0).cpu())
        plt.show()


    ps_list = []
    em_list = []

    for i in tqdm(range((len(targets)//batch_size)+1), desc="Calculate eval FID and IS"):
        with torch.no_grad():
            label = targets[i*batch_size : batch_size * (i + 1)].cuda()
            z = torch.randn(label.size(0), latent_dim).cuda()
            img_fake = G(z, label)

            embeddings, logits = eval_model.get_outputs(img_fake, quantize=True)
            ps = torch.nn.functional.softmax(logits, dim=1)
            ps_list.append(ps)
            em_list.append(embeddings)

    ps_list = torch.cat(ps_list)
    em_list = torch.cat(em_list)

    em_list = em_list.cpu().numpy().astype(np.float64)
    mu_fake, sigma_fake, _ = fid.calculate_mu_sigma(em_list, targets)
    #
    #
    # for mu_train_, sigma_train_, mu_fake_, sigma_fake_ in zip(mu_train, sigma_train, mu_fake, sigma_fake):
    #     # print(mu_train_, sigma_train_, mu_fake_, sigma_fake_)
    #     fid_score = fid.frechet_inception_distance(mu_train_, sigma_train_, mu_fake_, sigma_fake_)
    #     print(fid_score)
    #
    #
    # score, std, _ = ins.calculate_inception_score(ps_list, targets)



    # fid.frechet_inception_distance(mu1=mu_train, sigma1=sigma_train, mu2=)
    # m_scroe, std, label_list = ins.inception_score(data_loader, eval_model, quantize=True)

    # print(d_loss, g_loss)









