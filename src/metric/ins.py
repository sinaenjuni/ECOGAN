
import torch


def inception_softmax(eval_model, images, quantize):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps


def calculate_kl_div(ps, splits):
    scores = []
    num_samples = ps.shape[0]
    with torch.no_grad():
        for j in range(splits):
            part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
            kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            kl = torch.exp(kl)
            scores.append(kl.unsqueeze(0))

        scores = torch.cat(scores, 0)
        m_scores = torch.mean(scores).detach().cpu().numpy()
        m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores, m_std


import torch
from src.metric.inception_net import EvalModel

eval_model = EvalModel(torch.device('cuda'))

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader

cifar10 = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=False,
                        transform=Compose([ToTensor(),
                                           Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])]))
cifar10_loader = DataLoader(cifar10, 64)
img, label = iter(cifar10_loader).__next__()
# embedding, logit = eval_model.get_outputs(img, quantize=True)

ps_list = []
for img, label in cifar10_loader:
    img = img.cuda()
    with torch.no_grad():
        embedding, logit = eval_model.get_outputs(img, quantize=True)
        ps = torch.nn.functional.softmax(logit, dim=1)
    ps_list.append(ps)

ps_list = torch.cat(ps_list)

calculate_kl_div(ps_list, splits=10)


splits = 10
j = 1
scores = []
num_samples = ps_list.shape[0]
with torch.no_grad():
    for j in range(splits):
        part = ps_list[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
        kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
        kl = torch.mean(torch.sum(kl, 1))
        kl = torch.exp(kl)
        scores.append(kl.unsqueeze(0))

    scores = torch.cat(scores, 0)
    m_scores = torch.mean(scores).detach().cpu().numpy()
    m_std = torch.std(scores).detach().cpu().numpy()

parts = ps_list.chunk(10)
means = [torch.mean(part, 0, keepdim=True) for part in parts]
kl_ = [part * torch.log(part)  for part, mean in zip(parts, means)]
torch.unsqueeze(torch.mean(part, 0),0).size()
torch.mean(part, 0, keepdim=True).size()

torch.tensor([[1, 2],
             [1, 2]]) - torch.tensor([1,2]).unsqueeze(0)