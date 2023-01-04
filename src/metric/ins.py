
import torch
from tqdm import tqdm

def inception_softmax(eval_model, images, quantize):
    with torch.no_grad():
        embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
        ps = torch.nn.functional.softmax(logits, dim=1)
    return ps

def calculate_kl_div(ps_list, splits=10):
    # scores = []
    # num_samples = ps.shape[0]
    # with torch.no_grad():
    #     for j in range(splits):
    #         part = ps[(j * num_samples // splits):((j + 1) * num_samples // splits), :]
    #         kl = part * (torch.log(part) - torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
    #         kl = torch.mean(torch.sum(kl, 1))
    #         kl = torch.exp(kl)
    #         scores.append(kl.unsqueeze(0))
    #
    #     scores = torch.cat(scores, 0)
    #     m_scores = torch.mean(scores).detach().cpu().numpy()
    #     m_std = torch.std(scores).detach().cpu().numpy()
    # return m_scores, m_std
    parts = ps_list.chunk(splits)
    means = [torch.mean(part, 0, keepdim=True) for part in parts]
    kl_ = [part * (torch.log(part) - torch.log(mean)) for part, mean in zip(parts, means)]
    kl_ = [torch.mean(torch.sum(kl_, dim=1)) for kl_ in kl_]
    kl_ = [torch.exp(kl_) for kl_ in kl_]
    scores = torch.stack(kl_)
    m_scores = torch.mean(scores).detach().cpu().numpy()
    m_std = torch.std(scores).detach().cpu().numpy()
    return m_scores.item(), m_std.item()

def inception_scores(eval_loader, eval_model, quantize=True):
    ps_list = []
    label_list = []
    for img, label in tqdm(eval_loader, desc="Calculating Inception Score"):
        img = img.cuda()
        with torch.no_grad():
            ps = inception_softmax(eval_model, img, quantize=quantize)
            # embedding, logit = eval_model.get_outputs(img, quantize=True)
            # ps = torch.nn.functional.softmax(logit, dim=1)
        ps_list.append(ps)
        label_list.append(label)

    ps_list = torch.cat(ps_list)
    label_list = torch.cat(label_list)
    ps_list_per_cls = [ps_list[torch.where(label_list == idx)] for idx in torch.unique(label_list)] + [ps_list]

    score, std = zip(*[calculate_kl_div(ps_list_per_cls_, splits=10) for ps_list_per_cls_ in ps_list_per_cls])

    return score, std, label_list


def inception_score(eval_loader, eval_model, quantize=True):
    ps_list = []
    label_list = []
    for img, label in tqdm(eval_loader, desc="Calculating Inception Score"):
        img = img.cuda()
        with torch.no_grad():
            ps = inception_softmax(eval_model, img, quantize=quantize)
            # embedding, logit = eval_model.get_outputs(img, quantize=True)
            # ps = torch.nn.functional.softmax(logit, dim=1)
        ps_list.append(ps)
        label_list.append(label)

    ps_list = torch.cat(ps_list)
    label_list = torch.cat(label_list)
    ps_list_per_cls = [ps_list[torch.where(label_list == idx)] for idx in torch.unique(label_list)] + [ps_list]

    score, std = zip(*[calculate_kl_div(ps_list_per_cls_, splits=10) for ps_list_per_cls_ in ps_list_per_cls])

    return score, std, label_list


def calculate_inception_score(ps_list, label_list):
    ps_list_per_cls = [ps_list[torch.where(label_list == idx)] for idx in torch.unique(label_list)] + [ps_list]
    score, std = zip(*[calculate_kl_div(ps_list_per_cls_, splits=10) for ps_list_per_cls_ in ps_list_per_cls])

    return score, std, label_list




if __name__ == '__main__':
    import torch
    from src.metric.inception_net import EvalModel

    eval_model = EvalModel(torch.device('cuda'))


    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    cifar10 = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=False,
                            transform=Compose([ToTensor(),
                                               Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])]))

    cifar10_img = ImageFolder(root='/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train',
                              transform=Compose([ToTensor(),
                                                 Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])]))


    eval_loader = DataLoader(cifar10, 128)
    cifar10_img_loader = DataLoader(cifar10_img, 128)
    # img, label = iter(cifar10_loader).__next__()
    # embedding, logit = eval_model.get_outputs(img, quantize=True)
    m_score, m_std  = inception_score(eval_loader=eval_loader, eval_model=eval_model)
    m_score2, m_std2 = inception_score(eval_loader=cifar10_img_loader, eval_model=eval_model)

