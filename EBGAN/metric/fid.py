import numpy as np
from scipy import linalg
from tqdm import tqdm
import torch


def frechet_inception_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        "Training and binary_classification_test mean vectors have different lengths."
    assert sigma1.shape == sigma2.shape, \
        "Training and binary_classification_test covariances have different dimensions."

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def feature_stack_and_calculate_mu_sigma(eval_loader, eval_model, quantize=True):
    feat_list = []
    label_list = []
    for img, label in tqdm(eval_loader, desc="Calculating mu and sigma"):
        img = img.cuda()
        label_list.append(label)
        with torch.no_grad():
            embeddings, logits = eval_model.get_outputs(img, quantize=quantize)
            feat_list.append(embeddings)

    feat_list = torch.cat(feat_list)
    feat_list = feat_list.detach().cpu().numpy().astype(np.float64)
    label_list = np.concatenate(label_list)

    mu, sigma, label_list = calculate_mu_sigma(feat_list, label_list)

    # feat_list_per_cls = [feat_list[np.where(label_list == idx)] for idx in np.unique(label_list)] + [feat_list]

    # mu = [np.mean(feat_list_per_cls_, axis=0) for feat_list_per_cls_ in feat_list_per_cls]
    # sigma = [np.cov(feat_list_per_cls_, rowvar=False) for feat_list_per_cls_ in feat_list_per_cls]

    # mu = np.mean(feat_list, axis=0)
    # sigma = np.cov(feat_list, rowvar=False)

    return mu, sigma, label_list


def calculate_mu_sigma_with_list(feat_list, label_list):
    feat_list_per_cls = [feat_list[np.where(label_list == idx)] for idx in np.unique(label_list)] + [feat_list]

    mu = [np.mean(feat_list_per_cls_, axis=0) for feat_list_per_cls_ in feat_list_per_cls]
    sigma = [np.cov(feat_list_per_cls_, rowvar=False) for feat_list_per_cls_ in feat_list_per_cls]

    return mu, sigma, label_list


def calculate_mu_sigma(feat_list):
    mu = np.mean(feat_list, axis=0)
    sigma = np.cov(feat_list, rowvar=False)
    return mu, sigma


if __name__ == "__main__":
    # import torch
    # from tqdm import tqdm
    from src.metric.inception_net import EvalModel

    eval_model = EvalModel(torch.device('cuda'))


    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torch.utils.data import DataLoader

    cifar10_train = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=False,
                            transform=Compose([ToTensor(),
                                               Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])]))
    cifar10_test = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=False,
                            transform=Compose([ToTensor(),
                                               Normalize([0.5, 0.5, 0.5],
                                                         [0.5, 0.5, 0.5])]))
    eval_loader = DataLoader(cifar10_train, 128)
    eval_loader2 = DataLoader(cifar10_test, 128)
    # img, label = iter(cifar10_loader).__next__()
    # embedding, logit = eval_model.get_outputs(img, quantize=True)


    mu, sigma, label_list = calculate_mu_sigma(eval_loader, eval_model)
    mu1, sigma1, label_list1 = calculate_mu_sigma(eval_loader2, eval_model)

    for (idx_mu, mu_), (idx_sigma, sigma_) in zip(mu.items(), sigma.items()):
        print(idx_mu, idx_sigma)
        frechet_inception_distance(mu, sigma, mu1, sigma1)

    frechet_inception_distance(mu, sigma, mu1, sigma1)
    frechet_inception_distance(mu1, sigma1, mu, sigma)

    for mu1, sigma1, mu2, sigma2 in zip(mu.values(), sigma.values(), mu1.values(), sigma1.values()):
        fid = frechet_inception_distance(mu1, sigma1, mu2, sigma2)
        print(fid)
        # print(mu1)