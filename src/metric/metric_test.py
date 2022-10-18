from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode, Normalize
import torch
from tqdm import tqdm
import scipy
from scipy import linalg


cifar10_train = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=True,
                        transform=Compose([ToTensor(),
                                           Normalize([0.5, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5])]))
cifar10_train_loader = DataLoader(cifar10_train, 128)
# img, label = iter(cifar10_loader).__next__()

cifar10_test = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=False,
                  transform=Compose([Resize(size=299, interpolation=InterpolationMode.BILINEAR),
                                     ToTensor(),
                                     # Normalize(mean=model.mean, std=model.std)
                                     ]))
cifar10_test_loader = DataLoader(cifar10_test, 128)



feat_l_test = []
logit_l_test = []
for img, label in tqdm(cifar10_test_loader):
    img = (img * 255).to(torch.uint8).cuda()
    feat, logit = eval_model(img)
    feat_l_test.append(feat)
    logit_l_test.append(logit)



def inception_scroe(features, splits=10):
    features = torch.cat(features)
    idx = torch.randperm(features.shape[0])
    features = features[idx]

    # calculate probs and logits
    prob = features.softmax(dim=1)
    log_prob = features.log_softmax(dim=1)

    # split into groups
    prob = prob.chunk(splits, dim=0)
    log_prob = log_prob.chunk(splits, dim=0)

    # calculate score per split
    mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
    kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
    kl = torch.stack(kl_)

    return kl.mean(), kl.std()

inception_scroe(logit_l)



def sqrtm(input_data):
    m = input_data.detach().cpu().numpy().astype(np.float_)
    scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
    sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
    return sqrtm

def _compute_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    r"""
    Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def frechet_inception_distance(real_features, fake_features):
    """Calculate FID score based on accumulated extracted features from the two distributions."""
    real_features = torch.cat(real_features)
    fake_features = torch.cat(fake_features)
    # computation is extremely sensitive so it needs to happen in double precision
    orig_dtype = real_features.dtype
    real_features = real_features.double()
    fake_features = fake_features.double()

    # calculate mean and covariance
    n1 = real_features.shape[0]
    n2 = fake_features.shape[0]
    mean1 = real_features.mean(dim=0)
    mean2 = fake_features.mean(dim=0)
    diff1 = real_features - mean1
    diff2 = fake_features - mean2
    cov1 = 1.0 / (n1 - 1) * diff1.t().mm(diff1)
    cov2 = 1.0 / (n1 - 1) * diff2.t().mm(diff2)

    # compute fid
    return _compute_fid(mean1, cov1, mean2, cov2).to(orig_dtype)

def _compute_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
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


def frechet_inception_distance(real_features, fake_features):
    """Calculate FID score based on accumulated extracted features from the two distributions."""
    real_features = torch.cat(real_features)
    fake_features = torch.cat(fake_features)
    # computation is extremely sensitive so it needs to happen in double precision
    orig_dtype = real_features.dtype
    real_features = real_features.double().cpu().numpy()
    fake_features = fake_features.double().cpu().numpy()

    # calculate mean and covariance
    # n1 = real_features.shape[0]
    # n2 = fake_features.shape[0]
    # mean1 = real_features.mean(dim=0)
    # mean2 = fake_features.mean(dim=0)
    # diff1 = real_features - mean1
    # diff2 = fake_features - mean2
    # cov1 = 1.0 / (n1 - 1) * diff1.t().mm(diff1)
    # cov2 = 1.0 / (n1 - 1) * diff2.t().mm(diff2)

    mean1 = np.mean(real_features, axis=0)
    mean2 = np.mean(fake_features, axis=0)

    cov1 = np.cov(real_features, rowvar=False)
    cov2 = np.cov(fake_features, rowvar=False)

    # compute fid
    return _compute_fid(mean1, cov1, mean2, cov2)


frechet_inception_distance(feat_l_test, feat_l)
frechet_inception_distance(feat_l, feat_l_test)
frechet_inception_distance(fid.real_features, fid.fake_features)


real_features = torch.cat(feat_l)
real_features = real_features.double()
n = real_features.shape[0]

mean = real_features.mean(dim=0)
diff = real_features - mean
cov = 1.0 / (n - 1) * diff.t().mm(diff)

torch.cov(real_features)

n = real_features.shape[0]

((torch.tensor(cifar10.data) - 128) / 128).mean()


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []


class IncetpionNet:
    def __init__(self):
        model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                          "ResNet_torch": "pytorch/vision:v0.10.0",
                          "SwAV_torch": "facebookresearch/swav:main"}
        model_names = {"InceptionV3_torch": "inception_v3",
                       "ResNet50_torch": "resnet50",
                       "SwAV_torch": "resnet50"}
        SWAV_CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar"
        SWIN_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth"


        eval_backbone = 'InceptionV3_torch'
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.eval_model = torch.hub.load(model_versions[eval_backbone],
                                model_names[eval_backbone],
                                pretrained=True).cuda()

        self.eval_model.eval()

        self.save_output = SaveOutput()

        hook_handles = []
        for name, layer in self.eval_model.named_children():
            if name == "fc":
                handle = layer.register_forward_pre_hook(self.save_output)
                hook_handles.append(handle)

    def __call__(self, x):
        logit = self.eval_model(x)
        feat = self.save_output.outputs[0][0]
        self.save_output.clear()

        return logit, feat



# class Resizer:
#     def __init__(self, size):
#         self.output_size = (size, size)
#         self.filter = 'bilinear'
#
#     def __call__(self, x):
#         x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
#         x = F.interpolate(x, size=self.output_size, mode=self.filter, align_corners=False)
#         x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
#         return x

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

model = IncetpionNet()


cifar10_train = CIFAR10(root='/shared_hdd/sin/dataset/', download=True, train=True,
                  transform=Compose([Resize(size=299, interpolation=InterpolationMode.BILINEAR),
                                     ToTensor(),
                                     # Normalize(mean=model.mean, std=model.std)
                                     ]))
cifar10_train_loader = DataLoader(cifar10_train, 128)
img, label = iter(cifar10_loader).__next__()




logit_l = []
ref_l = []
for img, label in tqdm(cifar10_loader):
    img = (img*255).to(torch.uint8).cuda()
    # img = img.cuda()
    with torch.no_grad():
        logit, ref = model(img)
    logit = F.softmax(logit, dim=1).cpu().detach().numpy()
    ref = ref.cpu().detach().numpy()
    logit_l.append(logit)
    ref_l.append(ref)

logit_l = np.concatenate(logit_l)
ref_l = np.concatenate(ref_l)

from src.metric.ins import get_inception_score
m, std = get_inception_score(logit_l)

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

ins = InceptionScore().cuda()
ins.features = logit_l
ins.compute()

for img, label in tqdm(cifar10_loader):
    img = (img*255).to(torch.uint8).cuda()
    ins.update(img)

# save_output.outputs[0][0].size()

from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
metric_model = FeatureExtractorInceptionV3(name='inception-v3-compat', features_list=['logits_unbiased'])



output = metric_model(torch.tensor(cifar10.data[:10]).permute(0,3,1,2) )


fid = FrechetInceptionDistance().cuda()
for img, label in tqdm(cifar10_train_loader):
    img = (img*255).to(torch.uint8).cuda()
    fid.update(img, real=True)

for img, label in tqdm(cifar10_test_loader):
    img = (img*255).to(torch.uint8).cuda()
    fid.update(img, real=False)

fid.compute()

