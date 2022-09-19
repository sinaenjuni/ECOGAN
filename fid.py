
import torch
import numpy as np
from resize import build_resizer
from torchvision.transforms import ToTensor
from inception_net import InceptionV3
import math
from tqdm import tqdm


def quantize_images(x):
    x = (x + 1)/2
    x = (255.0*x + 0.5).clamp(0.0, 255.0)
    x = x.detach().cpu().numpy().astype(np.uint8)
    return x

def resize_images(x, resizer, ToTensor, mean, std, device):
    x = x.transpose((0, 2, 3, 1))
    x = list(map(lambda x: ToTensor(resizer(x)), list(x)))
    x = torch.stack(x, 0).to(device)
    x = (x/255.0 - mean)/std
    return x

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []


class LoadEvalModel(object):
    def __init__(self, eval_backbone, device):
        self.device = device
        self.eval_backbone = eval_backbone
        self.post_resizer = "legacy"

        if self.eval_backbone == 'InceptionV3_tf':
            self.res, mean, std = 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model = InceptionV3(resize_input=False, normalize_input=False).to(self.device)

        elif self.eval_backbone == 'InceptionV3_torch':
            self.res, mean, std = 299, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = torch.hub.load("pytorch/vision:v0.10.0",
                                    "inception_v3",
                                    pretrained=True)
            self.model = self.model.to(self.device)

            self.eval_backbone = "InceptionV3_torch"
            self.save_output = SaveOutput()

            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)


        self.resizer = build_resizer(resizer=self.post_resizer, backbone=self.eval_backbone, size=self.res)
        self.totensor = ToTensor()
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):
        if quantize:
            x = quantize_images(x)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)

        if self.eval_backbone == "InceptionV3_tf":
            repres, logits = self.model(x)
        elif self.eval_backbone == "InceptionV3_torch":
            logits = self.model(x)
            if len(self.save_output.outputs) > 1:
                repres = []
                for rank in range(len(self.save_output.outputs)):
                    repres.append(self.save_output.outputs[rank][0].detach().cpu())
                repres = torch.cat(repres, dim=0).to(self.device)
            else:
                repres = self.save_output.outputs[0][0].to(self.device)
            self.save_output.clear()

        return repres, logits

def calculate_moments(data_loader, eval_model, num_generate, batch_size, quantize, disable_tqdm, fake_feats=None):
    if fake_feats is not None:
        total_instance = num_generate
        acts = fake_feats.detach().cpu().numpy()[:num_generate]
    else:
        eval_model.eval()
        total_instance = len(data_loader.dataset)
        data_iter = iter(data_loader)
        num_batches = math.ceil(float(total_instance) / float(batch_size))

        acts = []
        for i in tqdm(range(0, num_batches), disable=disable_tqdm):
            start = i * batch_size
            end = start + batch_size
            try:
                images, labels = next(data_iter)
            except StopIteration:
                break

            images, labels = images.to("cuda"), labels.to("cuda")

            with torch.no_grad():
                embeddings, logits = eval_model.get_outputs(images, quantize=quantize)
                acts.append(embeddings)

        acts = torch.cat(acts, dim=0)
        acts = acts.detach().cpu().numpy()[:total_instance].astype(np.float64)

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    from torchvision.transforms import Compose, ToTensor, Normalize
    from dataset import Dataset_

    eval_dataset = CIFAR10('./data/', download=True, train=True, transform=Compose([ToTensor(),
                                                                               Normalize([0.5, 0.5, 0.5],
                                                                                         [0.5, 0.5, 0.5])]))

    eval_dataset = Dataset_(data_name="CIFAR10",
                            data_dir="./data/",
                            train=True,
                            crop_long_edge=False,
                            resize_size=None,
                            resizer="wo_resize",
                            random_flip=False,
                            hdf5_path=None,
                            normalize=True,
                            load_data_in_memory=False)


    loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
    iters = iter(loader)
    img, label = iters.next()

    img.mean()
    img.std()
    img.min()
    img.max()


    eval_model = LoadEvalModel("InceptionV3_tf", torch.device('cuda'))
    # embeddings, logits = eval_model.get_outputs(img)

    mu, sigma = calculate_moments(data_loader=loader,
                                  eval_model=eval_model,
                                  num_generate="N/A",
                                  batch_size=64,
                                  quantize=True,
                                  disable_tqdm=False,
                                  fake_feats=None)

    mu1, sigma1 = calculate_moments(data_loader=loader,
                                  eval_model=eval_model,
                                  num_generate="N/A",
                                  batch_size=64,
                                  quantize=True,
                                  disable_tqdm=False,
                                  fake_feats=None)

    path0 = "/home/dblab/git/PyTorch-StudioGAN/save_files/cifar10/base_loss/moments/CIFAR10_32_wo_resize_train_legacy_InceptionV3_tf_moments.npz"

    data0 = np.load(path0)
    mu_ori0, sigma_ori0 = data0['mu'], data0['sigma']


    path1 = "/home/dblab/git/PyTorch-StudioGAN/save_files/cifar10/ContraGAN_noattn/moments/CIFAR10_32_wo_resize_train_legacy_InceptionV3_tf_moments.npz"

    data1 = np.load(path1)
    mu_ori1, sigma_ori1 = data1['mu'], data1['sigma']



    path2 = "/home/dblab/git/PyTorch-StudioGAN/save_files/cifar10/contra_loss_noattn/moments/CIFAR10_32_wo_resize_train_legacy_InceptionV3_tf_moments.npz"

    data2 = np.load(path2)
    mu_ori2, sigma_ori2 = data2['mu'], data2['sigma']

    #
    #
    # # print(embeddings)
    # # print(logits)
    #
    # import torch
    # import torch.nn.functional as F
    #
    # tensor = torch.randn(3,32,32)
    # tensor[None, ...].size()

