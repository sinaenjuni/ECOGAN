import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_input):
        self.outputs.append(module_input)

    def clear(self):
        self.outputs = []


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

def build_resizer(output_size):
    def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            return x
    return func

model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}
model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}
class EvalModel:
    def __init__(self, device):
        self.eval_backbone = 'InceptionV3_torch'
        self.device = device
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.res = 299
        self.save_output = SaveOutput()

        self.model = torch.hub.load(model_versions[self.eval_backbone],
                                    model_names[self.eval_backbone],
                                    pretrained=True)

        self.model = self.model.to(self.device)
        hook_handles = []
        for name, layer in self.model.named_children():
            if name == "fc":
                handle = layer.register_forward_pre_hook(self.save_output)
                hook_handles.append(handle)

        self.resizer = build_resizer(output_size=(self.res, self.res))
        self.totensor = ToTensor()
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

        self.eval()

    def eval(self):
        self.model.eval()

    def get_outputs(self, x, quantize=False):
        if quantize:
            x = quantize_images(x)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)

        if self.eval_backbone in ["InceptionV3_tf", "DINO_torch", "Swin-T_torch"]:
            repres, logits = self.model(x)
        elif self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
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


