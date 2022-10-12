import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CIFAR10_LT(Dataset):
    def __init__(self, is_train, cls_target=None, transforms=None):
        if is_train:
            data_path = Path("/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/train/")
        else:
            data_path = Path("/home/dblab/git/PyTorch-StudioGAN/data/imb_cifar10/val/")
        self.images = []
        self.labels = []
        for path in data_path.glob("*/*.JPEG"):
            self.images.append(str(path))
            self.labels.append(int(path.parts[-2][-1]))
        self.labels = np.hstack(self.labels)
        self.images = np.hstack(self.images)

        if cls_target is not None:
            label_target = []
            image_target = []
            for cls in cls_target:
                idx_target = np.where(self.labels == cls)
                label_target.append(self.labels[idx_target])
                image_target.append(self.images[idx_target])

            self.images = np.hstack(image_target)
            self.labels = np.hstack(label_target)

        self.transforms = transforms
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

dataset = CIFAR10_LT(is_train=True, cls_target=(0, 1), transforms=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)
                              )]))
np.unique(dataset.labels, return_counts=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
# img, label = iter(loader).__next__()

test_dataset = CIFAR10_LT(is_train=False, cls_target=(0, 1), transforms=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)
                              )]))


encoder = resnet18(num_classes=2)
classifier = nn.Sequential(nn.ReLU(),
                           nn.Linear(2, 1))

optimizer = torch.optim.SGD(params=[{'params': encoder.parameters()},
                                    {'params': classifier.parameters()}], lr=0.1)
color = np.array(['r','b'])
for i in range(100):
    total_loss = 0
    e_list = []
    l_list = []
    for img, label in loader:
        optimizer.zero_grad()

        e = encoder(img)
        y_hat = classifier(e)
        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), label.float())
        total_loss += loss
        loss.backward()
        optimizer.step()


        e_list.append(e.detach().numpy())
        l_list.append(label.detach().numpy())
    print(loss/len(loader))

    e_list = np.concatenate(e_list)
    l_list = np.concatenate(l_list)
    plt.scatter(e_list[:,0], e_list[:,1], c=color[l_list])
    plt.show()


def plot_decision_boundary(X, y, w, b, path):
    plt.grid()
    # plt.xlim([-4.0, 20.0])
    # plt.ylim([-4.0, 20.0])
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.title('Decision boundary', size=18)

    xs = np.array([-4.0, 12.0])
    ys = (-w[0] * xs - b) / w[1]

    plt.scatter(X[:, 0], X[:, 1], s=10, c=color[y], alpha=.5)
    plt.plot(xs, ys, c='black')
    # plt.savefig(path)

plot_decision_boundary(e_list, l_list,
                       classifier[1].weight.detach().numpy()[0],
                       classifier[1].bias.detach().numpy()[0],
                       'image.png')
plt.show()
model.fc.weight.size()








