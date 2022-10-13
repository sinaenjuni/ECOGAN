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

dataset = CIFAR10_LT(is_train=False, cls_target=(0, 1), transforms=Compose([ToTensor(),
                                                                           Normalize((0.5, 0.5, 0.5),
                                                                                     (0.5, 0.5, 0.5))]))
np.unique(dataset.labels, return_counts=True)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
# img, label = iter(loader).__next__()

test_dataset = CIFAR10_LT(is_train=False, cls_target=(0, 1), transforms=Compose([ToTensor(),
                                                                                 Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))]))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

def get_loss_relu(z, y):
    return (y*F.relu(1-z)).mean() + ((1-y)*F.relu(1+z)).mean()

encoder = resnet18(num_classes=2).cuda()
classifier = nn.Sequential(nn.ReLU(),
                           nn.Linear(2, 2, bias=False)).cuda()

w = torch.tensor([[1,0]]).float().cuda()

# em = nn.Embedding(1, 2).cuda()
# w = torch.nn.utils.weight_norm(em, dim=1)
# em.weight.norm(dim=1)
# for W in em.parameters():
#     W = F.normalize(W, dim=1)

# em_weight = em.weight
# t = F.normalize(e(torch.range(0, 1).long()), dim=1)
plt.scatter(w[:,0].cpu().detach().numpy(), w[:,1].cpu().detach().numpy())
# plt.scatter(em_weight[:,0], em_weight[:,1])
plt.xlim([-2.0, 2.0])
plt.ylim([-2.0, 2.0])
plt.show()

optimizer = torch.optim.Adam(params=[{'params': encoder.parameters()},
                                    {'params': classifier.parameters()}], lr=0.1)
color = np.array(['r','b'])
color_test = np.array(['k','y'])

for i in range(50):
    total_loss = 0
    e_list = []
    l_list = []
    for img, label in loader:
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()

        e = encoder(img)
        y_hat = classifier(e)
        y_hat = F.normalize(y_hat, dim=1)
        y_hat = y_hat @ w.T

        # e = F.normalize(e, dim=1)

        # for W in classifier.parameters():
        #     W = F.normalize(W, dim=1)
        # y_hat = classifier(e)

        # loss = get_loss_relu(y_hat, label)

        # y_hat = F.normalize(y_hat, dim=1)


        # w = em.weight
        # y_hat @ w.T

        # loss = -torch.mean(torch.exp(y_hat @ w.T))

        # for W in classifier.parameters():
        #     W = F.normalize(W, dim=1)

        loss = F.binary_cross_entropy_with_logits(y_hat.squeeze(), label.float())
        # loss = get_loss_relu(y_hat.squeeze(), label.float())
        total_loss += loss.cpu().detach().numpy()
        loss.backward()
        optimizer.step()

        e = F.normalize(e, dim=1)
        e_list.append(e.cpu().detach().numpy())
        l_list.append(label.cpu().detach().numpy())
    print(total_loss/len(loader))

    e_list = np.concatenate(e_list)
    l_list = np.concatenate(l_list)
    plt.scatter(e_list[:,0], e_list[:,1], c=color[l_list])

    # xs = np.array([e_list[:,0].min(), e_list[:,0].max()])
    # w1 = classifier[1].weight.cpu().detach().numpy()[0][0]
    # w2 = classifier[1].weight.cpu().detach().numpy()[0][1]
    # b = classifier[1].bias.cpu().detach().numpy()[0]
    # ys = (-w1 * xs - b) / w2
    # ys = (-w1 * xs) / w2
    # plt.plot(xs, ys, c='black')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])
    plt.show()

#
# test_e_list = []
# test_l_list = []
# for img, label in test_loader:
#     img, label = img.cuda(), label.cuda()
#     e = encoder(img)
#     e = F.normalize(e, dim=1)
#     test_e_list.append(e.cpu().detach().numpy())
#     test_l_list.append(label.cpu().detach().numpy())
#
# test_e_list = np.concatenate(test_e_list)
# test_l_list = np.concatenate(test_l_list)
# #
# def plot_decision_boundary(X, y, w, b, path):
#     plt.grid()
#     # plt.xlim([-4.0, 20.0])
#     # plt.ylim([-4.0, 20.0])
#     plt.xlabel('$x_1$', size=20)
#     plt.ylabel('$x_2$', size=20)
#     plt.title('Decision boundary', size=18)
#
#     xs = np.array([-1.0, 1.0])
#     ys = (-w[0] * xs - b) / w[1]
#
#     plt.scatter(X[:, 0], X[:, 1], s=80, c=color[y], alpha=1)
#     plt.plot(xs, ys, c='black')
#     # plt.savefig(path)
#
# plot_decision_boundary(e_list, l_list,
#                        classifier[1].weight.cpu().detach().numpy()[0],
#                        classifier[1].bias.cpu().detach().numpy()[0],
#                        'image.png')
# plt.scatter(test_e_list[:, 0], test_e_list[:, 1], s=10, c=color_test[test_l_list], alpha=.7, marker='x')
#
# plt.show()
# # model.fc.weight.size()
#







