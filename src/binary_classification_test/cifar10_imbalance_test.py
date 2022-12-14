import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.models.resnet import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional import confusion_matrix
import seaborn as sns
from tqdm import tqdm
# import wandb
# wandb.init(project="BCE analysis", entity="sinaenjuni", name="imb-data_Hinge")



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

dataset = CIFAR10_LT(is_train=True, cls_target=(0, 1), transforms=Compose([ToTensor(),
                                                                           Normalize((0.5, 0.5, 0.5),
                                                                                     (0.5, 0.5, 0.5))]))
print(np.unique(dataset.labels, return_counts=True))
loader = DataLoader(dataset, batch_size=128, shuffle=True)
# img, label = iter(loader).__next__()

test_dataset = CIFAR10_LT(is_train=False, cls_target=(0, 1), transforms=Compose([ToTensor(),
                                                                                 Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))]))
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


encoder = resnet18(num_classes=2).cuda()
# classifier = nn.Sequential(nn.ReLU(),
#                            nn.Linear(2, 1, bias=False)).cuda()
classifier = nn.Linear(2, 2, bias=False).cuda()
# w = torch.tensor([[0.0, 1.0]], requires_grad=True).cuda()
# w = F.normalize(w, dim=1)
# torch.ran


def hinge(y_hat, label):
    return torch.mean( (  label) * F.relu(1. - y_hat)) +\
           torch.mean( (1-label) * F.relu(1. + y_hat))


def get_loss_relu(z, y):
    L = (y*F.relu(1-z)).sum(1) + ((1-y)*F.relu(1+z)).sum(1)
    return L.mean()


def get_loss_numerically_stable(y, z):
    L = -1 * (y * -1 * torch.log(1 + torch.exp(-z)) +
              (1-y) * (-z - torch.log(1 + torch.exp(-z))))
    return L.mean()



# em = nn.Embedding(2, 2).cuda()
# w = torch.nn.utils.weight_norm(em, dim=1)
# em.weight.norm(dim=1)
# for W in em.parameters():
#     W = F.normalize(W, dim=1)

# em_weight = em.weight
# t = F.normalize(e(torch.range(0, 1).long()), dim=1)
# plt.scatter(w[:,0].cpu().detach().numpy(), w[:,1].cpu().detach().numpy())
# plt.scatter(em_weight[:,0], em_weight[:,1])
# plt.xlim([-2.0, 2.0])
# plt.ylim([-2.0, 2.0])
# plt.show()

optimizer = torch.optim.Adam(params=[{'params': encoder.parameters()},
                                    {'params': classifier.parameters()}], lr=0.01)
color = np.array(['r','b'])
color_test = np.array(['k','y'])

# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/c41d599622b6a6ba7e5be6faf3a01da1202024ea/loss_functions.py#L6

# def remove_diagonal(M):
#     batch_size = M.size(0)
#     mask = torch.ones_like(M).fill_diagonal_(0).bool()
#     return M[mask].view(batch_size, -1).contiguous()
#
# s = 1
# m = 4.0


def vis(e_list, l_list):
    plt.scatter(e_list[:, 0],
                e_list[:, 1], c=color[l_list], alpha=0.3)

    # xs = np.array([e_list[:,0].min(), e_list[:,0].max()])
    # xs = np.array([-1.0, 1.0])
    # for W in classifier.parameters():
    #     W = F.normalize(W, dim=1)

    # w1 = classifier[1].weight.cpu().detach().numpy()[0][0]
    # w2 = classifier[1].weight.cpu().detach().numpy()[0][1]
    # b = classifier[1].bias.cpu().detach().numpy()[0]

    # plt.scatter(w1,
    #             w2, c='black', s=100, alpha=.3)

    # ys = (-w1 * xs - b) / w2
    # ys = (-w1 * xs) / w2
    # plt.plot(xs, ys, c='black')
    plt.xlim([-2.0, 2.0])
    plt.ylim([-2.0, 2.0])
    plt.show()


for i in range(50):
    total_loss = 0
    e_list = []
    l_list = []
    y_hat_list = []
    for img, label in loader:
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()

        e = encoder(img)
        e = F.normalize(e, dim=1)

        for W in classifier.parameters():
            W = F.normalize(W, dim=1)
        y_hat = classifier(e)
        y_hat = y_hat.squeeze()

        # loss = hinge(y_hat, label).mean()
        # loss = F.binary_cross_entropy_with_logits(y_hat, label.float())

        # numerator = s * (torch.diagonal(y_hat.transpose(0, 1)[label]) - m)
        numerator = torch.diagonal(y_hat.transpose(0, 1)[label] - 1)
        # excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        excl = torch.diagonal(y_hat.transpose(0, 1)[1-label])

        denominator = torch.exp(numerator) + torch.sum(torch.exp(excl))
        L = numerator - torch.log(denominator)
        loss = -torch.mean(L)


        # numerator = s * (torch.diagonal(y_hat.transpose(0, 1)[label]) - m)
        # excl = torch.cat([torch.cat((y_hat[i, :y], y_hat[i, y + 1:])).unsqueeze(0) for i, y in enumerate(label)], dim=0)
        # denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
        # L = numerator - torch.log(denominator)
        # loss = -torch.mean(L)

        # improved_sim_matrix = (1 - label) * torch.exp(y_hat)
        # pos_attr = label * F.relu((0.9 - y_hat))
        # compute negative repulsion term
        # neg_repul = torch.log(torch.exp(-pos_attr) + improved_sim_matrix.sum())
        # compute data to data cross-entropy criterion
        # loss = (pos_attr + neg_repul).mean()



        # loss = (label * (F.relu(10. - y_hat))).mean() +\
        #        ((1 - label) * (F.relu(10. + y_hat))).mean()

        # loss = (-1 * (label * -1 * torch.log(1 + torch.exp(-y_hat)) +
        #       (1 - label) * (-y_hat - torch.log(1 + torch.exp(-y_hat))))).mean()

        # loss =  -1 * (label * torch.log(1 + torch.exp(-y_hat)) +
        #               (1 - label) * torch.log(1 + torch.exp(-y_hat))).mean()
        # -1 * (label * -1 * torch.log(1 + torch.exp(-z)) + (1 - y) * (-z - np.log(1 + np.exp(-z))))
        # denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        # L = numerator - torch.log(denominator)
        # loss = -torch.mean(L)

        # loss = F.cross_entropy(y_hat, label)
        # loss = get_loss_relu(y_hat, label)

        # loss = get_loss_relu(label, y_hat)
        # p = F.normalize(p, dim=1)

        # sim = F.cosine_similarity(p.unsqueeze(0), p.unsqueeze(1), dim=2)
        # sim = remove_diagonal(sim)
        # mask_n = torch.ne(label.unsqueeze(0), label.unsqueeze(1))
        # mask_n = remove_diagonal(mask_n)
        # mask_p = torch.eq(label.unsqueeze(0), label.unsqueeze(1))
        # mask_p = remove_diagonal(mask_p)
        #
        # sim_p = sim * mask_p
        # sim_n = sim * mask_n

        # loss = -1 * (torch.log(sim_p) + torch.log(1 - sim_n))
        # loss = -1 * (-1 * torch.log(1 + torch.exp(-sim_p).sum(1)) +
        #              (-sim_n.sum(1) - torch.log(1 + torch.exp(-sim_n).sum(1)))).mean()

        # w = em(label)
        # w = F.normalize(w, dim=1)
        # y_hat = w @ p.T
        # y_hat = p @ w.T


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
        # pos_weight = torch.ones_like(label) * 2
        # loss = get_loss_relu(y_hat.squeeze(), label.float())\
        # loss = get_loss_numerically_stable(label, y_hat)


        total_loss += loss.cpu().detach().numpy()
        loss.backward()
        optimizer.step()

        # e = F.normalize(e, dim=1)

        e_list.append(e.cpu().detach())
        l_list.append(label.cpu().detach())
        y_hat_list.append(y_hat.cpu().detach())

    # e_list = np.concatenate(e_list)
    # l_list = np.concatenate(l_list)
    # y_hat_list = np.concatenate(y_hat_list)

    e_list = torch.cat(e_list)
    l_list = torch.cat(l_list)
    y_hat_list = torch.cat(y_hat_list)



    # acc = (y_hat_list.argmax(-1) == l_list).sum()/len(l_list)
    # acc = ((y_hat_list > 1.) == l_list).sum()/len(l_list)

    cm = confusion_matrix(preds=y_hat_list, target=l_list, num_classes=2, threshold=0.5)
    acc = cm.diagonal().sum() / len(l_list)
    acc_per_cls = cm.diagonal() / cm.sum(1)
    print(cm)
    print(total_loss/len(loader),
          acc,
          acc_per_cls)

    vis(e_list, l_list)

    # try:
    #     plt.figure(figsize=(10, 10))
    #     heat_map = sns.heatmap(cm.numpy(), annot=True, fmt='d')
    #     wandb.log({'Confusion matrix': wandb.Image(heat_map),
    #                'acc': acc})
    #
    #     wandb.log({f'acc_per_cls/{i}': acc_per_cls_ for i, acc_per_cls_ in enumerate(acc_per_cls)})
    # except NameError:
    #     print("Not defined Wandb")





    # print(total_loss/len(loader), acc)

    print(y_hat_list.min(),
          y_hat_list.max(),
          y_hat_list.mean(),
          y_hat_list.std())



test_y_hat_list = []
test_e_list = []
test_l_list = []
for img, label in tqdm(test_loader, desc="Testing"):
    img, label = img.cuda(), label.cuda()
    with torch.no_grad():
        test_e = encoder(img)
        test_e = F.normalize(test_e, dim=1)

        for W in classifier.parameters():
            W = F.normalize(W, dim=1)
        test_y_hat = classifier(test_e)

    test_y_hat = test_y_hat.squeeze()
    # test_y_hat_list.append(test_y_hat.cpu().detach().numpy())
    # test_e_list.append(test_e.cpu().detach().numpy())
    # test_l_list.append(label.cpu().detach().numpy())

    test_y_hat_list.append(test_y_hat)
    test_e_list.append(test_e)
    test_l_list.append(label)


# test_y_hat_list = np.concatenate(test_y_hat_list)
# test_e_list = np.concatenate(test_e_list)
# test_l_list = np.concatenate(test_l_list)

test_y_hat_list = torch.cat(test_y_hat_list)
test_e_list = torch.cat(test_e_list)
test_l_list = torch.cat(test_l_list)

cm_test = confusion_matrix(test_y_hat_list, test_l_list, num_classes=2)
acc_test = cm_test.diagonal().sum() / len(test_l_list)
acc_per_cls_test = cm_test.diagonal() / cm_test.sum(1)
print(cm_test)
print(acc_test,
      acc_per_cls_test)
# plt.scatter(test_e_list[:, 0],
#             test_e_list[:, 1], c=color[test_l_list], alpha=0.3)
#
# xs = np.array([-2.0, 2.0])
# w1 = classifier.weight.cpu().detach().numpy()[0][0]
# w2 = classifier.weight.cpu().detach().numpy()[0][1]
# ys = (-w1 * xs) / w2
# plt.plot(xs, ys, c='black')
#
# plt.xlim([-2.0, 2.0])
# plt.ylim([-2.0, 2.0])
# plt.show()
#
# acc = (test_y_hat_list.argmax(-1) == test_l_list).sum() / len(l_list)
# # acc = ((test_y_hat_list > 0.5) == test_l_list).sum() / len(test_l_list)
# print(acc)

# idx = np.where(test_l_list == 0)
# ((test_y_hat_list[idx] > 0.5) == test_l_list[idx]).sum() / len(test_l_list[idx])




#
# def plot_decision_boundary(X, y, w, b, path):
    # plt.grid()
    # plt.xlim([-4.0, 20.0])
    # plt.ylim([-4.0, 20.0])
    # plt.xlabel('$x_1$', size=20)
    # plt.ylabel('$x_2$', size=20)
    # plt.title('Decision boundary', size=18)

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







