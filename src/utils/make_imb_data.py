import torch
from torchvision.datasets import FashionMNIST, MNIST
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def save_imgs(base_path, imgs, labels):
    for idx, (img, label) in enumerate(zip(imgs, labels)):
        img = Image.fromarray(img.numpy())
        save_path = base_path / str(label.item())
        if not save_path.exists():
            save_path.mkdir(exist_ok=True, parents=True)
        img.save(save_path / f"{idx}.jpeg")


base_path = Path("/shared_hdd/sin/save_files/imb_FashionMNIST/")
# data_train = MNIST('/shared_hdd/sin/dataset/', download=False, train=True)
# data_test = MNIST('/shared_hdd/sin/dataset/', download=False, train=False)
data_train = FashionMNIST('/shared_hdd/sin/dataset/', download=False, train=True)
data_test = FashionMNIST('/shared_hdd/sin/dataset/', download=False, train=False)
classes, counts = torch.unique(data_train.targets, return_counts=True)

imgs = data_train.data
labels = data_train.targets

for c in range(1, 10, 1):
    imgs = torch.cat([imgs[labels != c], imgs[labels == c][:100 * c]])
    labels = torch.cat([labels[labels != c], labels[labels == c][:100 * c]])

save_imgs(base_path/'train', imgs, labels)

imgs = data_test.data
labels = data_test.targets

save_imgs(base_path/'val', imgs, labels)


