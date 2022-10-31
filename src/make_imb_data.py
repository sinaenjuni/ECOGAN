import torch
from torchvision.datasets import FashionMNIST, MNIST
from PIL import Image
from pathlib import Path


dataset = MNIST('/shared_hdd/sin/dataset/', download=False)
classes, counts = torch.unique(dataset.targets, return_counts=True)

imgs = dataset.data
labels = dataset.targets


for c in range(1, 10, 1):
    imgs = torch.cat([imgs[labels != c], imgs[labels == c][:100 * c]])
    labels = torch.cat([labels[labels != c], labels[labels == c][:100 * c]])


base_path = Path("/shared_hdd/sin/save_files/MNIST/")
for idx, (img, label) in enumerate(zip(imgs, labels)):
    img = Image.fromarray(img.numpy())
    save_path = base_path / str(label.item())
    if not save_path.exists():
        save_path.mkdir(exist_ok=True, parents=True)
    img.save(save_path / f"{idx}.jpeg")





