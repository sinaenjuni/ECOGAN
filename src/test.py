from torchvision.datasets import FashionMNIST, MNIST


dataset = FashionMNIST('/shared_hdd/sin/dataset/', download=False)


classes, counts = torch.unique(dataset.targets, return_counts=True)


dataset.targets[dataset.targets == 1]