from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class MergeDataset(Dataset):
    def __init__(self, path1, path2, transform=None):
        self.path1 = Path(path1)
        self.path2 = Path(path2)

        self.images = []
        self.targets = []

        self.get_data_paths(path1)
        self.get_data_paths(path2)

        self.transform = transform

    def get_data_paths(self, path):
        for path_ in Path(path).glob("*/*.JPEG"):
            self.images.append(path_)
            self.targets.append(int(path_.parts[-2][-1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label






