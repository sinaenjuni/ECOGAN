import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Sampler, WeightedRandomSampler

data_tensor = np.random.randn(10000, 128)
label_tensor = np.array([i % 10 for i in range(10000)])
print(data_tensor.shape, label_tensor.shape)

for c in range(1, 10):
    data_tensor = np.vstack([data_tensor[label_tensor != c], data_tensor[label_tensor == c][:100 * c]])
    label_tensor = np.append(label_tensor[label_tensor != c], np.ones(100 * c).astype(np.int_) * c)

print(data_tensor.shape, label_tensor.shape)


class MySampler(Sampler):
    def __init__(self, labels):
        self.unique, self.counts = np.unique(labels, return_counts=True)

    #     def __len__(self):
    #         return 2
    def __iter__(self):
        x = torch.randperm(self.dataset, 2)
        yield from x


#         return [1,2,3]


data_tensor = torch.from_numpy(data_tensor)
label_tensor = torch.from_numpy(label_tensor)

unique, counts = np.unique(label_tensor, return_counts=True)
n_sample = len(label_tensor)
weight = [n_sample / count for count in counts]
weights = [weight[label] for label in label_tensor]

dataset = TensorDataset(data_tensor, label_tensor)
loader = DataLoader(dataset, batch_size=100, shuffle=False, sampler=WeightedRandomSampler(weights, 5500))

count = []
for epoch in range(10):
    for batch in loader:
        data, label = batch
        print(label)
        count.append(label.numpy())

# print(np.unique(count, return_counts=True))


np.unique(np.concatenate(count), return_counts=True)