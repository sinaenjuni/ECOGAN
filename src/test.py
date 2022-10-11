import torch
import numpy as np
labels = torch.tensor([0,2,3,0,5,2])
labels = labels.detach().cpu().numpy()
n_samples = labels.shape[0]
num_classes = 10
mask_multi, target = np.zeros([num_classes, n_samples]), 1.0
for c in range(num_classes):
    c_indices = np.where(labels == c)
    mask_multi[c, c_indices] = target


labels.unsqueeze(1) @ labels.unsqueeze(0)