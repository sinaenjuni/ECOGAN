


import torch






import numpy as np
mat = np.zeros((10, 10))
x = np.array([1,1,2,1,1])
y = np.array([1,2,2,2,1])

np.equal(x.reshape(-1,1), y.reshape(1, -1))


for x, y in zip(x, y):
    mat[x, y] += 1

acc = mat.trace() / mat.sum()
acc_per_cls = mat.diagonal() / mat.sum(1)

(2, 2).shape