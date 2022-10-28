from torchvision.datasets import MNIST, FashionMNIST
import matplotlib.pyplot as plt
import tsnecuda
tsnecuda.test()


mnist = MNIST(root='/shared_hdd/sin/dataset/', download=True, train=True)
fmnist = FashionMNIST(root='/shared_hdd/sin/dataset/', download=True, train=True)


plt.imshow(mnist.data[0])
plt.show()

plt.imshow(fmnist.data[0])
plt.show()


import numpy as np
from tsnecuda import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# X_embedded = TSNE().fit_transform(X)
X_embedded = TSNE().fit_transform(mnist.data.view(-1, 28*28))
X_embedded.shape

plt.scatter(X_embedded[:,0], X_embedded[:, 1],
            alpha=.5, s=20,
            c=mnist.targets,
            cmap=plt.cm.get_cmap('rainbow', 10) )
plt.colorbar(ticks=range(10), format='color: %d', label='color')
plt.show()



X_embedded = TSNE().fit_transform(fmnist.data.view(-1, 28*28))
plt.scatter(X_embedded[:,0], X_embedded[:, 1],
            alpha=.5, s=20,
            c=fmnist.targets,
            cmap=plt.cm.get_cmap('rainbow', 10) )
plt.colorbar(ticks=range(10), format='color: %d', label='color')
plt.show()
