import numpy as np
import matplotlib.pyplot as plt

plt.style.use('classic')
np.set_printoptions(precision=3, suppress=True)

X = np.array([
    # [-0.1, 1.4],
    #           [-0.5, -0.1],
              [1.3, 0.9],
              # [-0.6, 0.4],
              # [-1.5, 0.4],
              [0.2, 0.2],
              [-0.3, -0.4],
              [0.7, -0.8],
              [1.1, -1.5],
              [-1.0, 0.9],
              [-0.5, -1.5],
              # [-1.3, -0.4],
              # [-1.4, -1.2],
              [-0.9, -1.1],
              [0.4, -1.3],
              [-0.4, 0.6],
              [0.3, -0.5]])

y = np.array([
    # 0,
    #           0,
              1,
              # 0,
              # 0,
              1,
              1,
              1,
              1,
              0,
              1,
              # 0,
              # 0,
              1,
              1,
              0,
              1])
colormap = np.array(['r', 'b'])


def plot_scatter(X, y, colormap, path):
    plt.grid()
    # plt.xlim([-2.0, 2.0])
    # plt.ylim([-2.0, 2.0])
    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.title('Input 2D points', size=18)
    plt.scatter(X[:, 0], X[:, 1], s=50, c=colormap[y])
#     plt.savefig(path)

plot_scatter(X, y, colormap, 'image.png')
plt.show()

# X = X - np.min(X) / np.max(X) - np.min(X)
X = X - np.mean(X) / np.std(X)
plot_scatter(X, y, colormap, 'image.png')
plt.show()

sigmoid = lambda x: 1/(1+np.exp(-x))
xs = np.arange(-10, 10, 0.001)
plt.plot(xs, sigmoid(xs), label=r'$g(z)= \frac{1}{1+e^{-z}}$')
plt.grid()
plt.show()


relu = lambda x: np.maximum(0, x)
xs = np.arange(-10, 10, 0.001)
plt.plot(xs, relu(1-xs), label=r'$g(z)= \frac{1}{1+e^{-z}}$')
plt.plot(xs, relu(1+xs), label=r'$g(z)= \frac{1}{1+e^{-z}}$')

plt.grid()
plt.show()









LEARNING_RATE = 1
NUM_EPOCHS = 100

def get_loss(y, a):
    return -1 * (y * np.log(a) +
               (1-y) * np.log(1-a))

def get_loss_numerically_stable(y, z):
    return -1 * (y * -1 * np.log(1 + np.exp(-z)) +
                (1-y) * (-z - np.log(1 + np.exp(-z))))

def get_loss_relu(y, z):
    return y*relu(1-z) + (1-y)*relu(1+z)

def get_gradient(y, z):
    if -1 < z < 1:
        return - y + (1-y)
    else:
        return 0

w_cache = []
b_cache = []
l_cache = []

# some nice initial value, so that the plot looks nice.
w = np.array([.0, .0])
b = 0.

for i in range(NUM_EPOCHS):
    dw = np.zeros(w.shape)
    db = 0.0
    loss = 0.0

    for j in range(X.shape[0]):
        x_j = X[j,:]
        y_j = y[j]

        z_j = w.dot(x_j) + b
        # a_j = sigmoid(z_j)

        # loss_j = get_loss_numerically_stable(y_j, z_j)
        loss_j = get_loss_relu(y_j, z_j)


        # dw_j = x_j * (a_j-y_j)
        dw_j = x_j * get_gradient(y_j, z_j)
        # db_j = a_j - y_j
        db_j = get_gradient(y_j, z_j)

        dw += dw_j
        db += db_j
        loss += loss_j

    # because we have 17 samples
    dw = (1.0/17) * dw
    db = (1.0/17) * db
    loss = (1.0/17) * loss

    w -= LEARNING_RATE * dw
    b -= LEARNING_RATE * db

    w_cache.append(w.copy())
    b_cache.append(b)
    l_cache.append(loss)



plt.grid()
plt.title('Loss', size=18)
plt.xlabel('Number of iterations', size=15)
plt.ylabel('Loss', size=15)
plt.plot(l_cache)
plt.show()


xs = np.array([-2.0, 2.0])
ys = (-w[0] * xs - b) / w[1]
# ys = (-w[0] * xs ) / w[1]

plt.scatter(X[:, 0], X[:, 1], s=50, c=colormap[y])
plt.plot(xs, ys, c='black')
plt.grid()
plt.xlim([-2.0, 2.0])
plt.ylim([-2.0, 2.0])
plt.show()

plt.plot(b_cache)
plt.show()