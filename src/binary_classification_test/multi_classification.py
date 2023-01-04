    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('classic')
    np.set_printoptions(precision=3, suppress=True)

    X = np.array([[-0.1, 1.4],
                  [-0.5, 0.2],
              [ 1.3, 0.9],
              [-0.6, 0.4],
              [-1.6, 0.2],
              [ 0.2, 0.2],
              [-0.3,-0.4],
              [ 0.7,-0.8],
              [ 1.1,-1.5],
              [-1.0, 0.9],
              [-0.5, 1.5],
              [-1.3,-0.4],
              [-1.4,-1.2],
              [-0.9,-0.7],
              [ 0.4,-1.3],
              [-0.4, 0.6],
              [ 0.3,-0.5],
              [-1.6,-0.7],
              [-0.5,-1.4],
              [-1.0,-1.4]])

    y = np.array([0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 2, 2, 2, 1, 0, 1, 2, 2, 2])
    Y = np.eye(3)[y]
    colormap = np.array(['r', 'g', 'b'])

    def plot_scatter(X, y, colormap, path):
       plt.grid()
       plt.xlim([-2.0, 2.0])
       plt.ylim([-2.0, 2.0])
       plt.xlabel('$x_1$', size=20)
       plt.ylabel('$x_2$', size=20)
       plt.title('Input 2D points', size=18)
       plt.scatter(X[:,0], X[:,1], s=50, c=colormap[y])
       # plt.savefig(path)

    plot_scatter(X, y, colormap, 'image.png')
    plt.show()
    # plt.close()
    # plt.clf()
    # plt.cla()



def stable_softmax(z):
  # z is 3 x 1
  a = np.exp(z - max(z)) / np.sum(np.exp(z - max(z)))
  # a is 3 x 1
  return a

def forward_propagate(x, W, b):
  # W is 3 x 2
  # x is 2 x 1
  # b is 3 x 1
  z = np.matmul(W, x) + b
  a = stable_softmax(z)
  # z is 3 x 1
  # a is 3 x 1
  return z, a

W = np.array([[ 0.31, 3.95],
              [ 7.07, -0.23],
              [-6.27, -2.35]])

b = np.array([[ 1.2  ],
              [ 2.93 ],
              [-4.14 ]])

z, a = forward_propagate(X[0,:].reshape(2,1), W, b)

print(z)
print(a)
print(y[0])



LEARNING_RATE = 2.0
# NUM_EPOCHS = 40
NUM_EPOCHS = 1

def get_loss(y, a):
  return -1 * np.sum(y * np.log(a))

def get_loss_numerically_stable(y, z):
   return -1 * np.sum(y * (z + (-z.max() - np.log(np.sum(np.exp(z-z.max()))))))

def get_gradients(x, z, a, y):
    da = (-y / a)

    matrix = np.matmul(a, np.ones((1, 3))) * (np.identity(3) - np.matmul(np.ones((3, 1)), a.T))
    dz = np.matmul(matrix, da)

    dW = dz * x.T
    db = dz.copy()

    return dz, dW, db

def gradient_descent(W, b, dW, db, learning_rate):
  W = W - learning_rate * dW
  b = b - learning_rate * db
  return W, b

# random initialization
W_initial = np.random.rand(3, 2)
W = W_initial.copy()
b = np.zeros((3, 1))

W_cache = []
b_cache = []
L_cache = []

for i in range(NUM_EPOCHS):
  dW = np.zeros(W.shape)
  db = np.zeros(b.shape)
  L = 0
  for j in range(X.shape[0]):
    x_j = X[j,:].reshape(2,1)
    y_j = Y[j,:].reshape(3,1)

    z_j, a_j = forward_propagate(x_j, W, b)
    loss_j = get_loss_numerically_stable(y_j, z_j)
    dZ_j, dW_j, db_j = get_gradients(x_j, z_j, a_j, y_j)

    dW += dW_j
    db += db_j
    L += loss_j

  dW *= (1.0/20)
  db *= (1.0/20)
  L *= (1.0/20)

  W, b = gradient_descent(W, b, dW, db, LEARNING_RATE)

  W_cache.append(W)
  b_cache.append(b)
  L_cache.append(L)

plt.grid()
plt.title('Loss', size=18)
plt.xlabel('Number of iterations', size=15)
plt.ylabel('Loss', size=15)
plt.ylim([0, max(L_cache) * 1.1])
plt.plot(L_cache)
plt.show()

# plt.savefig('image.png')
# plt.close()
# plt.clf()
# plt.cla()
