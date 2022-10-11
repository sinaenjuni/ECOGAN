import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
np.set_printoptions(precision=3, suppress=True)

X = np.array([[-0.1, 1.4],
              [-0.5,-0.1],
              [ 1.3, 0.9],
              [-0.6, 0.4],
              [-1.5, 0.4],
              [ 0.2, 0.2],
              [-0.3,-0.4],
              [ 0.7,-0.8],
              [ 1.1,-1.5],
              [-1.0, 0.9],
              [-0.5,-1.5],
              [-1.3,-0.4],
              [-1.4,-1.2],
              [-0.9,-1.1],
              [ 0.4,-1.3],
              [-0.4, 0.6],
              [ 0.3,-0.5]])

y = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
colormap = np.array(['r', 'b'])

def plot_scatter(X, y, colormap, path):
  plt.grid()
  plt.xlim([-2.0, 2.0])
  plt.ylim([-2.0, 2.0])
  plt.xlabel('$x_1$', size=20)
  plt.ylabel('$x_2$', size=20)
  plt.title('Input 2D points', size=18)
  plt.scatter(X[:,0], X[:, 1], s=50, c=colormap[y])
  plt.savefig(path)

plot_scatter(X, y, colormap, 'image.png')
plt.show()
# plt.close()
# plt.clf()
# plt.cla()


sigmoid = lambda x: 1/(1+np.exp(-x))

xs = np.arange(-10, 10, 0.001)
plt.plot(xs, sigmoid(xs), label=r'$g(z)= \frac{1}{1+e^{-z}}$')
plt.grid()
plt.show()



import matplotlib.animation as animation
f = lambda x: x**2 + 6*x + 5
x = 6.0
LEARNING_RATE = 0.1

steps = []
for i in range(21):
  steps.append([x, f(x)])
  x = x - LEARNING_RATE * (2 * x + 6)

steps = np.array(steps)

def animate(i):
  scatter_points.set_offsets(steps[0:i+1,:])
  text_box.set_text('Iteration: {}'.format(i))
  return scatter_points, text_box


fig = plt.figure()
xs = np.arange(-10, 10, 0.001)
ax = fig.add_subplot(111)
ax.grid()
ax.plot(xs, f(xs), label=r'$f(x)= x^2 + 6x + 5$')

scatter_points = ax.scatter([], [], c='red', s=50)
anim = animation.FuncAnimation(fig, animate, 20, blit=True, interval=500)



LEARNING_RATE = 8.0
NUM_EPOCHS = 20

def get_loss(y, a):
  return -1 * (y * np.log(a) +
               (1-y) * np.log(1-a))

def get_loss_numerically_stable(y, z):
   return -1 * (y * -1 * np.log(1 + np.exp(-z)) +
                (1-y) * (-z - np.log(1 + np.exp(-z))))

def get_loss_relu(y, z):


w_cache = []
b_cache = []
l_cache = []

# some nice initial value, so that the plot looks nice.
w = np.array([-4.0, 29.0])
b = 0.0

for i in range(NUM_EPOCHS):
  dw = np.zeros(w.shape)
  db = 0.0
  loss = 0.0

  for j in range(X.shape[0]):
    x_j = X[j,:]
    y_j = y[j]

    z_j = w.dot(x_j) + b
    a_j = sigmoid(z_j)

    loss_j = get_loss_numerically_stable(y_j, z_j)

    dw_j = x_j * (a_j-y_j)
    db_j = a_j - y_j

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

plt.scatter(X[:, 0], X[:, 1], s=50, c=colormap[y])
plt.plot(xs, ys, c='black')
plt.grid()
plt.xlim([-2.0, 2.0])
plt.ylim([-2.0, 2.0])
plt.show()