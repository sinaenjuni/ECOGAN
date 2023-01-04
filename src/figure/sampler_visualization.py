from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from utils.datasets import DataModule_
from torch.utils.data import DataLoader

dataset = DataModule_(data_name='Places_LT', img_size=64, img_dim=3, is_sampling=True, is_extension=False, batch_size=128)
dataset.setup('fit')

loader = dataset.train_dataloader()

temp = []
append = temp.append
for img, target in tqdm(loader):
    append(target)

classes, num_classes = np.unique(np.concatenate(temp), return_counts=True)



#
#
# base_path = Path('/shared_hdd/sin/gen/ori.h5py')
# h5_ori = h5py.File(base_path, 'r')
#
#
# data = h5_ori['Places_LT']['train']['data']
# targets = h5_ori['Places_LT']['train']['targets']
#
# classes, num_classes = np.unique(targets, return_counts=True)


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'


plt.bar(classes, num_classes, color='gray')

ax = plt.gca()
# ax.set_xticks(classes)
# ax.set_xticklabels(classes)
# ax.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.minorticks_on()
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out')
ax.xaxis.set_tick_params(which='minor', size=7, width=0, direction='in')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

# plt.tight_layout()
plt.show()
# plt.axis('off')
# plt.savefig('/home/dblab/git/ECOGNA/src/figure/image.png', bbox_inches='tight', pad_inches = 0)


grids = []
for i in range(10):
    print(i)
    idx = np.where(np.array(targets) == i)[0][:10]
    img = data[idx]
    grid = np.concatenate(img, axis=0)
    grids.append(grid)

grids = np.concatenate(np.array(grids), axis=1)
plt.imshow(grids)

# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
#             hspace = 0, wspace = 0)
# plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.axis(False)
# plt.tight_layout()
# plt.show()

plt.axis('off')
plt.savefig('/home/dblab/git/ECOGNA/src/figure/image.png', bbox_inches='tight', pad_inches = 0)