import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

fid = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/ReACGAN_fid_per_div.csv')
is_score = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/ReACGAN_is_per_div.csv')

print(fid.keys())
print(is_score.keys())

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'


# fig, (ax1, ax2) = plt.subplots(figsize=(10, 6))


step = fid['Step']
data_fid1 = fid['imbcifar10_CIFAR10_div5-ReACGAN_div5-train-2022_08_31_11_00_25 - FID score']
data_fid2 = fid['imbcifar10_CIFAR10_div4-ReACGAN_div4-train-2022_08_31_11_00_21 - FID score']
data_fid3 = fid['imbcifar10_CIFAR10_div3-ReACGAN_div3-train-2022_08_31_11_00_15 - FID score']
data_fid4 = fid['imbcifar10_CIFAR10_div2-ReACGAN_div2-train-2022_08_31_10_59_56 - FID score']
data_fid5 = fid['IMBCIFAR10_CIFAR10-ReACGAN-train-2022_08_26_04_47_01 - FID score']

data_is1 = is_score['imbcifar10_CIFAR10_div5-ReACGAN_div5-train-2022_08_31_11_00_25 - IS score']
data_is2 = is_score['imbcifar10_CIFAR10_div4-ReACGAN_div4-train-2022_08_31_11_00_21 - IS score']
data_is3 = is_score['imbcifar10_CIFAR10_div3-ReACGAN_div3-train-2022_08_31_11_00_15 - IS score']
data_is4 = is_score['imbcifar10_CIFAR10_div2-ReACGAN_div2-train-2022_08_31_10_59_56 - IS score']
data_is5 = is_score['IMBCIFAR10_CIFAR10-ReACGAN-train-2022_08_26_04_47_01 - IS score']


# name = [f"{i}" for i in range(10)]

# ax1.bar(name, data_fid1, color='gray')
# ax1.set_ylabel('FID')
# ax2.bar(name, data_fid2, color='gray')
# ax2.set_ylabel('IS')
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel('Step')
ax.set_yscale("log")
ax.plot(step, data_fid1, label='FID of Model 5', linestyle='--', color='y')
ax.plot(step, data_fid2, label='FID of Model 4', linestyle='--', color='b')
ax.plot(step, data_fid3, label='FID of Model 3', linestyle='--', color='g')
ax.plot(step, data_fid4, label='FID of Model 2', linestyle='--', color='r')
ax.plot(step, data_fid5, label='FID of Model 1', linestyle='--', color='k')


ax.plot(step, data_is1, label='IS of Model 5', linestyle='-', color='y')
ax.plot(step, data_is2, label='IS of Model 4', linestyle='-', color='b')
ax.plot(step, data_is3, label='IS of Model 3', linestyle='-', color='g')
ax.plot(step, data_is4, label='IS of Model 2', linestyle='-', color='r')
ax.plot(step, data_is5, label='IS of Model 1', linestyle='-', color='k')

ax.legend(bbox_to_anchor=(1, 0.7), loc=1, frameon=False, fontsize=16, ncol=2)
# ax2.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
# ax1.legend()


# ax1.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")
# ax2.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")

ax.xaxis.set_major_formatter(lambda x, pos: 0 if x == 0 else str(int(x/1000)) + "k")
# ax2.xaxis.set_major_formatter(lambda x, pos: 0 if x == 0 else str(int(x/1000)) + "k")

# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
ax.grid(visible=True, which='minor', alpha=0.8, linewidth=0.8, linestyle='--')
ax.grid(visible=True, which='major', alpha=0.8, linewidth=0.8, linestyle='-')
#
# ax2.grid(visible=True, which='minor', linestyle='--')
# ax2.grid(visible=True, which='major', linestyle='-')
#
ax.minorticks_on()
# ax2.minorticks_on()
ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
#
# ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='out')
# ax2.xaxis.set_tick_params(which='minor', size=7, width=0, direction='in')
# ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
# ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

plt.tight_layout()
plt.show()

