import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

fid = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/balanced_data_fid.csv')
is_score = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/balanced_data_is.csv')

print(fid.keys())
print(is_score.keys())

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'


# fig, (ax1, ax2) = plt.subplots(figsize=(10, 6))
fig, ax = plt.subplots(figsize=(10, 6))


step = fid['Step']
FID_ReACGAN = fid['CIFAR10-ReACGAN-train-2022_09_20_14_46_48 - FID score']
FID_ContraGAN = fid['CIFAR10-contraloss - FID score']
FID_ECOGAN = fid['CIFAR10-ourloss - FID score']
IS_ReACGAN = is_score['CIFAR10-ReACGAN-train-2022_09_20_14_46_48 - IS score']
IS_ContraGAN = is_score['CIFAR10-contraloss - IS score']
IS_ECOGAN = is_score['CIFAR10-ourloss - IS score']

# name = [f"{i}" for i in range(10)]

# ax1.bar(name, data1, color='gray')
# ax1.set_ylabel('FID')
# ax2.bar(name, data2, color='gray')
# ax2.set_ylabel('IS')
ax.set_xlabel('Step')
# ax.set_yscale("log")

ax.plot(step, FID_ContraGAN, label='2C loss', linestyle='--', color='r')
ax.plot(step, FID_ReACGAN, label='D2D-CE loss', linestyle='--', color='g')
ax.plot(step, FID_ECOGAN, label='EC loss (our)', linestyle='--', color='b')

ax.plot(step, IS_ContraGAN, label='2C loss', color='r')
ax.plot(step, IS_ECOGAN, label='D2D-CE loss', color='g')
ax.plot(step, IS_ReACGAN, label='EC loss (our)', color='b')

ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
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

