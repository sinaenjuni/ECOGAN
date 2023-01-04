import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator

fid = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/cifar10_fid.csv')
is_score = pd.read_csv('/home/dblab/git/ECOGNA/src/figure/data/cifar10_ins.csv')

print(fid.keys())
print(is_score.keys())

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'


# fig, (ax1, ax2) = plt.subplots(figsize=(10, 6))
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))


step = fid['Step']
FID_1 = fid['ECOGAN-sm - fid']
FID_2 = fid['ECOGAN - fid']
FID_3 = fid['EBGAN-pre - fid']
FID_4 = fid['EBGAN - fid']

IS_1 = is_score['ECOGAN-sm - ins']
IS_2 = is_score['ECOGAN - ins']
IS_3 = is_score['EBGAN-pre - ins']
IS_4 = is_score['EBGAN - ins']

# name = [f"{i}" for i in range(10)]

# ax1.bar(name, data1, color='gray')
# ax1.set_ylabel('FID')
# ax2.bar(name, data2, color='gray')
# ax2.set_ylabel('IS')
# ax.set_yscale("log")

ax2.set_xlabel('Step')
ax1.set_ylabel('FID')
ax2.set_ylabel('IS')

ax1.plot(step, FID_1, label='ECOGAN-sm', linestyle='--', color='r')
ax1.plot(step, FID_2, label='ECOGAN', linestyle='--', color='g')
ax1.plot(step, FID_3, label='EBGAN-pre', linestyle='--', color='b')
ax1.plot(step, FID_4, label='EBGAN', linestyle='--', color='b')

ax2.plot(step, IS_1, label='ECOGAN-sm', color='r')
ax2.plot(step, IS_2, label='ECOGAN', color='g')
ax2.plot(step, IS_3, label='EBGAN-pre', color='b')
ax2.plot(step, IS_4, label='EBGAN', color='b')

ax1.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
ax2.legend(bbox_to_anchor=(1, 0.6), loc=1, frameon=False, fontsize=16)
# ax2.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
# ax1.legend()


formatter_fn = lambda x, pos: 0 if x == 0 else str(int(x/1000)) + "k"
ax1.xaxis.set_major_formatter(formatter_fn)
ax2.xaxis.set_major_formatter(formatter_fn)

ax1.grid(visible=True, which='minor', alpha=0.8, linewidth=0.8, linestyle='--')
ax1.grid(visible=True, which='major', alpha=0.8, linewidth=0.8, linestyle='-')
ax2.grid(visible=True, which='minor', alpha=0.8, linewidth=0.8, linestyle='--')
ax2.grid(visible=True, which='major', alpha=0.8, linewidth=0.8, linestyle='-')

ax1.minorticks_on()
ax2.minorticks_on()


ax1.minorticks_on()
ax2.minorticks_on()
ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax1.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax1.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax2.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

plt.tight_layout()
plt.show()

