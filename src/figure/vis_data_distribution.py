import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = 'Arial'


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,6))

data1 = [5000] * 10
data2 = [5000] + [i*100 for i in range(1,10)]
name = [f"{i}" for i in range(10)]

ax1.bar(name, data1, color='gray')
ax1.set_xlabel('(a)')
ax2.bar(name, data2, color='gray')
ax2.set_xlabel('(b)')

ax1.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")
ax2.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.minorticks_on()
ax2.minorticks_on()
ax1.xaxis.set_tick_params(which='major', size=10, width=2, direction='out')
ax1.xaxis.set_tick_params(which='minor', size=7, width=0, direction='in')
ax1.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax1.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

ax2.xaxis.set_tick_params(which='major', size=10, width=2, direction='out')
ax2.xaxis.set_tick_params(which='minor', size=7, width=0, direction='in')
ax2.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
ax2.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')


plt.tight_layout()
plt.show()

