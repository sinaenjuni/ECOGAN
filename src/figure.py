

import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('grayscale')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 9}

plt.rc('font', **font)
# plt.rc('axes', titlesize=10)
# plt.rc('figure', titlesize=20)
# plt.rc('font', size=1)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9,4))


data1 = [5000] * 10
data2 = [5000] + [i*100 for i in range(1,10)]
name = [f"{i}" for i in range(10)]

ax1.bar(name, data1, color='black')
ax1.set_xlabel('(a)')
ax2.bar(name, data2, color='black')
ax2.set_xlabel('(b)')

# ax1.yaxis.set_minor_formatter('{x}k')
ax1.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")
ax2.yaxis.set_major_formatter(lambda x, pos: str(int(x/1000)) + "k")
plt.tight_layout()
plt.show()

