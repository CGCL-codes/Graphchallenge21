import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
 
open_file_path = "../data/neuron16384-l120-categories.tsv"

neuron = 16384
batch = 60000

file = open(open_file_path, 'r')

array = []
x_axis = []
for i in range(0, 6000):
    array.append(0)
    x_axis.append(i)

max_v = 0
for eachline in file.readlines():
    x = eachline.split(' ')
    xx = int(x[0]) - 1
    print(xx)
    array[int(xx/10)] = array[int(xx/10)] + 1
    if max_v < array[int(xx/10)]:
        max_v = array[int(xx/10)]



plt.xlim(xmax = 256, xmin = 0)
plt.ylim(ymax = max_v, ymin = 0)

plt.plot(x_axis, array, '.')

plt.savefig("tmp.png")