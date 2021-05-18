import matplotlib.pyplot as plt
from numpy import fromfile
import numpy as np
from scipy.sparse import coo_matrix


num = 1
neuron = 1024
bucketnumber = 64
tile_size = 64
open_file_path='../src/tmp.txt'
save_path_root = "../data_show/"


matrix = np.zeros((neuron, neuron))
file = open(open_file_path, 'r')
for eachline in file.readlines():
    x = eachline.split(' ')
    matrix[int(x[1])][int(x[0])] = 1
file_name = 'l' + str(num) + '_b' + str(bucketnumber) + '_t' + str(tile_size) + '.png'
plt.title(file_name)
plt.xlim(xmax = neuron, xmin = 0)
plt.ylim(ymax = 128, ymin = 0)
plt.xlabel("col")
plt.ylabel("row")
plt.plot(matrix)
save_file_path = save_path_root + 'n' + str(neuron) + "/" + "l" + str(num) + "/"
plt.savefig(save_file_path + "all.png")

for row_block in range(0, int((neuron + tile_size - 1) / tile_size)):
    for col_block in range(0, int((neuron + tile_size - 1)/ tile_size)):
        file =open(open_file_path, 'r')
        a = []
        b = []
        for eachline in file.readlines():
            x = eachline.split(' ')
            if(int(x[0]) >= tile_size * row_block and int(x[0]) < tile_size * (row_block + 1)):
                if(int(x[1]) >= tile_size * col_block and int(x[1]) < tile_size * (col_block + 1)):
                    a.append(int(x[1]))
                    b.append(int(x[0]))
        print(len(a))
        if(len(a) == 0):
            continue

        file_name = 'l' + str(num) + '_b' + str(bucketnumber) + '_r' + str(row_block) + '_c' + str(col_block) + '_t' + str(tile_size) + '.png'
        save_file_path = save_path_root + 'n' + str(neuron) + "/" + "l" + str(num) + "/"
        plt.title(file_name)
        plt.xlim(xmax = tile_size * (col_block + 1), xmin = tile_size * col_block)
        plt.ylim(ymax = tile_size * (row_block + 1), ymin = tile_size * row_block)
        plt.xlabel("col")
        plt.ylabel("row")
        plt.plot(a, b, '.')
        plt.savefig(save_file_path + file_name)

# plt.show()
