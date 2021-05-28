import matplotlib.pyplot as plt
from numpy import fromfile
import numpy as np
from scipy.sparse import coo_matrix
import sys


num = sys.argv[1]
neuron = 16384
bucketnumber = 512
tile_size_str = sys.argv[2]

draw_num = sys.argv[3]
draw_num = int(draw_num)

tile_size = int(tile_size_str)

open_file_path='../data/neuron16384/n16384-l'+ num + '.tsv'
save_path_root = "../data_show/"


def hashElement(v):
    if v >= neuron:
        return v
    else:
        return v / bucketnumber + (v % bucketnumber) * (neuron / bucketnumber)


now_num = 0
for row_block in range(0, int((neuron + tile_size - 1) / tile_size)):
    for col_block in range(0, int((neuron + tile_size - 1)/ tile_size)):
        file = open(open_file_path, 'r')
        row = []
        col = []
        for eachline in file.readlines():
            x = eachline.split('\t')
            col_h = hashElement(int(x[0]) - 1)
            row_h = hashElement(int(x[1]) - 1)
            if(col_h >= tile_size * col_block and col_h < tile_size * (col_block + 1)):
                if(row_h >= tile_size * row_block and row_h < tile_size * (row_block + 1)):
                    col.append(col_h)
                    row.append(row_h)
        print(len(row))
        if(len(row) == 0):
            continue
        now_num = now_num + 1
        file_name = 'l' + str(num) + '_b' + str(bucketnumber) + '_r' + str(row_block) + '_c' + str(col_block) + '_t' + str(tile_size) + '.png'
        save_file_path = save_path_root + 'n' + str(neuron) + "/" + "l" + str(num) + "/"
        plt.title(file_name)
        plt.xlim(xmax = tile_size * (row_block + 1), xmin = tile_size * row_block)
        plt.ylim(ymax = tile_size * (col_block + 1), ymin = tile_size * col_block)
        plt.xlabel("row")
        plt.ylabel("col")
        plt.plot(row, col, '.')
        plt.savefig(save_file_path + file_name)
        if(draw_num == now_num):
            exit()
        


# plt.show()
