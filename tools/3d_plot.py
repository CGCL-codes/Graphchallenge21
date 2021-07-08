import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



labels_x = ['(b)\n(n)', '', '(b)\n(nn,k)', '(b,n)\n(nn,kk)', '', '', '', '']
labels_y = ['(b0,k0,n0)', '(b0,k0,n1)', '(b0,k1,n1)', '', '', '', '', '']
labels_z = ['(b0,k0,n0)', '', '', '(b,n,k,nn)', '(b,n,k,kk,nn)', '', '', '']


xs1 = [3]
ys1 = [2]
zs1 = [4]


xs2 = [4]
ys2 = [3]
zs2 = [5]



# 方式1：设置三维图形模式
fig = plt.figure() # 创建一个画布figure，然后在这个画布上加各种元素。
ax = Axes3D(fig) # 将画布作用于 Axes3D 对象上。

ax.scatter(xs1,ys1,zs1) # 画出(xs1,ys1,zs1)的散点图。
ax.scatter(xs2,ys2,zs2,c='r',marker='^')



ax.set_xlabel('Parallelism') # 画出坐标轴
ax.set_ylabel('Loop Tiling')
ax.set_zlabel('Execute Order')

locsx, labelsx = plt.xticks()  # Get the current locations and labels.
locsy, labelsy = plt.yticks()  # Get the current locations and labels.
# locsz, labelsz = plt.zticks()  # Get the current locations and labels.

plt.xticks(locsx, labels_x)  # Set label locations.
plt.yticks(locsy, labels_y)  # Set label locations.
# plt.zticks(locsz, labels_z)  # Set label locations.

plt.savefig("3d.png")
# plt.show()