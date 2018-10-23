#!/usr/bin/python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X1 = np.arange(0.01, 0.99, 0.01).reshape(1, -1)
X2 = np.arange(0.01, 0.99, 0.01).reshape(1, -1)
Y = np.random.randn(X1.shape[0], X1.shape[1])
Y[Y < 0.5] = 0
Y[Y >= 0.5] = 1

X1, X2 = np.meshgrid(X1, X2)
R = Y * np.log(X1) + (1 - Y) * np.log(1 - X1) + Y * np.log(X2) + (1 - Y) * np.log(1 - X2)

Z = R

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X1, X2, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()
