# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

#demo data
amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - 0.5 * amplitude

#noise data

#首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）
# 还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）
# 原始矩阵，而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），
# 会影响（reflects）原始矩阵。
Y = np.sinc(X).ravel()
Y += 0.2 * (0.5 - np.random.rand(Y.size))

plt.figure()
# c means color
plt.scatter(X,Y,s=40,c='k')
plt.title('Input data')
# np.linespace 定义范围以及个数
#
# x = np.arange(3) --> array([0,1,2])
# >> x[:,np.newaxis]
# array([[0],
#        [1],
#        [2]])
#所以说np.ndarray[:,np.newaxis]的功能近似于np.ndarray.reshape(-1,1)
# review
# reshape(-1,1) 是多个数据(一列)
# reshape(1,-1) 是单个数据(一行)
x_values = np.linspace(-0.5*amplitude, 0.5*amplitude, 10*num_points)[:, np.newaxis]
print type(x_values)
print x_values
n_neighbors = 8

knn_regression = neighbors.KNeighborsRegressor(n_neighbors,weights='distance')
y_values = knn_regression.fit(X,Y).predict(x_values)

plt.figure()
# 训练数据
plt.scatter(X,Y,s=40,c='k',facecolors='none',label='input data')
# plot 函数用来画线
plt.plot(x_values,y_values,c='k',linestyle='--',label='predicted values')
# xlim和ylim定义x&y坐标轴的范围
plt.xlim(X.min()-1,X.max()+1)
plt.ylim(Y.min()-1,Y.max()+1)
# plt.aopxis('tight')
## legned 是图例 0代表选择最佳位置
plt.legend(loc=0)
plt.title('K nearest Neighbors Regressor')
plt.show()
