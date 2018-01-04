# -*- coding: utf-8 -*-
'''
SVM常用核函数(讲解很透彻!): http://blog.csdn.net/batuwuhanpei/article/details/52354822
SVM通俗导论: http://blog.csdn.net/v_july_v/article/details/7624837
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import utilities

# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)
ΩΩ
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

'''
show the data
'''
# plt.figure()
# plt.scatter(class_0[:,0],class_0[:,1],facecolors='black',edgecolors='black',marker='s')
# plt.scatter(class_1[:,0],class_1[:,1],facecolors='None',edgecolors='black',marker='s')
#
# plt.title('input data')
# plt.show()

X_train,X_test,y_train,y_train = train_test_split(X,y,test_size=0.25,random_state=5)

