# -*- coding: utf-8 -*-
import  sys
import numpy as np
from sklearn import linear_model
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import cPickle as pickle

x = []
y = []

with open("data_singlevar.txt",'r') as f:
    for line in f.readlines():
        xt,yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)

num_training = int(0.8*len(x))
num_test = len(x)-num_training
#reshape的参数意思为(row,col) 具体参见https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape

#训练数据
x_train = np.array(x[:num_training]).reshape((num_training,1))
print x_train,"\n"
y_train = np.array(y[:num_training])
print y_train
#测试数据
x_test = np.array(x[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])
#create linear_regressor
linear_regressor = linear_model.LinearRegression()
#训练模型
linear_regressor.fit(x_train,y_train)

y_train_pred = linear_regressor.predict(x_train)

def train():

    # figure start
    plt.figure()
    # 数据点的分布
    plt.scatter(x_train,y_train,color = 'green')
    # 线性回归预测
    plt.plot(x_train,y_train_pred,color='black',linewidth=4)
    #figure title
    plt.title('Training data')
    # show the pic
    plt.show()

# ---
#使用测试数据来看预测数据的准确性

def predict():
    y_test_pred = linear_regressor.predict(x_test)
    plt.scatter(x_test,y_test,color='green')
    plt.plot(x_test,y_test_pred,color='black',linewidth=4)
    plt.show()

def saveModel():
    #save model
    output_model_file = 'saved_model.pkl'
    with open(output_model_file,'w') as f:
        pickle.dump(linear_regressor,f)
def loadModel():
    output_model_file = 'saved_model.pkl'
    with  open(output_model_file,'r') as f:
        model_linregr = pickle.load(f)

if __name__ == '__main__':

    train()
    # predict()