# -*- coding: utf-8 -*-
import sys
import  numpy as np
from matplotlib import pyplot as plt

x =[]
y =[]

with open('data_multivar.txt','r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt,yt = data[:-1],data[-1]
        x.append(xt)
        y.append(yt)

num_training = int(0.8*len(x))
num_test = len(x)-num_training

x_train = np.array(x[:num_training])
y_train = np.array(y[:num_training])

x_test = np.array(x[num_training:])
y_test = np.array(y[num_training:])

from sklearn import linear_model
# sklearn的metric包含衡量回归器拟合效果的重要指标(metric)
from sklearn import metrics as sm

#如果希望模型对异常值不那么敏感,就需要设置一个较大的Alpha值
rigde_regressor = linear_model.Ridge(alpha=0.01)

#train
rigde_regressor.fit(x_train,y_train)

#predict the output
y_train_pred_ridge = rigde_regressor.predict(x_train)
y_test_pred_ridge = rigde_regressor.predict(x_test)


def metric():
    print "\nRIDGE:"
    print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2)
    print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2)
    print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2)
    print "Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2)
    print "R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2)

def show():

    plt.figure()
    plt.scatter(x_train,y_train,color='green')
    plt.plot(x_train,y_train_pred_ridge,color='black',linewidth=4)
    plt.title('multivar_regression')
    plt.show()

if __name__ == '__main__':
        metric()
