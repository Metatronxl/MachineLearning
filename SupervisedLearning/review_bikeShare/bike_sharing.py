# -*- coding: utf-8 -*-
import sys
import csv
import numpy as np
##tips:sys中append('..')后,子模块才能找到其他文件夹中的文件(改文件夹中已经生产__init__.py),
sys.path.append("..")
from sklearn.ensemble import RandomForestRegressor
from SupervisedLearning.guessHousePrice.housing import plot_feature_importances
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,explained_variance_score


def load_dataset(filename):
    file_reader = csv.reader(open(filename,'rb'),delimiter=',')
    X,Y = [],[]
    for row in file_reader:
        X.append(row[2:13])
        Y.append(row[-1])
    feature_names = np.array(X[0])
    return np.array(X[1:]).astype(np.float32),np.array(Y[1:]).astype(np.float32),feature_names


if __name__ == '__main__':
    x,y,feature_names = load_dataset('bike_day.csv')
    x,y = shuffle(x,y,random_state=7)

    num_traing = int(len(x)*0.9)
    x_train,y_train = x[:num_traing],y[:num_traing]
    x_test,y_test = x[num_traing:],y[num_traing:]
    #n_estimators是指评估器(estimator)的数量,表示随机森林需要使用的决策树数量
    #max_depth是指每个决策树的最大深度
    #min_samples_split是指决策树分裂一个节点需要用到的最小数据样本量
    rf_regressor = RandomForestRegressor(n_estimators=1000,max_depth=10,min_samples_split=2)
    rf_regressor.fit(x_train,y_train)

    y_pred = rf_regressor.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    evs = explained_variance_score(y_test,y_pred)

    print "\n#### Random Forest regressor performance ####"
    print "Mean squared error =", round(mse, 2)
    print "Explained variance score =", round(evs, 2)

    plot_feature_importances(rf_regressor.feature_importances_,'Random Forest regresor',feature_names)
