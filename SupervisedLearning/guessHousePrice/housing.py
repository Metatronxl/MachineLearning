# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data =datasets.load_boston()

X,Y = shuffle(housing_data.data,housing_data.target,random_state = 7)

num_training = int(len(X)*0.8)
X_train,Y_train = X[:num_training],Y[:num_training]
X_test,Y_test = X[num_training:],Y[num_training:]

#决策树回归模型拟合
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train,Y_train)
#带AdaBoost算法的决策树回归模型拟合
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=400,random_state=7)
ab_regressor.fit(X_train,Y_train)

def test_dt_regressor():
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(Y_test,y_pred_dt)
    evs = explained_variance_score(Y_test,y_pred_dt)
    print "\n ### Decision Tree performance ###"
    print "Mean squared error =",round(mse,2)
    print "Explained variance score = ",round(evs,2)

def test_ada_regressor():
    y_pred_dt = ab_regressor.predict(X_test)
    mse = mean_squared_error(Y_test,y_pred_dt)
    evs = explained_variance_score(Y_test,y_pred_dt)
    print "\n ### AdaBoost performance ###"
    print "Mean squared error =",round(mse,2)
    print "Explained variance score = ",round(evs,2)

def plot_feature_importances(feature_importances,title,feature_names):
    #将数据放置在0~100范围内
    feature_importances = 100.0*(feature_importances/max(feature_importances))
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0])+0.5
    plt.figure()
    plt.bar(pos,feature_importances[index_sorted],align='center')
    plt.xticks(pos,feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()



def show_feature_importance():
    # 判断哪些特征更加重要
    plot_feature_importances(dt_regressor.feature_importances_,'Decision Tree regressor',housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_,'AdaBoost regressor',housing_data.feature_names)


if __name__ == '__main__':
    # Conclusion:研究结果表明AdaBoost算法可以让误差更小,且解释方差分更接近1
    # adaBoost 的思想是将不同版本的算法结果进行组合,加权汇总获取最终结果
    # 详解AdaBoost:http://www.jianshu.com/p/389d28f853c0
    # Scikit learn adaBoost参数详解 http://www.cnblogs.com/pinard/p/6136914.html
    # test_ada_regressor()
    # test_dt_regressor()
    show_feature_importance()