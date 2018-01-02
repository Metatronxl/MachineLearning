# coding=utf-8
"""
朴素贝叶斯讲解 http://www.cnblogs.com/leoo2sk/archive/2010/09/17/naive-bayesian-classifier.html
[机器学习]Cross-Validation（交叉验证）详解 https://zhuanlan.zhihu.com/p/24825503?utm_source=tuicool&utm_medium=referral
"""
from sklearn.naive_bayes import GaussianNB
from logistic_regression import plot_classifier
# from sklearn import cross_validation
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np

input_file = 'data_multivar.txt'

X=[]
Y=[]

with open(input_file,'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        Y.append(data[-1])
X = np.array(X)
Y = np.array(Y)

classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X,Y)
y_pred = classifier_gaussiannb.predict(X)


# numpy shape[0] 返回第一纬度的个数(此处为X.shape【0】为400 ,X.shape[1]为2
#(Y == y_pred).sum()为numpy方法,返回两个list中相同的个数
accurancy = 100.0 * (Y == y_pred).sum() / X.shape[0]
#accruancy 为分类器的精准性
# print accurancy

# plot_classifier(classifier_gaussiannb,X,Y)
"""
#############添加测试集#########
"""
#test_size 为测试集的比例
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=5)

classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train,Y_train)

y_test_pred = classifier_gaussiannb_new.predict(X_test)

accurancy = 100.0*(y_test_pred == Y_test).sum() / X_test.shape[0]
# print accurancy
# plot_classifier(classifier_gaussiannb_new,X_test,Y_test)

"""
##############交叉验证############
"""
num_validations = 5
accurancy = cross_val_score(classifier_gaussiannb,X,Y,scoring='accuracy',cv = num_validations)
print "Accurancy:" + str(round(100*accurancy.mean(),2))+"%" # Accurancy:99.5%


"""
# print Y==y_pred  result:
# [ True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
    。。。
    。。。
    。。。
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True  True  True  True]

"""