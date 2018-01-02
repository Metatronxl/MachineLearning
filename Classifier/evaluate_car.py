# coding=utf-8
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve,learning_curve
from matplotlib import pyplot as plt

"""
numpy.array[:]用法

array[:i]  为取前行
array[:,i] 为取第i列


转换器用于数据预处理和数据转换，主要是三个方法：
fit()：训练算法，设置内部参数。
transform()：数据转换。
fit_transform()：合并fit和transform两个方法。

"""


input_file = 'car.data.txt'
X = []
count = 0



with open(input_file,'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)

X = np.array(X)
label_encoder = []
X_encoder = np.empty(X.shape)
for i,item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoder[:,i]=label_encoder[-1].fit_transform(X[:,i])

X = X_encoder[:,:-1].astype(int)
Y = X_encoder[:,-1].astype(int)

#建立随机森林分类器
params = {'n_estimators':200,'max_depth':8,'random_state':7}
classifier = RandomForestClassifier(**params)
classifier.fit(X,Y)

'''
    交叉验证
    CV代表选择的验证方法
    3 means three-fold cross-validation (三折交叉验证)
'''
# accuracy = cross_val_score(classifier,X,Y,scoring='accuracy',cv=3)
# print "Accuracy of the classifier:" + str(round(100*accuracy.mean(),2)) + "%"
# Accuracy of the classifier:78.19%


'''
验证曲线

numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
在指定的间隔内返回均匀间隔的数字。
返回num均匀分布的样本，在[start, stop]。
这个区间的端点可以任意的被排除在外

'''

# classifier = RandomForestClassifier(max_depth=4,random_state=7)
# parameter_grid = np.linspace(25,200,8).astype(int)
# train_scores, validation_scores = validation_curve(classifier, X, Y,
#         "n_estimators", parameter_grid, cv=5)
# print "\n##### VALIDATION CURVES #####"
# print "\nParam: n_estimators\nTraining scores:\n", train_scores
# print "\nParam: n_estimators\nValidation scores:\n", validation_scores


'''
学习曲线

axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；
axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。
还可以这么理解，axis是几，那就表明哪一维度被压缩成1。

'''
classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])

train_sizes,train_scores,validation_scores = learning_curve(classifier,X,Y,train_sizes=parameter_grid,cv=5)

# print "\n##### LEARNING CURVES #####"
# print "\nTraining scores:\n", train_scores
# print "\nValidation scores:\n", validation_scores

print np.average(train_scores,axis=0)
print np.average(train_scores,axis=1)
print train_scores

plt.figure()
plt.plot(parameter_grid,100*np.average(train_scores,axis=1),color = 'black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()