# coding=utf-8

'''
朴素贝叶斯讲解:http://www.cnblogs.com/pinard/p/6069267.html

conclusion:朴素贝叶斯模型在这里做了一个大胆的假设，即n个维度之间相互独立,从上式可以看出，
这个很难的条件分布大大的简化了，但是这也可能带来预测的不准确性。你会说如果我的特征之间非常不独立怎么办？
如果真是非常不独立的话，那就尽量不要使用朴素贝叶斯模型了，考虑使用其他的分类方法比较好。
但是一般情况下，样本的特征之间独立这个条件的确是弱成立的，尤其是数据量非常大的时候。
虽然我们牺牲了准确性，但是得到的好处是模型的条件分布的计算大大简化了，这就是贝叶斯模型的选择。
'''


from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import numpy as np
input_file = 'adult.txt'
X=[]
Y=[]
count_lessthan50K=0
count_morethan50k=0
num_images_threshold = 10000

'''
使用数据集中的20000个数据点--每种类型10000个,保证初始类型没有偏差。
在模型训练时,如果大部分数据都属于一个类型,那么分类器就会倾向于这个类型
所以使用每个类型数据点数量相等的数据进行训练
'''

with open(input_file,'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_lessthan50K<num_images_threshold:
            X.append(data)
            count_lessthan50K +=1
        elif data[-1] == '>50K' and count_morethan50k<num_images_threshold:
            X.append(data)
            count_morethan50k +=1
        if count_morethan50k >=num_images_threshold and count_lessthan50K >=num_images_threshold:
            break
X = np.array(X)

label_encoder=[]
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:,i] = X[:,i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:,i] = label_encoder[-1].fit_transform(X[:,i])

X = X_encoded[:,:-1].astype(int)
Y = X_encoded[:,-1].astype(int)

# 建立分类器
# GaussianNB 为高斯分布
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X,Y)

#交叉验证
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
#random_state 为随机种子
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.25,random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train,Y_train)

y_test_pred = classifier_gaussiannb.predict(X_test)

f1 = cross_val_score(classifier_gaussiannb,X,Y,scoring='f1_weighted',cv=5)
print "F1 score: " + str(round(100*f1.mean(), 2)) + "%"

# Testing encoding on single data instance
input_data = ['39', 'Local-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', '2174', '0', '40', 'United-States']
count = 0
input_data_encoded = [-1] * len(input_data)
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        '''
        此处书中有误,LabelEncoder在分类新的单一数据时,仍要使用[]
        即labelEncoer.transform(['data'])
        '''
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1

input_data_encoded = np.array(input_data_encoded)
'''
此处书中有误,input_data_encoded后必须加上reshape(1,-1)
因为训练的分类器接受的是2D模型,所以最后只输入1D数据时,必须告之分类器这是只有一列的数据
'''
output_class = classifier_gaussiannb.predict(input_data_encoded.reshape(1,-1))
print label_encoder[-1].inverse_transform(output_class)[0]