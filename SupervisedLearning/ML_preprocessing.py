# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing

# 注意numpy的array最多只接受2个参数,所以把多个数组放到一个数组中形成多维数组就可以解决问题
data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9, -4.3]
                 ])

# 1.均值移除 Mean removal
def MeanRemoval():
#    Standardization即标准化，尽量将数据转化为均值为零，方差为一的数据，
#   形如标准正态分布（高斯分布）。实际中我们会忽略数据的分布情况，仅仅是通过改变均值来集中数据，
#  然后将非连续特征除以他们的标准差。sklearn中 scale函数提供了简单快速的singlearray-like数据集操作。
    data_standardized = preprocessing.scale(data)

    print "\n Mean = ",data_standardized.mean(axis=0)
    print "\n deviation = ",data_standardized.std(axis=0)
#output :Mean = [[ 0.         -1.22474487  1.33630621]
#                 [ 1.22474487  0.         -0.26726124]
#                 [-1.22474487  1.22474487 -1.06904497]]
#        deviation = [ 1.  1.  1.  1.]


#2. 范围缩放 Scaling
def Scaling():
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data_scaled = data_scaler.fit_transform(data)
    print "\n Min max scaled data = ",data_scaled

# output: 范围缩放后,所有数据点的特征数值都位于指定的数值范围内
 # Min max scaled data =  [[ 1.          0.          1.          0.        ]
 # [ 0.          1.          0.41025641  1.        ]
 # [ 0.33333333  0.87272727  0.          0.14666667]]

# 归一化 Normalization

def Normalization():
# 使用L1范数可以使特征向量的数值之和为1,这个方法经常用于确保数据点没有因为特征的基本性质而产生较大差异,即确保数据处于同一数量级,提高不同
# 数据特征的可比性
   data_normalized = preprocessing.normalize(data,norm='l1')
   print "\n normalized data = ",data_normalized

#output:
 # normalized data =  [ [ 0.25210084 -0.12605042  0.16806723 -0.45378151]
 #                      [ 0.          0.625      -0.046875    0.328125  ]
 #                      [ 0.0952381   0.31428571 -0.18095238 -0.40952381]]


# 二值化 Binarization
def Binarization():
    data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
    print "\n Binarization data = ",data_binarized

#output:
# Binarization data =  [[ 1.  0.  1.  0.]
#                       [ 0.  1.  0.  1.]
#                       [ 0.  1.  0.  0.]]

# 独热编码 One-Hot Encoding
def One_hot_encoding():
    encoder = preprocessing.OneHotEncoder()
    encoder.fit([[0,2,1,12],[1,3,5,3],[2,3,5,12],[1,2,4,3]])
    encoder_vector = encoder.transform([[2,3,5,3]]).toarray()
    print "\n Encoded vector = ",encoder_vector
#ex:每个特征向量的第三个特征是1,5,2,4四个不重复的值,所以独热编码向量的长度是4,对5进行编码,结果为【0,1,0,0】
#output:
# Encoded vector =  [[ 0.  0.  1.  0.  1.  0.  0.  0.  1.  1.  0.]]
if __name__ == '__main__':

    # MeanRemoval()
    # Scaling()
    # Normalization()
    # Binarization()
    One_hot_encoding()