# -*- coding: utf-8 -*-
from numpy import *
import operator
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    # axis代表的是第几维的数组 0为1维,1为2维
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []

    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3] # 第index行的数据
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector


## Normalization
def autoNorm(dataSet):
    minVals = dataSet.min(0) ## 0 means 数组的第二维(纵列)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  #shape[1]为第二维的长度(横列),shape[0]为第一维的长度(纵列)

    ## newValue = (oldvalue-min)/(max-min)
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))


    return normDataSet,ranges,minVals

## 分类器针对约会网站的测试代码
def datingClassTest():

    hoRatio = 0.10
    datingDataMat,datingLables = file2matrix("datingTestSet2.txt")
    normMat,ranges,MinVals = autoNorm(datingDataMat)
    m = normMat.shape[0] ## cols
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLables[numTestVecs:m],3)
        print "the classifier came back with :%d , the real answer is :%d"%(classifierResult,datingLables[i])

        if(classifierResult != datingLables[i]): errorCount +=1.0
    print "the total error rate is :%f" %(errorCount/float(numTestVecs))


if __name__ == '__main__':

    # returnMat,classLabelVector = file2matrix("datingTestSet2.txt")
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(returnMat[:,1],returnMat[:,2])
    # plt.show()

    datingClassTest()