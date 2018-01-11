# -*- coding: utf-8 -*-

'''
numpy数据过滤技巧:
http://blog.csdn.net/linzch3/article/details/58584865

'''

'''
此处使用单层决策树来作为弱学习器
同时基于权重D(自定义)而不是其他错误计算指标来评价分类器

详细代码解释参考《MachineLearning in Action》p121
'''


from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneg):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneg == 'lt':
        #数据过滤
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0 ; bestStump = {};bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/ numSteps
        for j in range(-1,int(numSteps)+1): ## 阀值可以设置在取值范围之外
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightdError = D.T * errArr
                print "split : dim %d ,thresh %.2f , thresh inequal: %s, the weighted error is %.3f" % \
                    (i,threshVal,inequal,weightdError)

                if weightdError < minError:
                    minError = weightdError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump,minError,bestClasEst




if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    D = mat(ones((5,1))/5)
    print(buildStump(dataMat,classLabels,D))
    # print