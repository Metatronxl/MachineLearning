# -*- coding: utf-8 -*-
'''
将所有的弱分类器的结果加权求和就可以得到最后的结果 :P

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
                # print "split : dim %d ,thresh %.2f , thresh inequal: %s, the weighted error is %.3f" % \
                #     (i,threshVal,inequal,weightdError)

                if weightdError < minError:
                    minError = weightdError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40): # 单层决策树(decision stump)
    weakClassArr = []
    m = shape(dataArr)[0] # 数组第一维的个数
    D = mat(ones((m,1))/m) # 一开始的权重是相等的
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print "current D:",D.T
        # 算法为分配器分配的权重值
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))  # 确保不会发生除0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T

        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D= multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst #将错误率累加计算,可以更好的估计出这个值是否是错误的
        print "aggClassEst: ",aggClassEst.T

        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) # numpy.sign() 大于0的返回1,小于0的返回-1 ,等于0的返回0
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate,"\n"
        if errorRate == 0.0:break
    return weakClassArr


def addClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst #将多个弱分类器的错误率累加计算,可以更好的估计出这个值是否是错误的
        print aggClassEst
    return sign(aggClassEst)

if __name__ == '__main__':
    dataMat,classLabels = loadSimpData()
    adaBoostTrainDS(dataMat,classLabels,9)