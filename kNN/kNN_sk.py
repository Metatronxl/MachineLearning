# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utilities import load_data,file2Metrix
from sklearn import neighbors,preprocessing


def testKNN():

   returnMat,classLabel = file2Metrix("datingTestSet2.txt")
   hoRatio = 0.10
   num_neighbors = 3
   normMat = preprocessing.normalize(returnMat,'l2')
   m = normMat.shape[0]
   numTestVecs = int(m*hoRatio)
   classifier = neighbors.KNeighborsClassifier(num_neighbors,weights='distance')
   classifier.fit(returnMat[numTestVecs:,],classLabel[numTestVecs:])

   # dist,indices = classifier.kneighbors(returnMat[:numTestVecs,])
   #
   # for (d,i) in zip(dist,indices):
   #     print 'dist:%s  ---  indices:%s'%(d,i)

    ### predict 来得到预测的结果
   error_count = 0
   for item,result in zip(returnMat[:numTestVecs],classLabel[:numTestVecs]):
       #告诉模型这是一个example,而不是多个
       test_value = item.reshape(1,-1)
       predict_result = classifier.predict(test_value)[0]
       print "predicted output:",classifier.predict(test_value)[0]
       if predict_result != result:
           error_count +=1
   print "error_rate is %f" % (error_count/float(m*0.9))






if __name__ == '__main__':


    # showFigure(returnMat)
    testKNN()
