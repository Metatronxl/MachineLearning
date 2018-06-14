import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation

# Load multivar data in the input file
def load_data(input_file):
    X = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data)

    return np.array(X)


## data preprocessing
def file2Metrix(filename):

    file = open(filename)
    arrayOfLines = file.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []

    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVector