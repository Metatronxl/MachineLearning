import numpy as np
import matplotlib.pyplot as plt

X = np.array([[3,1],[2,5],[1,8],[6,4],[5,2],[3,5],[4,7],[4,-1]])
Y = [0,1,1,0,0,1,1,0]

class_0 = np.array([X[i] for i in range(len(X)) if Y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if Y[i] == 1])

plt.figure()
# class_0[:,0]means the first element in (x,y),
#ex: class_0[:,0] is [3 6 5 4]
#    class_0[:,1] is [1 4 2 -1]
plt.scatter(class_0[:,0],class_0[:,1],color='black',marker='s')
plt.scatter(class_1[:,0],class_1[:,1],color='black',marker='x')

line_x = range(10)
line_y = line_x

plt.plot(line_x,line_y,color='blue',linewidth=3)

plt.show()

