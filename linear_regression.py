# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:06:35 2019

@author: reuve
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X=np.array([[31,22],[22,21],[40,37],[26,25]])
Y=np.array([2,3,8,12])

#def linear_regression(X,y):
#    return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(),X)),X.transpose()),y)
def my_regression(X,Y):
    res = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    print('\nmy_regression\t',res)
    return (res)

def check_regression(X,Y):
    res=my_regression(X,Y)
    print('predicted y\t\t:', np.dot(X,res))
    print('loss :\t\t', ((np.dot(X,res)-Y)**2).mean())

# 3D plane crossing (0,0,0)
check_regression(X,Y)

# 3rd feature: constant
X1=np.ones((4,3))
X1[:,:-1]=X
print("\nadd new constance feature")
check_regression(X1,Y)
#
# 3rd feature: x1-x2
X2=X1.copy()
X2[:,2]=X[:,0]-X[:,1]
print("\nadd new feature x2-x1")
check_regression(X2,Y)

# 3rd feature: (x1-x2)^2
X3=X1.copy()
X3[:,2]=(X[:,0]-X[:,1])**2
print("\nadd new feature (x1-x2)^2")
check_regression(X3,Y)

# 3rd feature: (x1-x2)^2, 4th feature constant
X4=np.ones((4,4))
X4[:,:-1]=X3
print("\nadd two new features (x1-x2)^2 and a constant(1)")
check_regression(X4,Y)

#2
max_num_points = 200

#2.1:
training = pd.read_csv('data_for_linear_regression.csv')

##2.2
# Convert all data to matrix for easy consumption
x_training_all = training['x'].values
x_training = x_training_all[0:max_num_points]

y_training_all = training['y'].values
y_training = y_training_all[0:max_num_points]

#2.3:
plt.title('X and Y')
plt.scatter(x_training, y_training,  color='red', marker=".")

#2.4:
b_all = np.ones((1,x_training_all.shape[0]))
b = b_all[:,0:max_num_points]
b_and_1 = np.hstack(
        (x_training.reshape(max_num_points, 1),
         b.reshape(max_num_points ,1))
        )
res = my_regression(b_and_1, y_training)

#2.5:
y_out = res[0]*x_training + res[1]*b
plt.plot(x_training, y_out.reshape(max_num_points,),  color='black')
plt.show()

#2.6
plt.figure()
plt.title("test group")
y_out = res[0]*x_training_all[max_num_points:] + res[1]*b_all[:,max_num_points:]
#print(y_out.shape)
plt.plot(x_training_all[max_num_points:], y_out.reshape(500,),  color='black')

plt.xlim((0,100))
plt.ylim(0,100)
plt.scatter(x_training_all[max_num_points:], 
            y_training_all[max_num_points:],  
            color='red', marker=".")
plt.show()

