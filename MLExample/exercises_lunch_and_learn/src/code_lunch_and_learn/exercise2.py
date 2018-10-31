# -*- coding: utf-8 -*-
"""
Linear regression example 2
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def featureNormalize(X, m, n):
    X_norm = np.zeros((m,n));
    mu = np.zeros((1,n))
    sigma = np.zeros((1,n))

    mu = np.mean(X, axis=0, keepdims=1)
    sigma = np.std(X, axis=0, keepdims=1)

    for k in range(n):
        X_norm[:, k] = (X[:, k] - mu[:, k]) / sigma[:, k]

    return X_norm

def computeCost(X, y, theta):
    h0 = np.sum(np.asarray(theta.transpose()) * np.asarray(X), axis=1, keepdims=1)
    h0 -= y
    h0 = np.sum(np.power(h0, 2))
    J_history = h0 / (2*X.shape[0])

    return J_history

def gradientDescent(X, y, theta, alpha, num_iters):

    length_theta = theta.shape[0]
    J_history = np.zeros((num_iters, 1));
    temp = np.zeros((length_theta, 1));

    for i in range(num_iters):

        h0 = np.sum(theta.transpose() * X, axis=1, keepdims=1)
        h0 -= y

        for k in range(length_theta):
            temp[k] = np.sum(np.multiply(h0, X[:, k:k+1]), axis=0)
            theta[k] = theta[k] - ((alpha / X.shape[0]) * temp[k])

        J_history[i] = computeCost(X, y, theta);

    return [theta, J_history]


###############################################################################
#Example 2 hypothesis with 2 features
###############################################################################

#Opening file with data
input_file = 'car_data.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))
X_train = np.c_[data[0:300, 1:2], data[0:300, 2:3]]
X_test = np.c_[data[300:, 1:2], data[300:, 2:3]]
y_train = data[0:300,0:1]
y_test = data[300:,0:1]
m = X_train.shape[0]    # Training data set size
m_t = X_test.shape[0]   # Test data set size
n = X_train.shape[1]    # Number of features

#Plot input data
plt.figure(num='Input data (horsepower)')
plt.plot(X_train[:,0:1], y_train, 'rx')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

plt.figure(num='Input data (weigth)')
plt.plot(X_train[:,1:2], y_train, 'rx')
plt.xlabel('Weight (Kg)')
plt.ylabel('Miles per Galon')

#Normalize features
X_norm = featureNormalize(X_train, m, n)
X_test_norm = featureNormalize(X_test, m_t, n)

#Add an additional column with 1's
X_norm = np.c_[np.ones(m), X_norm]
X_test_norm = np.c_[np.ones(m_t), X_test_norm]

alpha = 0.01
num_iters = 3000
#Initialize theta to 0
theta = np.zeros((n + 1, 1))

#Run gradient descent
theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

#Get predicted values from obtained theta
prediction =np.asmatrix(X_test_norm) * theta

#Plot the results
plt.figure(num='Results for exercise 2')
plt.plot(X_train[:, 0:1], y_train, 'rx', X_test[:, 0:1], prediction, 'o')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')




