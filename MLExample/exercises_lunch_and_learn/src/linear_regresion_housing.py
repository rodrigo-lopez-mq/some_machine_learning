# -*- coding: utf-8 -*-
"""
Linear regression example
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

    return [X_norm, mu, sigma]

def computeCost(X, y, theta):
    h0 = np.sum(np.asarray(theta.transpose()) * X, axis=1, keepdims=1)
    h0 -= y
    h0 = np.sum(np.power(h0, 2))
    J_history = h0 / (2*m)

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
            theta[k] = theta[k] - ((alpha / m) * temp[k])

        J_history[i] = computeCost(X, y, theta);

    return [theta, J_history]

###############################################################################
#Example 1 with small data set
###############################################################################

#Opening file with data
input_file = 'real_state_simple.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,1))
X = data[:,1:]
y = data[:,0:1]
m = X.shape[0]    #Data set size
n = X.shape[1]    #Number of features

##Plot input data
#plt.figure(num='Input data 1')
#plt.plot(X, y, 'rx')
#plt.xlabel('Size ($feet^2$)')
#plt.ylabel('Price ($)')

#Normalize features
X_norm, mu, sigma = featureNormalize(X, m, n)

#Add an additional column with 1's
X_norm = np.c_[np.ones(m), X_norm]

alpha = 0.01
num_iters = 300
theta = np.zeros((n + 1, 1))

#Run gradient descent
theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)

#Get predicted values from obtained theta
prediction =np.asmatrix(X_norm) * theta

#Plot the results
plt.figure(num='Results 1')
plt.plot(X[:, 0:], y, 'rx', X[:, 0:], prediction, '-')
plt.xlabel('Size ($feet^2$)')
plt.ylabel('Price ($)')

#Plot Cost calculated during gradient descent
#plt.figure(num='Cost during gradient descent')
#plt.plot(J_history)
#plt.xlabel('Iterations')
#plt.ylabel('Cost J($\\theta_0,\\theta_1$)')


###############################################################################
#Example 2 with big data set
###############################################################################

##Opening file with data
#input_file = 'real_state.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(2,5))
#X = data[:,1:2]
#y = data[:,0:1]
#m = X.shape[0]    #Data set size
#n = X.shape[1]    #Number of features
#
##Plot input data
#plt.figure(num='Input data 2')
#plt.plot(X, y, 'rx')
#plt.xlabel('Size ($feet^2$)')
#plt.ylabel('Price (USD)')
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.01
#num_iters = 300
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
#
##Get predicted values from obtained theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results 2')
#plt.plot(X[:, 0:], y, 'rx', X[:, 0:], prediction, '-')
#plt.xlabel('Size ($feet^2$)')
#plt.ylabel('Price (USD)')

###############################################################################
#Example 3 with better hypothesis with madeup feature
###############################################################################

##Opening file with data
#input_file = 'real_state.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(2,5))
#X = data[:,1:2]
#y = data[:,0:1]
#m = X.shape[0]    #Data set size
#
##Adding a new feature
#X_f = np.c_[X, 3000/X]
#
#n = X_f.shape[1]    #Number of features
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X_f, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.1
#num_iters = 300
##Initialize theta to some value
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
#
##Get predicted values from obtained theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results with made up feature')
#plt.plot(X[:, 0:], y, 'rx', X[:, 0:], prediction, 'o')
#plt.xlabel('Size ($feet^2$)')
#plt.ylabel('Price (USD)')

###############################################################################
#Example 4 hypothesis with 2 features
###############################################################################

##Opening file with data
#input_file = 'real_state.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(2,5,6))
#X = np.c_[data[:, 1:2], data[:, 2:3]**2]
#y = data[:,0:1]
#m = X.shape[0]    #Data set size
#n = X.shape[1]    #Number of features
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.01
#num_iters = 3000
##Initialize theta to some value
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
#
#
##Get predicted values from obtained theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results with an additional feature')
#plt.plot(X[:, 0:1], y, 'rx', X[:, 0:1], prediction, 'o')

###############################################################################
#Example 5 hypothesis with 2 features + better fit curve
###############################################################################

##Opening file with data
#input_file = 'real_state.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(2,5,6))
#X = np.c_[data[:, 1:2], data[:, 2:3]**2]
#y = data[:,0:1]
#
#m = X.shape[0]    #Data set size
#
#vector = X[:,0:1]
#poly_params = np.polyfit(vector.flatten(), y, deg=3)
#
#new_feature = (poly_params[0,0]*X[:,0:1]**3) + (poly_params[1,0]*X[:,0:1]**2) + (poly_params[2,0]*X[:,0:1]) + poly_params[3,0]
#X = np.c_[X[:,0:1], new_feature[:,0:1]**2 ,X[:,1:2]]
#
#n = X.shape[1]    #Number of features
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.1
#num_iters = 3000
##Initialize theta to some value
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
#
##Get predicted values from obtained theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results for exercise 5')
#plt.plot(X[:, 0:1], y, 'rx', X[:, 0:1], prediction, 'o')