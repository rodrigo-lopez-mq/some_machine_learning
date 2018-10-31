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
#Example 1 with small data set
###############################################################################

##Opening file with data
#input_file = 'car_data_simple.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,1))
#X = data[:,1:]
#y = data[:,0:1]
#m = X_train.shape[0]    #Data set size
#n = X_train.shape[1]    #Number of features
#
##Plot input data
#plt.figure(num='Input data 1')
#plt.plot(X, y, 'rx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')
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
#plt.figure(num='Results 1')
#plt.plot(X[:, 0:], y, 'rx', X[:, 0:], prediction, '-')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

##############################################################################
#Example 2 with big data set
##############################################################################

#Opening file with data
input_file = 'car_data.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))
X_in = data[:,1:2]
X_train = data[0:300,1:2]
X_test = data[300:,1:2]
y_train = data[0:300,0:1]
y_test = data[300:,0:1]
m = X_train.shape[0]    # Training data set size
m_t = X_test.shape[0]   # Test data set size
n = X_train.shape[1]    # Number of features

#Plot input data
plt.figure(num='Input data 2')
plt.plot(X_train, y_train, 'rx')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

#Normalize features
X_norm = featureNormalize(X_train, m, n)
X_test = featureNormalize(X_test, m_t, n)

#Add an additional column with 1's
X_norm = np.c_[np.ones(m), X_norm]

alpha = 0.01
num_iters = 300
theta = np.zeros((n + 1, 1))

#Run gradient descent
theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

#Get predicted values from obtained theta
prediction =np.asmatrix(X_norm) * theta

#Plot the results
plt.figure(num='Results 2')
plt.plot(X_train[:, 0:], y_train, 'rx', X_train[:, 0:], prediction, 'bx')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')
#
###############################################################################
##Visualization of cost for given theta values
###############################################################################
#
##Plot Cost calculated during gradient descent
#plt.figure(num='Cost during gradient descent')
#plt.plot(J_history)
#plt.xlabel('Iterations')
#plt.ylabel('Cost J($\\theta_0,\\theta_1$)')
#
#theta0_vector = np.linspace(-50, 50, 100).reshape((1,100))
#theta1_vector = np.linspace(-50, 50, 100).reshape((1,100))
#
#J_vals = np.zeros((theta0_vector.shape[1],theta1_vector.shape[1]))
#
#for i in range(theta0_vector.shape[1]):
#    for j in range(theta1_vector.shape[1]):
#        t = np.matrix([[theta0_vector[0,i]], [theta1_vector[0,j]]])
#        J_vals[i,j] = computeCost(X_norm, y, t)
#
#fig = plt.figure(num='Projection of calculated cost')
#ax = fig.gca(projection='3d')
#aX_train.set_xlabel('$\\theta_0$')
#aX_train.set_ylabel('$\\theta_1$')
#aX_train.set_zlabel('Cost ($\\theta_0,\\theta_1$)')
#
#theta0_vector, theta1_vector = np.meshgrid(theta0_vector, theta1_vector)
#
#surf = aX_train.plot_surface(theta0_vector, theta1_vector, J_vals, cmap=cm.jet, linewidth=0, antialiased=False)
#
#plt.figure(num='Contours of calculated cost')
#level_list = np.arange(15, 4000, 200)
#cs = plt.contour(theta0_vector, theta1_vector, J_vals,levels=level_list)
#plt.xlabel('$\\theta_0$')
#plt.ylabel('$\\theta_1$')

###############################################################################

###############################################################################
#Example 3 with better hypothesis with madeup feature
###############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))
#X = data[:,1:2]
#y = data[:,0:1]
#m = X_train.shape[0]    #Data set size
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
##Initialize theta to 0
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
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

###############################################################################
#Example 4 hypothesis with 2 features
###############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))
#X = np.c_[data[:, 1:2], data[:, 2:3]]
#y = data[:,0:1]
#m = X_train.shape[0]    #Data set size
#n = X_train.shape[1]    #Number of features
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.01
#num_iters = 3000
##Initialize theta to 0
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
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

###############################################################################
#Example 5 hypothesis with 2 features + better fit curve
###############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))
#X = np.c_[data[:, 1:2], data[:, 2:3]]
#y = data[:,0:1]
#m = X_train.shape[0]    #Data set size
#
##Fitting a curve to input data
#poly_degree = 4
#new_feature = np.c_[np.ones(m)]
#poly_coef = np.polyfit(X[:,0:1].flatten(), y, deg=poly_degree)
#
#degree = poly_degree
#for i in range(poly_degree + 1):
#    new_feature[:] += poly_coef[i,0]*(X[:,0:1]**degree)
#    degree -= 1
#
##Adding the new feature
#X = np.c_[X[:,0:1], new_feature[:,0:1] ,X[:,1:2]]
#n = X_train.shape[1]    #Number of features
#
##Normalize features
#X_norm, mu, sigma = featureNormalize(X, m, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#
#alpha = 0.1
#num_iters = 3000
#
##Initialize theta to 0
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
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')


