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


##############################################################################
#Example 1 with big data set
##############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))
#X_train = data[0:300,1:2]
#X_test = data[300:,1:2]
#y_train = data[0:300,0:1]
#y_test = data[300:,0:1]
#m = X_train.shape[0]    # Training data set size
#m_t = X_test.shape[0]   # Test data set size
#n = X_train.shape[1]    # Number of features
#
##Plot input data
#plt.figure(num='Input data')
#plt.plot(X_train, y_train, 'rx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')
#
##Normalize features
#X_norm = featureNormalize(X_train, m, n)
#X_test_norm = featureNormalize(X_test, m_t, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#X_test_norm = np.c_[np.ones(m_t), X_test_norm]
#
#alpha = 0.01
#num_iters = 300
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)
#
##Estimate output from calculated thetas
#prediction =np.asmatrix(X_test_norm) * theta
#
##Plot the results
#plt.figure(num='Results 1')
#plt.plot(X_train[:, 0:], y_train, 'rx', X_test[:, 0:], prediction, 'bx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

###############################################################################
##Graphic representation of cost for given theta values
###############################################################################

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
#        J_vals[i,j] = computeCost(X_norm, y_train, t)
#
#fig = plt.figure(num='Projection of calculated cost')
#ax = fig.gca(projection='3d')
#ax.set_xlabel('$\\theta_0$')
#ax.set_ylabel('$\\theta_1$')
#ax.set_zlabel('Cost ($\\theta_0,\\theta_1$)')
#
#theta0_vector, theta1_vector = np.meshgrid(theta0_vector, theta1_vector)
#
#surf = ax.plot_surface(theta0_vector, theta1_vector, J_vals, cmap=cm.jet, linewidth=0, antialiased=False)
#
#plt.figure(num='Contours of calculated cost')
#level_list = np.arange(15, 4000, 200)
#cs = plt.contour(theta0_vector, theta1_vector, J_vals,levels=level_list)
#plt.xlabel('$\\theta_0$')
#plt.ylabel('$\\theta_1$')

###############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))
#
#X_in = np.c_[data[:,1:2], 1/data[:,1:2]]
#X_train = X_in[0:300, :]
#X_test = X_in[300:, :]
#y_train = data[0:300,0:1]
#y_test = data[300:,0:1]
#m = X_train.shape[0]    # Training data set size
#m_t = X_test.shape[0]   # Test data set size
#n = X_train.shape[1]    # Number of features
#
###Plot input data
#plt.figure(num='Input data')
#plt.plot(X_train[:, 0:1], y_train, 'rx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')
#
##Normalize features
#X_norm = featureNormalize(X_train, m, n)
#X_test_norm = featureNormalize(X_test, m_t, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#X_test_norm = np.c_[np.ones(m_t), X_test_norm]
#
#alpha = 0.01
#num_iters = 300
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)
#
##Estimate output from calculated thetas
##prediction =np.asmatrix(X_test_norm) * theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results 1')
##plt.plot(X_train[:, 0:1], y_train, 'rx', X_test[:, 0:1], prediction, 'bx')
#plt.plot(X_train[:, 0:1], y_train, 'rx', X_train[:, 0:1], prediction, 'bx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

###############################################################################
#Example 2 hypothesis with 2 features
###############################################################################

##Opening file with data
#input_file = 'car_data.csv'
#data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))
#X_train = np.c_[data[0:300, 1:2], 1/data[0:300, 2:3]]
#X_test = np.c_[data[300:, 1:2], data[300:, 2:3]]
#y_train = data[0:300,0:1]
#y_test = data[300:,0:1]
#m = X_train.shape[0]    # Training data set size
#m_t = X_test.shape[0]   # Test data set size
#n = X_train.shape[1]    # Number of features
#
###Plot input data
##plt.figure(num='Input data (horsepower)')
##plt.plot(X_train[:,0:1], y_train, 'rx')
##plt.xlabel('Horsepower (hp)')
##plt.ylabel('Miles per Galon')
##
##plt.figure(num='Input data (weigth)')
##plt.plot(X_train[:,1:2], y_train, 'rx')
##plt.xlabel('Weigth (Kg)')
##plt.ylabel('Miles per Galon')
#
##Normalize features
#X_norm = featureNormalize(X_train, m, n)
#X_test_norm = featureNormalize(X_test, m_t, n)
#
##Add an additional column with 1's
#X_norm = np.c_[np.ones(m), X_norm]
#X_test_norm = np.c_[np.ones(m_t), X_test_norm]
#
#alpha = 0.01
#num_iters = 3000
##Initialize theta to 0
#theta = np.zeros((n + 1, 1))
#
##Run gradient descent
#theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)
#
##Get predicted values from obtained theta
##prediction =np.asmatrix(X_test_norm) * theta
#prediction =np.asmatrix(X_norm) * theta
#
##Plot the results
#plt.figure(num='Results for exercise 2')
##plt.plot(X_train[:, 0:1], y_train, 'rx', X_test[:, 0:1], prediction, 'o')
#plt.plot(X_train[:, 0:1], y_train, 'rx', X_train[:, 0:1], prediction, 'o')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')

###############################################################################
#Example 3: hypothesis with 2 features + better fit curve
###############################################################################

#Opening file with data
input_file = 'car_data.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))
#X_in = np.c_[data[:, 1:2], 1/data[:, 1:2], 1/data[:, 2:3]]
X_in = np.c_[1/data[:, 1:2], 1/data[:, 2:3]]


y_in = data[:,0:1]

X_train = X_in[0:300,:]
X_test = X_in[300:,:]
y_train = data[0:300,0:1]
y_test = data[300:,0:1]
m = X_train.shape[0]    # Training data set size
m_t = X_test.shape[0]   # Test data set size
n = X_train.shape[1]    # Number of features

#Normalize features
X_norm = featureNormalize(X_train, m, n)
X_test_norm = featureNormalize(X_test, m_t, n)

#Add an additional column with 1's
X_norm = np.c_[np.ones(m), X_norm]
X_test_norm = np.c_[np.ones(m_t), X_test_norm]

alpha = 0.01
num_iters = 500

#Initialize theta to 0
theta = np.zeros((n + 1, 1))

#Run gradient descent
theta, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

#Get predicted values from obtained theta
#prediction =np.asmatrix(X_test_norm) * theta
prediction =np.asmatrix(X_norm) * theta

#Plot the results
plt.figure(num='Results for exercise 3')
#plt.plot(X_train[:, 3:4], y_train, 'rx', X_test[:, 3:4], prediction, 'o')
#plt.plot(X_train[:, 0:1], y_train, 'rx', X_train[:, 0:1], prediction, 'o')
plt.plot(data[0:300, 1:2], data[0:300, 0:1], 'rx', data[0:300, 1:2], prediction, 'o')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

#Plot the results
plt.figure(num='Results for exercise 3_')
plt.plot(data[0:300, 2:3], data[0:300, 0:1], 'rx', data[0:300, 2:3], prediction, 'o')
plt.xlabel('Weight (Kg)')
plt.ylabel('Miles per Galon')
#

