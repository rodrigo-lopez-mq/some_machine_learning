# -*- coding: utf-8 -*-
"""
Example 1 for linear regression
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python import debug as tf_debug


#################################################################

#log_file = "C:\Users\r\AppData\Local\Temp"

input_file = './car_data.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3,4))

train_data = data[0:288, 1:3]
test_data = data[288:,1:3]
train_labels = data[0:288,0:1]
test_labels = data[288:,0:1]

##Normalize
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data_norm = (train_data - mean) / std
test_data_norm = (test_data - mean) / std

m=train_data.shape[0]
n = train_data_norm.shape[1]

X_in = tf.placeholder(tf.float32, [None, n], "X_in")
w = tf.Variable(tf.random_normal([n, 1]), name="w")
b = tf.Variable(tf.constant(0.1, shape=[]), name="b")
h = tf.add(tf.matmul(X_in, w), b)

y_in = tf.placeholder(tf.float32, [None, 1], "y_in")
loss_op = tf.reduce_mean(tf.square(tf.subtract(y_in, h)), name="loss")
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss_op)
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(10000):
    sess.run(train_op, feed_dict={X_in: train_data_norm, y_in: train_labels})
    
w_computed = sess.run(w)
b_computed = sess.run(b)
    
sess.close()

results1 = (train_data_norm[:,0] * w_computed[0]) + (train_data_norm[:,1] * w_computed[1]) + b_computed

plt.figure(num='1')
plt.plot(train_data[:,0],  train_labels[:,0], 'rx', train_data[:,0], results1,'o')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

#etrror = np.absolute(train_labels[:,0] - results1)
#
#plt.figure(num='2')
#plt.plot(etrror, 'rx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')
