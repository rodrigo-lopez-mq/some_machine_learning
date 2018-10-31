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

input_file = './car_data.csv'
data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))

train_data = data[0:288, 1:2]
test_data = data[288:,1:2]
train_labels = data[0:288,0:1]
test_labels = data[288:,0:1]

train_data = np.squeeze(train_data)
train_labels = np.squeeze(train_labels)
#
##Normalize
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data_norm = (train_data - mean) / std
test_data_norm = (test_data - mean) / std

#Getting the dataset
points=train_data.shape[0]

x = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
tf.set_random_seed(5)
w = tf.Variable(tf.random_normal([1]), name="weights")
b = tf.Variable(tf.random_normal([1]), name='bias')


Y_Pred = tf.add(tf.multiply(x, w), b) # same as ϴ1*X + ϴ0
#Y_Pred = (w * x) + b

error=tf.square(Y_Pred-Y)
f_error=tf.reduce_sum(error)/(points) 

optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(f_error)

    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(1000):
    sess.run(optimizer, {x: train_data_norm, Y: train_labels})
    
print(sess.run(w))
print(sess.run(b))

w_val = sess.run(w)
b_val = sess.run(b)
    
sess.close()


results1 = (train_data_norm * w_val) + b_val 

plt.figure(num='1')
plt.plot(train_data, train_labels, 'rx', train_data, results1,'o')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

results2 = (test_data_norm * w_val) + b_val 

plt.figure(num='1')
plt.plot(train_data, train_labels, 'rx', test_data, results2,'o')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')
