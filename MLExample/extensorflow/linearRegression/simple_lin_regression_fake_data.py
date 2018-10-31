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

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
#  #plt.ylim([0, 5])

input_file = './car_data.csv'
#data_from_file = pd.read_csv(input_file)

data = np.loadtxt(open(input_file, "rb"), dtype='float', delimiter=',', skiprows=1, usecols=(0,3))

train_data = data[0:288, 1:2]
test_data = data[288:,1:2]

train_labels = data[0:288,0:1]
test_labels = data[288:,0:1]

#
#plt.figure(num='1')
#plt.plot(train_data, train_labels, 'rx')
#plt.xlabel('Horsepower (hp)')
#plt.ylabel('Miles per Galon')
#
##Normalize
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data_norm = (train_data - mean) / std
test_data_norm = (test_data - mean) / std


#https://learningtensorflow.com/lesson7/
#https://www.guru99.com/linear-regression-tensorflow.html
#https://blog.altoros.com/using-linear-regression-in-tensorflow.html

#Getting the dataset
points=100
input_d=np.linspace(-3,3,points)
np.random.seed(6)
output=np.sin(input_d)+np.random.uniform(-0.5,0.5,points)

#plt.figure(num='2')
#p=plt.plot(input_d,output,'ro')
#plt.axis([-4,4,-2.0,2.0])
#plt.show()

epochs = 1000



x = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
tf.set_random_seed(5)
w = tf.Variable(tf.random_normal([1]), name="weights")
b = tf.Variable(tf.random_normal([1]), name='bias')


Y_Pred = tf.add(tf.multiply(x, w), b) # same as ϴ1*X + ϴ0


error=tf.square(Y_Pred-Y)
#we just reduced error by dividing the no of input_d values (Good practice)
f_error=tf.reduce_sum(error)/(points-1) 

optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(f_error)

    
init = tf.global_variables_initializer()
sess = tf.Session()
#sess = tf.InteractiveSession()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init)

# Fit all training data

#loss_expected=0
#for epoch in range(epochs):
#    
#    for (x_point,y_point) in zip(input_d,output):
#        sess.run(optimizer,{x:x_point,Y:y_point})
#    
#    loss_per_epoch = sess.run(f_error,{x:input_d,Y:output}) 
#    # for testing we give the same training set as the test set
#       
#    if epoch % 10 == 0:
#        plt.axis([-4,4,-2.0,2.0])
#        plt.plot(input_d,Y_Pred.eval(feed_dict={x: input_d}, session=sess),'b', alpha=epoch / epochs)
#
#
#    # Allow the training to quit if we've reached a minimum
#    if np.abs(loss_expected - loss_per_epoch) < 0.000001:
#        break
#    loss_expected = loss_per_epoch

for _ in range(1000):
    sess.run(optimizer, {x: input_d, Y: output})
    
print(sess.run(w))
print(sess.run(b))
    
sess.close()

#plt.figure(num='3')
#plt.scatter(output, Y_Pred)
#plt.xlabel('True Values [1000$]')
#plt.ylabel('Predictions [1000$]')
#plt.axis('equal')
#plt.xlim(plt.xlim())
#plt.ylim(plt.ylim())
#_ = plt.plot([-100, 100], [-100, 100])
        
#plt.show()

