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
data_from_file = pd.read_csv(input_file)

mpg_data = data_from_file['mpg']
#print (mpg_data.head())
hp_data = data_from_file['horsepower']

train_data = hp_data[0:288]
test_data = hp_data[288:]

test_data2 =  test_data.tolist()

train_labels = mpg_data[0:288]
test_labels = mpg_data[288:]

plt.figure(num='1')
plt.plot(train_data, train_labels, 'rx')
plt.xlabel('Horsepower (hp)')
plt.ylabel('Miles per Galon')

#Normalize
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data_norm = (train_data - mean) / std
test_data_norm = (test_data - mean) / std

#train_data_norm.head()

#Build model
model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(1,)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)])

optimizer = tf.train.RMSPropOptimizer(0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

model.summary()
#model.
#keras.utils.plot_model(model, to_file='model.png')
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data_norm, train_labels, epochs=500,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(test_data_norm, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data_norm).flatten()

plt.figure(num='2')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])

#error = test_predictions - test_labels
#plt.hist(error, bins = 50)
#plt.xlabel("Prediction Error [1000$]")
#_ = plt.ylabel("Count")

#https://learningtensorflow.com/lesson7/
#https://www.guru99.com/linear-regression-tensorflow.html

