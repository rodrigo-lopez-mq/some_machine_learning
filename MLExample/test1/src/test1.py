# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

data_mat = sio.loadmat('dataX.mat')
X = data_mat['X']
data_mat = sio.loadmat('datay.mat')
y = data_mat['y']
data_mat = sio.loadmat('dataTheta1.mat')
Theta1 = data_mat['Theta1']
data_mat = sio.loadmat('dataTheta2.mat')
Theta2 = data_mat['Theta2']


m = X.shape[0]
sel = np.matrix(list(range(100)))
sel = sel.transpose()
#sel = np.random.permutation(m)
#sel = sel[0:100]
######################

num_labels = Theta2.shape[0]
size_th = Theta1.shape[0]

p = np.zeros((m,1)) 
X = np.c_[np.ones(m), X]

#plt.plot(sel)
#plt.ylabel('some numbers')
#plt.show()