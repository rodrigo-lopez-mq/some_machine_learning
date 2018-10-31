# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-z))

