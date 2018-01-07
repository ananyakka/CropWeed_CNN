from __future__ import print_function
import os
import pickle
import numpy as np
from crop_weed_utils import crop_weed_data

# The data, shuffled and split between train and test sets:
x_train, y_train, x_test, y_test, x_val, y_val = crop_weed_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_train.shape[0])
print(y_test.shape[0])



print(y_train)