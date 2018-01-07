import numpy as np
import pickle
import crop_weed



def cifar10_process(x):
    # x = x.astype(np.float32) / 255.0
    return x

def crop_weed_data():
	(x_train, y_train), (x_test, y_test), (x_val, y_val)= crop_weed.load_data()
	return cifar10_process(x_train), y_train, cifar10_process(x_test), y_test, cifar10_process(x_val), y_val
