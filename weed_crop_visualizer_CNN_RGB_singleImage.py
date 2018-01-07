''' 
A Convolutional Network implementation example using TensorFlow library.
The image database used contains pictures of crops+weeds and only crops. 
Author: Ananya
Based on :

'''
from __future__ import print_function
import keras
from crop_weed_utils import crop_weed_data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
import h5py as h5py
import pandas as pd
import keras.backend as K
from pprint import pprint
import os
import pickle
import numpy as np
import PIL
import pickle
from PIL import Image, ImageChops
from numpy import array
from glob import glob
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_size = 200
batch_size = 32
num_classes = 2
epochs = 200
# plant = Image.open('IMG_0028.JPG') # load the image file
plant = Image.open('IMG_0236.JPG') # load the image file

# plant = plant.resize([image_size, image_size])
# plant.show()
plant_arr = array(plant, np.uint8)
print(plant_arr.shape)


model = Sequential()
# model.add(Conv2D(3, (4, 4), padding='same',
#                  input_shape=(200, 200, 3)))
model.add(Conv2D(3, (3, 3), padding='same',
                 input_shape=plant_arr.shape))
###
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(5,5)))
# model.add(Conv2D(3, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(5,5)))

# # model.add(Conv2D(3, (4, 4), padding='same',
# #                  input_shape=plant_arr.shape))

# # (7077, 200, 200, 3)

# # model.add(Activation('relu'))
# # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))


# model.add(LeakyReLU(0.2))
# # # model.add(Conv2D(15, (4, 4)))
# model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))

# # model.add(Activation('relu'))
# # # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
# model.add(LeakyReLU(0.2))
# model.add(AveragePooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))

# # # model.add(Conv2D(10, (4, 4), padding='same'))
# model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))
# # # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
# # # model.add(Activation('relu'))
# model.add(LeakyReLU(0.2))
# # # model.add(Conv2D(10, (4, 4)))
# model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))
# # # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
# # # model.add(Activation('relu'))
# model.add(LeakyReLU(0.2))
###
model.add(LeakyReLU(0.2))
# model.add(Conv2D(15, (4, 4)))
model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))

# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

# model.add(Conv2D(10, (4, 4), padding='same'))
model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
# model.add(Activation('relu'))
model.add(LeakyReLU(0.2))
# model.add(Conv2D(10, (4, 4)))
model.add(Conv2D(3, (4, 4), padding='same', kernel_regularizer=regularizers.l2(0.01)))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
# model.add(Activation('relu'))
# model.add(LeakyReLU(0.2))
# model.add(AveragePooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.25))
## till flatten


plant_batch = np.expand_dims(plant, axis=0)

conv_plant = model.predict(plant_batch)

def visualize_plant(plant_batch):
    plant_out = np.squeeze(plant_batch, axis=0)
    plant_out = np.uint8(plant_out)
    print(plant_out.shape)
    plant_filter = Image.fromarray(plant_out)
    plant_filter.show()

def nice_plant_printer(model, plant):
    '''prints the cat as a 2d array'''
    plant_batch = np.expand_dims(plant,axis=0)
    conv_plant2 = model.predict(plant_batch)

    conv_plant2 = np.squeeze(conv_plant2, axis=0)
    conv_plant2 = np.uint8(conv_plant2)
    print (conv_plant2.shape)
    conv_plant2 = conv_plant2.reshape(conv_plant2.shape[:2])

    print (conv_plant2.shape)
    plant_filter = Image.fromarray(conv_plant2)
    plant_filter.show()
# visualize_plant(plant_batch)