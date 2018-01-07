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

batch_size = 32
num_classes = 2
epochs = 200
data_augmentation = False
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cropweedCNN_trained_model.h5'



# pprint(y_test.tolist())

model = Sequential()
# model.add(Conv2D(20, (4, 4), padding='same',
#                  input_shape=x_train.shape[1:]))
model.add(Conv2D(3, (4, 4), padding='same',
                 input_shape=(200, 200, 3)))

# (7077, 200, 200, 3)

# model.add(Activation('relu'))
# model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
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
model.add(LeakyReLU(0.2))
model.add(AveragePooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
# model.add(Activation('relu'))
model.add(LeakyReLU(0.2))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# The data, shuffled and split between train and test sets:
x_train, y_train, x_test, y_test, x_val, y_val = crop_weed_data()
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# print(type(y_test))
# for i in range(500,y_test.shape[0]):
#   print(y_test[i])

# print('Actual Label = '+ str(y_test[3]))
num_predictions = y_val.shape[0]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_test /= 255
x_val /= 255

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])




print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'vaildation samples')

csvpath = os.path.join("output/cnn_crop_weed", "history.csv")
if os.path.exists(csvpath):
    print("Already exists: {}".format(csvpath))
    # return

csv_logger = keras.callbacks.CSVLogger('training.log')

print('Not using data augmentation.')
history=model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs, callbacks=[csv_logger],
    validation_data=(x_val, y_val),
    shuffle=True)

# save history to CSV
df = pd.DataFrame(history.history)
df.to_csv(csvpath)


'''
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
'''
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# # Load label names to use in prediction results
# label_list_path = 'datasets/cifar-10-batches-py/batches.meta'


keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
datadir_base = os.path.expanduser(keras_dir)
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
# label_list_path = os.path.join(datadir_base, label_list_path)

# with open(label_list_path, mode='rb') as f:
#     labels = pickle.load(f)

# # Evaluate model with test data set and share sample prediction results
# evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
#                                       batch_size=batch_size),
#                                       steps=x_test.shape[0] // batch_size)

# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Model Accuracy = %.2f' % (evaluation[1]))
print(x_test.shape)
# predict_gen = model.predict_generator(datagen.flow(x_test, y_test,
#                                       batch_size=batch_size),
#                                       steps=x_test.shape[0] // batch_size)
prediction = model.predict(x_test, batch_size=batch_size) 

# for i in range(500,y_test.shape[0]):
#   print(str(y_test[i]))
# i=1
for predict_index, predicted_y in enumerate(prediction):
    # actual_label = labels['label_names'][np.argmax(y_test[predict_index])]
    # predicted_label = labels['label_names'][np.argmax(predicted_y)]
    # actual_label = np.argmax(y_test[predict_index])
    # predicted_label = np.argmax(predicted_y)
    # print('Actual Label = %d vs. Predicted Label = %d' % (actual_label,
    #                                                       predicted_label))
    # print(y_test[i])
    # i=+1
    print(str(predict_index)+ 'Actual Label = '+ str(y_test[predict_index]) + 'vs. Predicted Label = ' +
                                                           str(predicted_y))
    if predict_index == num_predictions:
      break

