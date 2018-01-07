''' 
Create crop-weed image dataset as ndarray and save it using pickle
Optionally do data augmentation with the keras image processing

To read the pickle file:
import pickle

with open('my_cropsweeds_dataset.pickle', 'rb') as data:
    (x_train, y_train), (x_test, y_test) = pickle.load(data)

Author: Ananya, 07/26/17
'''

import PIL
import pickle
from PIL import Image, ImageChops
from numpy import array
from glob import glob
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator

image_size = 200
data_augmentation = False
instance_here1 = 0
instance_here2 = 0


def dataset_maker(label_key, glob_file_key, loc_train_labels=""):
	if label_key ==1:
		# output = open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsWeeds.pickle', 'wb') 
		dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
		labels_created = np.empty([0]) # create empty 
		data_size = 0
		# instance_here1 = instance_here1 + 1
		print ('Are we in the loop ?')
		# Converting image to ndarray

		for filename in enumerate(glob(glob_file_key)):
			# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
			# print ('Are we in the loop ?')
			img = Image.open(filename[1]) # load the image file
			img = img.resize([image_size, image_size], resample = PIL.Image.LANCZOS)
			# img.show()
			arr = array(img)			  # convert PIL image to array
			# print(arr.shape)

			dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
			labels_created = np.append (labels_created, 1)
		
		# combined_dataset = ([dataset, labels_created])
		# print(combined_dataset[0].shape)
		# print('Dumping last bit of label1 data')
		
		# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/my_cropsweeds_dataset_size200_greenOnly_cropsWeeds.pickle', 'ab') as output:
		# 	pickle.dump(combined_dataset, output, protocol=pickle.HIGHEST_PROTOCOL)
		# output.close()

	elif label_key ==0:
		# output2 = open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsOnly.pickle', 'wb') 

		dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
		labels_created = np.empty([0]) # create empty 
		data_size = 0
	
		# Converting image to ndarray

		for filename in enumerate(glob(glob_file_key)):
			# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
			print ('Are we in the loop ?')
			img = Image.open(filename[1]) # load the image file
			img = img.resize([image_size, image_size], resample = PIL.Image.LANCZOS)
			# img.show()
			arr = array(img)			  # convert PIL image to array
			# print(arr.shape)

			dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
			labels_created = np.append (labels_created, 0)
		


		# combined_dataset = ([dataset, labels_created])
		# print('Dumping last bit of label0 data')
		# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/my_cropsweeds_dataset_size200_greenOnly_cropsOnly.pickle', 'ab') as output:
		# 	pickle.dump(combined_dataset, output2, protocol=pickle.HIGHEST_PROTOCOL)
		# output.close()
	# if len(loc_train_labels) > 0:
	# 	labels = pd.read_csv(loc_train_labels)
	# 	return np.array(dataset), np.array(labels["Class"])
	# else:
	# 	print('datasize', data_size)
	# 	return np.array(dataset)
	return dataset, labels_created


def dataset_maker_aug(label_key, glob_file_key, loc_train_labels=""):
	if label_key ==1:
		j =1
		# output = open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsWeeds.pickle', 'wb') 
		dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
		labels_created = np.empty([0]) # create empty 
		data_size = 0
		# instance_here1 = instance_here1 + 1
		print ('Are we in the loop ?')
		# Converting image to ndarray

		for filename in enumerate(glob(glob_file_key)):
			# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
			# print ('Are we in the loop ?')
			img = Image.open(filename[1]) # load the image file
			img = img.resize([image_size, image_size])
			# img.show()
			arr = array(img)			  # convert PIL image to array
			# print(arr.shape)

			dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
			labels_created = np.append (labels_created, 1)
		
			if dataset.shape[0] >100: 
				data_size = data_size +dataset.shape[0]
				combined_dataset = ([dataset, labels_created])
				print('Dumping label1 data')
				print(j)
				j =j+1
				print(combined_dataset[0].shape)
				with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsWeeds.pickle', 'ab') as output:
					pickle.dump(combined_dataset, output, protocol=pickle.HIGHEST_PROTOCOL)
				output.close()
				dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
				labels_created = np.empty([0]) # create empty 

		combined_dataset = ([dataset, labels_created])
		print(combined_dataset[0].shape)
		print('Dumping last bit of label1 data')
		
		with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsWeeds.pickle', 'ab') as output:
			pickle.dump(combined_dataset, output, protocol=pickle.HIGHEST_PROTOCOL)
		output.close()

	elif label_key ==0:
		# output2 = open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsOnly.pickle', 'wb') 

		dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
		labels_created = np.empty([0]) # create empty 
		data_size = 0
	
		# Converting image to ndarray

		for filename in enumerate(glob(glob_file_key)):
			# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
			print ('Are we in the loop ?')
			img = Image.open(filename[1]) # load the image file
			img = img.resize([image_size, image_size])
			# img.show()
			arr = array(img)			  # convert PIL image to array
			# print(arr.shape)

			dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
			labels_created = np.append (labels_created, 0)
		
			# if dataset.shape[0] >100: 
			# data_size = data_size +dataset.shape[0]
				# combined_dataset = ([dataset, labels_created])
				# print('Dumping label0 data')
				# # with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsOnly.pickle', 'wb') as output:
				# pickle.dump(combined_dataset, output2, protocol=pickle.HIGHEST_PROTOCOL)
				# dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
				# labels_created = np.empty([0]) # create empty 

		# combined_dataset = ([dataset, labels_created])
		# print('Dumping last bit of label0 data')
		# # with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsOnly.pickle', 'wb') as output:
		# pickle.dump(combined_dataset, output2, protocol=pickle.HIGHEST_PROTOCOL)

	# if len(loc_train_labels) > 0:
	# 	labels = pd.read_csv(loc_train_labels)
	# 	return np.array(dataset), np.array(labels["Class"])
	# else:
	# 	print('datasize', data_size)
	# 	return np.array(dataset)
	return dataset, labels_created

labels =None
labels2=None

if data_augmentation == False:
	# Reading in the images and labels
	# images, labels= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.jpg","/data_2/Ananya_files_2/Separated Photos_200by200/crops_weeds_label.csv")
	images, labels= dataset_maker(1, "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/Crops_andWeeds_cropped/*.jpg")
	# # for *JPG ending images
	imagesCAPS, labelsCAPS= dataset_maker(1, "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/Crops_andWeeds_cropped/*.JPG")
	# combined_datasetCW = ([images, labels])



	# # Create empty list to store all x and y values
	# x_values_total = np.empty([0, image_size, image_size, 3]) 
	# y_values_total = np.empty([0])
	# i=1
	# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/my_cropsweeds_dataset_size200_cropsWeeds.pickle', 'rb') as data:
	#     # ((x_train, y_train), (x_test, y_test), (x_val, y_val)) = pickle.load(data)
	# 	while True:
	# 		try:
	# 			print('loading data')
	# 			print(i)
	# 			i=i+1
	# 			(x_values, y_values) = pickle.load(data)
	# 			print(x_values.shape)
	# 			# print(x_values_total.shape)
	# 		except EOFError:
	# 			print('End of file')
	# 			break
	# 		else:
	# 			for x in x_values:
	# 				x_values_total = np.append(x_values_total, [x], axis=0) 
	# 			for y in y_values:
	# 				y_values_total = np.append(y_values_total, [y])

	# no_of_images = len(x_values_total)
	# print(no_of_images)
	# no_of_labels = len(y_values_total)
	# print(no_of_labels)
	# ## test 1 commented

	# # for *JPG ending images
	# imagesCAPS, labelsCAPS= dataset_maker(1, "/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.JPG")
	# Reading in the second folder of images
		# Reading in the images and labels

	images2, labels2= dataset_maker(0, "/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/Crops_only_cropped/*.jpg")
	# for *JPG ending images
	images2CAPS, labels2CAPS = dataset_maker(0, "extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/Crops_only_cropped/*.JPG")

	print(images2CAPS.shape)

	imagesCW = np.append(images, imagesCAPS, axis = 0)
	labelsCW = np.append(labels, labelsCAPS, axis = 0)
	imagesC = np.append(images2, images2CAPS, axis = 0)
	labelsC = np.append(labels2, labels2CAPS, axis = 0)

	combined_datasetCW = ([imagesCW, labelsCW])
	combined_datasetC = ([imagesC, labelsC])


## Saving to pickle file (label1)


## test 1 commented
with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/my_cropsweeds_dataset_size200_greenOnly_cropsWeeds.pickle', 'wb') as output1:
	pickle.dump(combined_datasetCW, output1, protocol=pickle.HIGHEST_PROTOCOL)

with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Masked Photos Green/my_cropsweeds_dataset_size200_greenOnly_cropsOnly.pickle', 'wb') as output2:
	pickle.dump(combined_datasetC, output2, protocol=pickle.HIGHEST_PROTOCOL)

## test 1 commented---end



# else: 
# 	print('I am here')

# 	##------------dividing into train, test and val directly
# 	# no_of_images, w, h, channels = images.shape
# 	# print(no_of_images)
	
# 	# #Label= 1
# 	# # Seperate into training and testing datasets
# 	# x_train = images[:int(no_of_images/2), :, :, :]
# 	# y_train = np.ones(int(no_of_images/2))
# 	# x_test  = images[int(no_of_images/2):, :, :, :]
# 	# y_test= np.ones(no_of_images- int(no_of_images/2))

# 	# combined_dataset1 = ([x_train, y_train], [x_test, y_test])

# 	# no_of_images, w, h, channels = imagesCAPS.shape
# 	# print(no_of_images)
# 	# # Seperate into training and testing datasets
# 	# x_train = np.append(x_train, imagesCAPS[:int(no_of_images/2), :, :, :],axis=0)
# 	# y_train = np.append(y_train, np.ones(int(no_of_images/2)),axis=0)
# 	# x_val = imagesCAPS[int(no_of_images/2):, :, :, :]
# 	# y_val = np.ones(no_of_images- int(no_of_images/2))

# 	# #Label =0
# 	# no_of_images, w, h, channels = images2.shape
# 	# print(no_of_images)
# 	# # Seperate into training and testing datasets
# 	# x_train = np.append(x_train, images2[:int(no_of_images/2), :, :, :],axis=0)
# 	# y_train = np.append(y_train, np.zeros(int(no_of_images/2)),axis=0)
# 	# x_test = np.append(x_test,images2[int(no_of_images/2):, :, :, :],axis=0)
# 	# y_test = np.append(y_test, np.zeros(no_of_images- int(no_of_images/2)),axis=0)

# 	# # print(images.shape)
# 	# no_of_images, w, h, channels = images2CAPS.shape
# 	# print(no_of_images)
# 	# # Seperate into training and testing datasets
# 	# x_train = np.append(x_train, images2CAPS[:int(no_of_images/2), :, :, :],axis=0)
# 	# y_train = np.append(y_train, np.zeros(int(no_of_images/2)),axis=0)
# 	# x_val = np.append(x_val,images2CAPS[int(no_of_images/2):, :, :, :],axis=0)
# 	# y_val = np.append(y_val,np.zeros(no_of_images- int(no_of_images/2)),axis=0)
# 	##-------------------------------------

# 	##------ Separating into two classes and then dividing into train, val and test images depending on the desired ratio

# 	## Label =1
# 	no_of_images, w, h, channels = images.shape
# 	print(no_of_images)
# 	x_1 = images
# 	y_1 = np.ones(int(no_of_images))

# 	no_of_images, w, h, channels = imagesCAPS.shape
# 	print(no_of_images)
# 	x_1   = np.append(x_1, imagesCAPS[:, :, :, :],axis=0)
# 	y_1 = np.append(y_1, np.ones(int(no_of_images)),axis=0)

# 	#Label =0
# 	no_of_images, w, h, channels = images2.shape
# 	print(no_of_images)
# 	x_0 = images2
# 	y_0 = np.zeros(int(no_of_images))

# 	no_of_images, w, h, channels = images2CAPS.shape
# 	print(no_of_images)
# 	x_0  = np.append(x_0 , images2CAPS[:, :, :, :],axis=0)
# 	y_0  = np.append(y_0 , np.zeros(int(no_of_images)),axis=0)

# 	x_train_size = int(x_1.shape[0] *0.8)
# 	x_val_size = int(x_1.shape[0] *0.9)

# 	x_train1, x_val1, x_test1 = np.split(x_1, [x_train_size, x_val_size])
# 	y_train1, y_val1, y_test1 = np.split(y_1, [x_train_size, x_val_size])

# 	x_train_size = int(x_0.shape[0] *0.8)
# 	x_val_size = int(x_0.shape[0] *0.9)

# 	x_train0, x_val0, x_test0 = np.split(x_0, [x_train_size, x_val_size])
# 	y_train0, y_val0, y_test0 = np.split(y_0, [x_train_size, x_val_size])

# 	x_train = np.append(x_train0, x_train1, axis=0)
# 	x_val = np.append(x_val0, x_val1,axis=0)
# 	x_test = np.append(x_test0, x_test1,axis=0)
# 	y_train = np.append(y_train0, y_train1,axis=0)
# 	y_val = np.append(y_val0, y_val1,axis=0)
# 	y_test = np.append(y_test0, y_test1,axis=0)

# 	print('Labelled1')
# 	print(x_train1.shape)
# 	print(x_test1.shape)
# 	print(y_train1.shape)
# 	print(y_test1.shape)
# 	print(x_val1.shape)
# 	print(y_val1.shape)

# 	print('Labelled0')
# 	print(x_train0.shape)
# 	print(x_test0.shape)
# 	print(y_train0.shape)
# 	print(y_test0.shape)
# 	print(x_val0.shape)
# 	print(y_val0.shape)



# 	print('Combined')
# 	print(x_train.shape)
# 	print(x_test.shape)
# 	print(y_train.shape)
# 	print(y_test.shape)
# 	print(x_val.shape)
# 	print(y_val.shape)

# 	# Combine the image data and the corresponding labels
# 	# combined_training_dataset = [x_train, y_train]
# 	# combined_test_dataset = [x_test, y_test] 

# 	combined_dataset = ([x_train, y_train], [x_test, y_test], [x_val, y_val])

# # Seperate into training and testing datasets
# x_train = images[:int(no_of_images/2), :, :, :]
# y_train = np.zeros(len(x_train))
# x_test  = images[int(no_of_images/2):, :, :, :]
# y_test= np.zeros(len(x_test ))

# combined_dataset = ([x_train, y_train], [x_test, y_test])

# Save the image data(array) and labels in a pickle file
# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/my_cropsweeds_dataset_size200_part_aug.pickle', 'wb') as output:
# 	pickle.dump(combined_dataset1, output, protocol=pickle.HIGHEST_PROTOCOL)

# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/my_cropsweeds_dataset_size200_all_801010.pickle', 'wb') as output:
# 	pickle.dump(combined_dataset, output, protocol=pickle.HIGHEST_PROTOCOL)



# with open('/data_2/Ananya_files_2/my_cropsweeds_dataset_size200_all.txt', 'w') as f:
#   json.dump(combined_dataset, f)


