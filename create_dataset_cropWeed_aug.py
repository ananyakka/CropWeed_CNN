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
# import hickle
from PIL import Image, ImageChops
from numpy import array
from glob import glob
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
import re

image_size = 200
data_augmentation = True


def dataset_maker(glob_file_key, loc_train_labels=""):
	dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images

	# Converting image to ndarray
	#input_image = input("Enter the name of the image file:") 
	#for filename in enumerate(sorted(glob(glob_file_key),key=len) ):
	for filename in enumerate(glob(glob_file_key)):
		# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
		# print ('Are we in the loop ?')s
		img = Image.open(filename[1]) # load the image file
		img = img.resize([image_size, image_size])
		# img.show()
		arr = array(img)			  # convert PIL image to array
		# print(arr.shape)

		#arr = arr.transpose(2, 0, 1)
		#dataset = np.concatenate([arr[np.newaxis]])

		dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
		# dataset_trans = dataset.transpose(0, 3, 1, 2)

		# if dataset[0, :, :, 1] == dataset_trans[0, 1, :, :]: return true
		#print(dataset[0, :, :, 1] == dataset_trans[0, 1, :, :])
		# dataset2 = dataset.reshape(dataset.shape[0], dataset.shape[3], dataset.shape[1], dataset.shape[2])
		#dataset = np.array(dataset)
		#print(type(arr)) #<class 'numpy.ndarray'>
	#print(type(arr))
	# print(dataset_trans.shape)
	# Read the label from .csv file and store it
	if len(loc_train_labels) > 0:
		labels = pd.read_csv(loc_train_labels)
		return np.array(dataset), np.array(labels["Class"])
	else:

		return np.array(dataset)

def dataset_maker_aug(glob_file_key, loc_train_labels=""):
	dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
	# datagen = ImageDataGenerator(
 #    featurewise_center=False,
 #    featurewise_std_normalization=False,
 #    rotation_range=180,
 #    width_shift_range=0.2,
 #    height_shift_range=0.2,
 #    horizontal_flip=False,
 #    vertical_flip = False)

	print('Doing data augmentation!')
	
	max_angle = 180
	incr_angle = (max_angle-0)/100
	# print(incr_angle)
	max_offset=20
	x_incr = 5
	y_incr = 5
	# step =10
	# Converting image to ndarray
	#input_image = input("Enter the name of the image file:") 
	#for filename in enumerate(sorted(glob(glob_file_key),key=len) ):
	for filename in enumerate(glob(glob_file_key)):
		# print (filename) # (0, '/home/ananya/Weeding Bot Project/Data/sample_pepper_images/peppers1.jpg')
		# print ('Are we in the loop ?')s
		img = Image.open(filename[1]) # load the image file
		img = img.resize([image_size, image_size])
		# img.show()
		# arr = array(img)			  # convert PIL image to array
		# print(arr.shape)

		#arr = arr.transpose(2, 0, 1)
		#dataset = np.concatenate([arr[np.newaxis]])

		# dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set
		angles = np.arange(0, max_angle, incr_angle)
		for ang in angles:
			# x_offset = 10
			# y_offset = 10
			print('Number of augmented images:')
			print(dataset.shape[0])
			print('Angle:')
			print(ang)

			for x_offset in range(0, int(image_size/2), x_incr):

				for y_offset in range(0, int(image_size/2), y_incr):
					img2 = ImageChops.offset(img, x_offset, y_offset)
					img2.paste((0, 0, 0), (0, 0, image_size, y_offset))
					img2.paste((0, 0, 0), (0, 0, x_offset, image_size))
					img3 = ImageChops.offset(img, -x_offset, -y_offset)
					img3.paste((0, 0, 0), (0, image_size-y_offset, image_size, image_size))
					img3.paste((0, 0, 0), (image_size-x_offset, 0, image_size, image_size))
					# img3.show()
					# if dataset.shape[0] ==50:
					# 	return
					# print('angle augmentation')
					img4= img2.rotate(ang)
					arr=array(img4)
					dataset= np.append(dataset, [arr], axis=0)
					img5= img3.rotate(ang)
					arr=array(img5)
					dataset= np.append(dataset, [arr], axis=0)

					# Cheacking if rotation is good
					if dataset.shape[0]>20000 and dataset.shape[0] <20500:
						#Saving images to directory to check correct augmentation						
						a = re.split('[, /.]', filename[1])
						# print(a[-2])
						img4.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Augmented Photos/'+a[-2]+str(dataset.shape[0]), "JPEG")
						img5.save('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Augmented Photos/'+'neg'+a[-2]+str(dataset.shape[0]), "JPEG")
			print('x_offset:')
			print(x_offset)

					# print(dataset.shape[0])
				# img3.paste((0, 0, 0), (0, 0, x_offset, image_size))
				# arr=array(img3)
				# dataset= np.append(dataset, [arr], axis=0)
								
		# dataset_trans = dataset.transpose(0, 3, 1, 2)

		# if dataset[0, :, :, 1] == dataset_trans[0, 1, :, :]: return true
		#print(dataset[0, :, :, 1] == dataset_trans[0, 1, :, :])
		# dataset2 = dataset.reshape(dataset.shape[0], dataset.shape[3], dataset.shape[1], dataset.shape[2])
		#dataset = np.array(dataset)
		#print(type(arr)) #<class 'numpy.ndarray'>
	#print(type(arr))
	# print(dataset_trans.shape)
	# Read the label from .csv file and store it
	if len(loc_train_labels) > 0:
		labels = pd.read_csv(loc_train_labels)
		return np.array(dataset), np.array(labels["Class"])
	else:

		return np.array(dataset)
# Reading in the images and labels
# images, y = dataset_maker("/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/Crops_andWeeds_cropped/*.cleajpg","/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/crops_weeds_label.csv")
# file_path = input(" Enter the path to the first directory containing the images:")
# images = dataset_maker(os.path.join(file_path, "*.jpg"))

# images = dataset_maker('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/All_photos/*')

labels =None
labels2=None

if data_augmentation == False:
	# Reading in the images and labels
	# images, labels= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.jpg","/data_2/Ananya_files_2/Separated Photos_200by200/crops_weeds_label.csv")
	images= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.jpg")

	# for *JPG ending images
	imagesCAPS= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.JPG")
	# Reading in the second folder of images
		# Reading in the images and labels

	images2= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_only_cropped/*.jpg")
	# for *JPG ending images
	images2CAPS = dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_only_cropped/*.JPG")

	print(images.shape)
else:
	# Reading in the images and labels
	# images, labels= dataset_maker("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.jpg","/data_2/Ananya_files_2/Separated Photos_200by200/crops_weeds_label.csv")
	images= dataset_maker_aug("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.jpg")

	# for *JPG ending images
	imagesCAPS= dataset_maker_aug("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_andWeeds_cropped/*.JPG")
	# Reading in the second folder of images
		# Reading in the images and labels

	images2= dataset_maker_aug("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_only_cropped/*.jpg")
	# for *JPG ending images
	images2CAPS = dataset_maker_aug("/data_2/Ananya_files_2/Separated Photos_200by200/Crops_only_cropped/*.JPG")

	print(images.shape)


if labels is not None and labels2 is not None:

	#### following section needs correction. Labels aren't proper; data hasn't been seperated into validation and test-----------------
	# no_of_images, w, h, channels = images.shape
	no_of_images, w, h, channels= images.shape
	print(no_of_images)
	# Seperate into training and testing datasets
	x_train = images[:int(no_of_images/2), :, :, :]
	n = x_train.shape[0]
	y_train = labels[:x_train.shape[0]]
	# x_test  = images[int(no_of_images/2):, :, :, :]
	x_test  = images[int(no_of_images/2):, :, :, :]
	m=x_test.shape[0]
	y_test= labels[:x_test.shape[0]]

	no_of_images, w, h, channels= imagesCAPS.shape
	print(no_of_images)
	# Seperate into training and testing datasets
	x_train = np.append(x_train, imagesCAPS[:int(no_of_images/2), :, :, :], axis=0)
	y_train = np.append(y_train, labelsCAPS[n:x_train.shape[0]], axis=0)
	# x_test  = images[int(no_of_images/2):, :, :, :]
	x_test = np.append(x_test,images1CAPS[int(no_of_images/2):, :, :, :],axis=0)
	y_test = np.append(y_test,labels1CAPS[m:x_train.shape[0]],axis=0)

	# print(images.shape)
	no_of_images, w, h, channels= images2.shape
	print(no_of_images)
	# Seperate into training and testing datasets
	x_train = np.append(x_train, images2[:int(no_of_images/2), :, :, :],axis=0)
	y_train = np.append(y_train, labels2[x_train.shape[0]],axis=0)
	x_test = np.append(x_test,images2[int(no_of_images/2):, :, :, :],axis=0)
	y_test = np.append(y_test,labels2[x_train.shape[0]],axis=0)

	no_of_images, w, h, channels= images2CAPS.shape
	print(no_of_images)
	# Seperate into training and testing datasets
	x_train = np.append(x_train, images2CAPS[:int(no_of_images/2), :, :, :],axis=0)
	y_train = np.append(y_train, labels2CAPS[x_train.shape[0]],axis=0)
	x_test = np.append(x_test,images2CAPS[int(no_of_images/2):, :, :, :],axis=0)
	y_test = np.append(y_test,labels2CAPS[x_train.shape[0]],axis=0)


	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)
	
	# Combine the image data and the corresponding labels
	# combined_training_dataset = [x_train, y_train]
	# combined_test_dataset = [x_test, y_test] 

	
	combined_dataset = ([x_train, y_train], [x_test, y_test])
	###--------------------------------------------------------------------------------
else: 
	print('I am here')

	# ##------------dividing into train, test and val directly
	# no_of_images, w, h, channels = images.shape
	# print(no_of_images)
	
	# #Label= 1
	# # Seperate into training and testing datasets
	# x_train = images[:int(no_of_images/2), :, :, :]
	# y_train = np.ones(int(no_of_images/2))
	# x_test  = images[int(no_of_images/2):, :, :, :]
	# y_test= np.ones(no_of_images- int(no_of_images/2))

	# combined_dataset1 = ([x_train, y_train], [x_test, y_test])

	# no_of_images, w, h, channels = imagesCAPS.shape
	# print(no_of_images)
	# # Seperate into training and testing datasets
	# x_train = np.append(x_train, imagesCAPS[:int(no_of_images/2), :, :, :],axis=0)
	# y_train = np.append(y_train, np.ones(int(no_of_images/2)),axis=0)
	# x_val = imagesCAPS[int(no_of_images/2):, :, :, :]
	# y_val = np.ones(no_of_images- int(no_of_images/2))

	# #Label =0
	# no_of_images, w, h, channels = images2.shape
	# print(no_of_images)
	# # Seperate into training and testing datasets
	# x_train = np.append(x_train, images2[:int(no_of_images/2), :, :, :],axis=0)
	# y_train = np.append(y_train, np.zeros(int(no_of_images/2)),axis=0)
	# x_test = np.append(x_test,images2[int(no_of_images/2):, :, :, :],axis=0)
	# y_test = np.append(y_test, np.zeros(no_of_images- int(no_of_images/2)),axis=0)

	# # print(images.shape)
	# no_of_images, w, h, channels = images2CAPS.shape
	# print(no_of_images)
	# # Seperate into training and testing datasets
	# x_train = np.append(x_train, images2CAPS[:int(no_of_images/2), :, :, :],axis=0)
	# y_train = np.append(y_train, np.zeros(int(no_of_images/2)),axis=0)
	# x_val = np.append(x_val,images2CAPS[int(no_of_images/2):, :, :, :],axis=0)
	# y_val = np.append(y_val,np.zeros(no_of_images- int(no_of_images/2)),axis=0)
	# ##-------------------------------------
	##------ Separating into two classes and then dividing into train, val and test images depending on the desired ratio

	## Label =1
	no_of_images, w, h, channels = images.shape
	print(no_of_images)
	x_1 = images
	y_1 = np.ones(int(no_of_images))

	no_of_images, w, h, channels = imagesCAPS.shape
	print(no_of_images)
	x_1   = np.append(x_1, imagesCAPS[:, :, :, :],axis=0)
	y_1 = np.append(y_1, np.ones(int(no_of_images)),axis=0)

	#Label =0
	no_of_images, w, h, channels = images2.shape
	print(no_of_images)
	x_0 = images2
	y_0 = np.zeros(int(no_of_images))

	no_of_images, w, h, channels = images2CAPS.shape
	print(no_of_images)
	x_0  = np.append(x_0 , images2CAPS[:, :, :, :],axis=0)
	y_0  = np.append(y_0 , np.zeros(int(no_of_images)),axis=0)

	x_train_size = int(x_1.shape[0] *0.6)
	x_val_size = int(x_1.shape[0] *0.8)

	x_train1, x_val1, x_test1 = np.split(x_1, [x_train_size, x_val_size])
	y_train1, y_val1, y_test1 = np.split(y_1, [x_train_size, x_val_size])

	x_train_size = int(x_0.shape[0] *0.6)
	x_val_size = int(x_0.shape[0] *0.8)

	x_train0, x_val0, x_test0 = np.split(x_0, [x_train_size, x_val_size])
	y_train0, y_val0, y_test0 = np.split(y_0, [x_train_size, x_val_size])

	x_train = np.append(x_train0, x_train1, axis=0)
	x_val = np.append(x_val0, x_val1,axis=0)
	x_test = np.append(x_test0, x_test1,axis=0)
	y_train = np.append(y_train0, y_train1,axis=0)
	y_val = np.append(y_val0, y_val1,axis=0)
	y_test = np.append(y_test0, y_test1,axis=0)

	print('Labelled1')
	print(x_train1.shape)
	print(x_test1.shape)
	print(y_train1.shape)
	print(y_test1.shape)
	print(x_val1.shape)
	print(y_val1.shape)

	print('Labelled0')
	print(x_train0.shape)
	print(x_test0.shape)
	print(y_train0.shape)
	print(y_test0.shape)
	print(x_val0.shape)
	print(y_val0.shape)



	print('Combined')
	print(x_train.shape)
	print(x_test.shape)
	print(y_train.shape)
	print(y_test.shape)
	print(x_val.shape)
	print(y_val.shape)

	# Combine the image data and the corresponding labels
	# combined_training_dataset = [x_train, y_train]
	# combined_test_dataset = [x_test, y_test] 

	combined_dataset = ([x_train, y_train], [x_test, y_test], [x_val, y_val])
# # Seperate into training and testing datasets
# x_train = images[:int(no_of_images/2), :, :, :]
# y_train = np.zeros(len(x_train))
# x_test  = images[int(no_of_images/2):, :, :, :]
# y_test= np.zeros(len(x_test ))

# combined_dataset = ([x_train, y_train], [x_test, y_test])

# Save the image data(array) and labels in a pickle file
# with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/my_cropsweeds_dataset_size200_part_aug.pickle', 'wb') as output:
# 	pickle.dump(combined_dataset1, output, protocol=pickle.HIGHEST_PROTOCOL)

with open('/extend_sda/Ananya_files/Weeding Bot Project/Farm Photos/Separated Photos_blackwhite/my_cropsweeds_dataset_size200_all_aug.pickle', 'wb') as output:
	pickle.dump(combined_dataset, output, protocol=pickle.HIGHEST_PROTOCOL)



# with open('/data_2/Ananya_files_2/my_cropsweeds_dataset_size200_all.txt', 'w') as f:
#   json.dump(combined_dataset, f)