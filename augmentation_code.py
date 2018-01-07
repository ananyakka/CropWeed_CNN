def dataset_maker_aug(glob_file_key, loc_train_labels=""):
	dataset = np.empty([0, image_size, image_size, 3])  # Create empty array to store pixel values of new images
	datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False,
    vertical_flip = False)
	
	max_angle = 180
	incr_angle = (max_angle-0)/100
	max_offset=20
	x_loc =10
	y_loc =10
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
		arr = array(img)			  # convert PIL image to array
		# print(arr.shape)

		#arr = arr.transpose(2, 0, 1)
		#dataset = np.concatenate([arr[np.newaxis]])

		dataset= np.append(dataset, [arr], axis=0)  # append the array to the image data set

		for ang in range(0, max_angle, incr_angle):
			# x_offset = 10
			# y_offset = 10
			img2= img.rotate(ang)
			for x_offset in range(0, image_size, 10):
				for y_offset in range(0, image_size, 10):
					img3 = ImageChops.offset(img2, x_offset, y_offset)
					img3.paste((0, 0, 0), (0, 0, image_size, y_offset))
					arr=array(img3)
					dataset= np.append(dataset, [arr], axis=0)
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