import pickle


# pickle_filename = input(" Enter the name of the pickle file (Eg.: my_cropweed_dataset2.pickle):")
# with open(pickle_filename, 'rb') as data:
#     dataset = pickle.load(data)

with open('my_cropsweeds_BW_dataset_size200_all.pickle', 'rb') as data:
    (x_train, y_train), (x_test, y_test) = pickle.load(data)

#print(dataset[0])
#print(type(dataset)
#print(dataset[1].shape)
#print(type(dataset))
#y_shape= dataset[1].shape
print(y_test)

x_train_shape = x_train.shape
y_train_shape = y_train.shape
x_test_shape = x_test.shape
y_test_shape = y_test.shape

print(x_train_shape)
print(y_train_shape )
print(x_test_shape )
print(y_test_shape )
#print(y_shape)