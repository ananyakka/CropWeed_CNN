''' 
Create crop-weed image dataset as ndarray and save it using pickle

Author: Ananya, 07/26/17

Incomplete
'''

import PIL
import pickle
from PIL import Image
from numpy import array
from glob import glob
import numpy as np
import pandas as pd
import os

image = Image.open(glob("/home/ananya/Weeding Bot Project/Data/sample_pepper_images/*.jpg"))
print ("Creating numpy representation of image %s " % file)
resize = image.resize((300,300), Image.NEAREST) 
resize.load()
data = np.asarray( resize, dtype="uint8" )
print(data.shape)
master_dataset.append(data)