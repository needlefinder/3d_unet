
# coding: utf-8

# In[ ]:

get_ipython().magic('matplotlib inline')
import os
import shutil
from time import gmtime, strftime
import numpy as np
from collections import OrderedDict
import logging
import nrrd
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib.pylab as plt
from PIL import Image
logging.basicConfig(filename="logging_info_"+strftime("%Y-%m-%d %H:%M:%S", gmtime())+".log",level=logging.DEBUG, format='%(asctime)s %(message)s')

from syntheticdata import synthetic_generation


# In[ ]:

Num_Training = 50
Num_Validation = 5
Num_Testing = 5

training_array = []
validation_array = []
testing_array = []

synthetic = synthetic_generation.Synthetic()

for num in range(Num_Training):
    while True:
        labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 10000:
                training_array.append([case, labelmap])
                break
    
print('*'* 20)

for num in range(Num_Validation):
    while True:
        labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 10000:
                validation_array.append([case, labelmap])
                break

print('*'*20)    
    
for num in range(Num_Testing):
    while True:
        labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 10000:
                testing_array.append([case, labelmap])
                break

                
SAVEPATH = "/home/ubuntu/ziyang/preprocessed_data/LabelMaps_1.00-1.00-1.00_synthetic/"
NUMPYPATH = SAVEPATH + "numpy/"
NRRDPATH = SAVEPATH + "nrrd/"
assert len(training_array) == Num_Training
for num, array in enumerate(training_array):
    case = array[0]
    labelmap = array[1]
    np.save(NUMPYPATH + "training_labelmap{}.npy".format(num+1), labelmap)
    np.save(NUMPYPATH + "training_case{}.npy".format(num+1), case)
    nrrd.write(NRRDPATH + "training_labelmap{}.nrrd".format(num+1), labelmap)
    nrrd.write(NRRDPATH + "training_case{}.nrrd".format(num+1), case)
    
assert len(validation_array) == Num_Validation
for num, array in enumerate(validation_array):
    case = array[0]
    labelmap = array[1]
    np.save(NUMPYPATH + "validation_labelmap{}.npy".format(num+1), labelmap)
    np.save(NUMPYPATH + "validation_case{}.npy".format(num+1), case)
    nrrd.write(NRRDPATH + "validation_labelmap{}.nrrd".format(num+1), labelmap)
    nrrd.write(NRRDPATH + "validation_case{}.nrrd".format(num+1), case)
    
assert len(testing_array) == Num_Testing
for num, array in enumerate(testing_array):
    case = array[0]
    labelmap = array[1]
    np.save(NUMPYPATH + "testing_labelmap{}.npy".format(num+1), labelmap)
    np.save(NUMPYPATH + "testing_case{}.npy".format(num+1), case)
    nrrd.write(NRRDPATH + "testing_labelmap{}.nrrd".format(num+1), labelmap)
    nrrd.write(NRRDPATH + "testing_case{}.nrrd".format(num+1), case)

