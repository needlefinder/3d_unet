
# coding: utf-8

# In[1]:

from fns import *


# In[ ]:

rootPath = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
claheDataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00_clahe/"
trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")
files = np.concatenate([[claheDataPath + name + '/case.nrrd' for name in trainingCases], 
                [claheDataPath + name + '/case.nrrd' for name in validationCases], 
                [claheDataPath + name + '/case.nrrd' for name in testingCases]])
files_labels = np.concatenate([[claheDataPath + name + '/needles.nrrd' for name in trainingCases], 
                [claheDataPath + name + '/needles.nrrd' for name in validationCases], 
                [claheDataPath + name + '/needles.nrrd' for name in testingCases]])


# In[ ]:

for file in files:
    data, options = nrrd.read(file)
    data = data.astype(np.float32)
    arr_data = cutVolume(data)
    for i, img in enumerate(arr_data):
        name = file.replace('.nrrd', '')
        name += '_%d.nrrd'%i
        nrrd.write(name, img, options=options)


# In[ ]:

for file in files_needles:
    data, options = nrrd.read(file)
    data = data.astype(np.float32)
    arr_data = cutVolume(data)
    for i, img in enumerate(arr_data):
        name = file.replace('.nrrd', '')
        name += '_%d.nrrd'%i
        nrrd.write(name, img, options=options)


# In[ ]:



