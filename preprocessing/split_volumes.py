
# coding: utf-8

# In[1]:


import sys, sh
sys.path.append('../')
from fns import *


# In[2]:


rootPath = "/home/gp1514/SSD/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
savePath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00_split2/"

trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")
files = np.concatenate([[dataPath + name + '/case.nrrd' for name in trainingCases], 
                [dataPath + name + '/case.nrrd' for name in validationCases], 
                [dataPath + name + '/case.nrrd' for name in testingCases]])
files_labels = np.concatenate([[dataPath + name + '/needles.nrrd' for name in trainingCases], 
                [dataPath + name + '/needles.nrrd' for name in validationCases], 
                [dataPath + name + '/needles.nrrd' for name in testingCases]])


# In[56]:


get_ipython().magic('matplotlib inline')
file = files[10]
data, options = nrrd.read(file)
data = data.astype(np.float32)


clahe_filter = True
if clahe_filter:
    nx,ny,nz = data.shape
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(5, 5))
    data_res = []
    for k in range(nz):
        cl = data[...,k].copy()
        cl = cl.astype('uint16')
        data_res.append(clahe.apply(cl))
    data_res = np.array(data_res)
    data_res = np.swapaxes(data_res, 0, 1)
    data_res = np.swapaxes(data_res, 1, 2)
    data_ = data_res
    
#     clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(10, 10))
#     data_res = []
#     for k in range(nx):
#         cl = data[k,...].copy()
#         cl = cl.astype('uint16')
#         data_res.append(clahe.apply(cl))
#     data_res = np.array(data_res)
#     data_ += data_res


# In[57]:



f(15,15)
print(data_.shape)
plt.imshow(data_[...,50], cmap=plt.cm.RdBu)


# In[4]:


for j in trange(len(files)):
    file = files[j]
    data, options = nrrd.read(file)
    data = data.astype(np.float32)
    
    clahe_filter = True
    if clahe_filter:
        nx,ny,nz = data.shape
        clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(30, 30))
        data_res = []
        for k in range(nz):
            cl = data[..., k].copy()
            cl = cl.astype('uint16')
            data_res.append(clahe.apply(cl))
        data_res = np.swapaxes(data_res, 0, 1)
        data_res = np.swapaxes(data_res, 1, 2)
        data = data_res
    
    
    arr_data = cutVolume(data, tile=144)
    file= file.replace(dataPath,savePath)
    
    dirPath = os.path.dirname(file)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    if not os.path.exists(file):
        nrrd.write(file, data)
    for i in trange(len(arr_data)):
        img = arr_data[i]
        name = file.replace('.nrrd', '')
        name += '_%d.npy'%i
        np.save(name, img)


# In[6]:


for file in files_labels:
    data, options = nrrd.read(file)
    data = data.astype(np.uint8)
    arr_data = cutVolume(data)
    file= file.replace(dataPath,savePath)
    if not os.path.exists(file):
        nrrd.write(file, data)
    dirPath = os.path.dirname(file)
    if not os.path.exists(dirPath):
        sh.mkdir(dirPath)
    for i in trange(len(arr_data)):
        img = arr_data[i]
        name = file.replace('.nrrd', '')
        name += '_%d.npy'%i
        np.save(name, img)


# In[ ]:




