
# coding: utf-8

# In[47]:

import os
import nrrd
import scipy.misc
import numpy as np
from transforms3d.euler import euler2mat, mat2euler
#np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
DATAPATH = "/home/ubuntu/preprocessed_data/LabelMaps_1.00-1.00-1.00/007/"

import skimage.transform
import SimpleITK as sitk

labelmap = sitk.GetArrayFromImage(sitk.ReadImage("/home/ubuntu/preprocessed_data/LabelMaps_1.00-1.00-1.00/007/labelmap.nrrd"))     
case = sitk.GetArrayFromImage(sitk.ReadImage("/home/ubuntu/preprocessed_data/LabelMaps_1.00-1.00-1.00/007/case.nrrd"))
needlenames = [s for s in os.listdir("/home/ubuntu/preprocessed_data/LabelMaps_1.00-1.00-1.00/007/") if 'needle' in s]
needlepaths = [DATAPATH + s for s in needlenames]
print('NeedlePath:\n',needlepaths)


# In[48]:

needles = []
for needlepath in needlepaths:
    needleimg = (sitk.ReadImage(needlepath))
    needlearray = sitk.GetArrayFromImage(needleimg)
    needlearray[needlearray!=0.0] = 1
    needles.append(needlearray)


# In[49]:

print("Shape Before Swap:",needles[0].shape)

needles_rot1 = np.swapaxes(needles[0], 1, 2)
print("Shape After Swap:",needles_rot1.shape)

needles_rot1_rs = skimage.transform.resize(needles_rot1, needles[0].shape, order=0)
print("Shape After Resize:",needles_rot1_rs.shape)

needles_rot1_rs[needles_rot1_rs != 0] = 1
needles_rot1_rs = needles_rot1_rs.astype(np.int8)
unique, count = np.unique(needles_rot1_rs, return_counts=True)
print(unique, count)


# In[50]:

xs, ys, zs = np.where(needles[0] != 0)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)


# In[51]:

xs, ys, zs = np.where(needles_rot1_rs != 0)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)


# In[52]:

needle0 = needles[0]
#rotate the needle in each dimension for -60 degree to 60 degree randomly
X, Y, Z = needle0.shape
unique,count = np.unique(needle0,return_counts=True)
print(count)
print(X, Y, Z)


# In[53]:

degree1 = np.clip(np.random.normal(),-2,2)*5
for x in range(X):
    needle0[x,:,:] = scipy.misc.imrotate(needle0[x,:,:], degree1)
    
unique,count = np.unique(needle0,return_counts=True)
print(count)
print(needle0.shape)


# In[54]:

degree2 = np.clip(np.random.normal(),-2,2)*30
for y in range(Y):
    needle0[:,y,:] = scipy.misc.imrotate(needle0[:,y,:], degree2)
    
unique,count = np.unique(needle0,return_counts=True)
print(count)
print(needle0.shape)


# In[55]:

degree3 = np.clip(np.random.normal(),-2,2)*30
for z in range(Z):
    needle0[:,:,z] = scipy.misc.imrotate(needle0[:,:,z], degree3)
    
unique,count = np.unique(needle0,return_counts=True)
print(count)
print(needle0.shape)


# In[56]:

xs, ys, zs = np.where(needle0 == 1)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)


# In[11]:

minval = np.amin(case)
maxval = np.amax(case)
x , y ,z = case.shape
random_volume = minval + (maxval - minval) * np.random.rand(x,y,z)
needle_volume = random_volume * needle0


# In[ ]:



