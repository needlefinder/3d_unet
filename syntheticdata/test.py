
# coding: utf-8

# In[5]:

import os
import numpy as np
import random
from transforms3d.euler import euler2mat, mat2euler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
import skimage.transform
import nrrd


# In[6]:

DATAPATH = "/home/ubuntu/preprocessed_data/LabelMaps_1.00-1.00-1.00"
WRITINGPATH = "/home/ubuntu/ziyang/preprocessed_data/LabelMaps_1.00-1.00-1.00"
CASEPATH = [DATAPATH + '/' + s for s in os.listdir(DATAPATH) if s != '.DS_Store']


needlepaths = []
casepaths = []
labelmappaths = []
for case in CASEPATH:
    if not os.path.exists(case.replace(DATAPATH,WRITINGPATH)):
        os.makedirs(case.replace(DATAPATH,WRITINGPATH))
    needlepath = [case + '/' + s for s in os.listdir(case) if 'needle' in s]
    casepath = [case + '/' + s for s in os.listdir(case) if 'case' in s]
    labelmappath = [case + '/' + s for s in os.listdir(case) if 'labelmap' in s]
    needlepaths = needlepaths + needlepath
    casepaths = casepaths + casepath
    labelmappaths = labelmappaths + labelmappath
needlepaths.sort()
casepaths.sort()
labelmappaths.sort()

assert len(casepaths) == len(labelmappaths),"Number of case.nrrd and labelmap.case doesn't match"
print('Number of Needles:',len(needlepaths))
print("Number of Cases:",len(casepaths))
    
#load needles
needles = []
#random.shuffle(needlepaths)
'''
for needlepath in needlepaths:
    needleimg = nrrd.read(needlepath)
    needlearray = needleimg[0]
    nrrd.write(needlepath.replace(DATAPATH,WRITINGPATH),needlearray)
    needlearray[needlearray != 0] = 1
    needles.append(needlearray)
'''

labelmaps = []
for labelmappath in labelmappaths:
    labelimg = nrrd.read(labelmappath)
    labelarray = labelimg[0]
    labelarray = labelarray - np.amin(labelarray)
    nrrd.write(labelmappath.replace(DATAPATH,WRITINGPATH),labelarray)
    labelmaps.append(labelarray)
        
#load cases
cases = []
for casepath in casepaths:
    caseimg = nrrd.read(casepath)
    casearray = caseimg[0]
    nrrd.write(casepath.replace(DATAPATH,WRITINGPATH),casearray)
    cases.append(casearray)


# In[7]:

for i,labelmappath in enumerate(labelmappaths):
    print(i,labelmappath,'Shape: ',labelmaps[i].shape,np.unique(labelmaps[i], return_counts=True))


# In[19]:

for i,needlepath in enumerate(needlepaths):
    print(i,needlepath,'Shape: ',needles[i].shape)


# In[ ]:

xs, ys, zs = np.where(needles[0] != 0)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)


# In[ ]:



