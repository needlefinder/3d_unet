
# coding: utf-8

# In[12]:

import os
import numpy as np
import random
from transforms3d.euler import euler2mat, mat2euler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
import skimage.transform
import nrrd


# In[13]:

class Synthetic(object):
    def __init__(self):
        DATAPATH = "/home/ubuntu/ziyang/preprocessed_data/LabelMaps_1.00-1.00-1.00"
        CASEPATH = [DATAPATH + '/' + s for s in os.listdir(DATAPATH) if s != '.DS_Store']

        self.needlepaths = []
        self.casepaths = []
        for case in CASEPATH:
            needlepath = [case + '/' + s for s in os.listdir(case) if 'needle' in s]
            casepath = [case + '/' + s for s in os.listdir(case) if 'case' in s]
            self.needlepaths = self.needlepaths + needlepath
            self.casepaths = self.casepaths + casepath
        print('Number of Needles:',len(self.needlepaths))
        print("Number of Cases:",len(self.casepaths))
    
        #load needles
        self.needles = []
        for needlepath in self.needlepaths:
            needleimg = nrrd.read(needlepath)
            needlearray = needleimg[0]
            needlearray[needlearray != 0] = 1
            self.needles.append(needlearray)
        
        #load cases
        self.cases = []
        for casepath in self.casepaths:
            caseimg = nrrd.read(casepath)
            casearray = caseimg[0]
            self.cases.append(casearray)

            
    def labelmap_resize(self, data, shape):
        """
        Crops from center
        """
        offset0 = (data.shape[0] - shape[0])//2
        offset1 = (data.shape[1] - shape[1])//2
        
        if data.shape[2] >= shape[2]:
            offset2 = (data.shape[2] - shape[2])//2
            out = data[offset0:offset0+shape[0], offset1:offset1+shape[1], offset2:offset2+shape[2]]
        else:
            offset2 = (shape[2] - data.shape[2])//2
            out = np.zeros(shape,int)
            out[:,:,offset2:offset2+data.shape[2]] = data[offset0:offset0+shape[0], offset1:offset1+shape[1], :]
            
        return out

    
    def get_array(self, num_needles, degree, shape, std=None, mean=None):
        #################
        ## generate synthetic labelmap
        ##################
        random.shuffle(self.needles)
        labelmap_syn = np.zeros(shape,int)           ## int may need to be replaced by float
        for i,needle in enumerate(self.needles[0:num_needles]):
            xs, ys, zs = np.where(needle == 1)
            I = np.asarray([[x, y, z] for x, y, z in zip(xs,ys,zs)])
            I = np.transpose(I)
            x_angle = (random.random()-0.5)*2 *degree*np.pi/180
            y_angle = (random.random()-0.5)*2 *degree*np.pi/180
            z_angle = (random.random()-0.5)*2 *degree*np.pi/180
            R = euler2mat(x_angle, y_angle, z_angle, 'sxyz')
            try:
                M = np.transpose(np.dot(R,I)).astype(int)
            except ValueError:
                print("Empty Needle",self.needlepaths[i])
                continue
            needle_rot = np.zeros(needle.shape,int)
            for x,y,z in M:
                try:
                    needle_rot[x][y][z] = 1
                except IndexError:
                    continue
            needle_crop = self.labelmap_resize(needle_rot, shape)
            labelmap_syn = labelmap_syn + needle_crop   ## There will be chance of needle overlap
            
        #################
        ## generate synthetic case
        ##################
        if std == None:
            std = np.std(self.cases[0])
        if mean == None:
            mean = np.mean(self.cases[0])
        case_syn = np.random.normal(std, mean, shape)
        case_syn[labelmap_syn != 0] = 0
    
        return labelmap_syn,case_syn


# In[14]:

#synthetic = Synthetic() 
#labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (260, 320, 320))


# In[21]:

#labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (260, 320, 320))


# In[22]:

#xs, ys, zs = np.where(labelmap != 0)
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=10)



