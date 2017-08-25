
# coding: utf-8

# In[4]:


get_ipython().magic('matplotlib inline')
import sys
sys.path.append('../')
from fns import *
from syntheticdata import synthetic_generation
from skimage import morphology


# In[5]:


training_array = []
validation_array = []
testing_array = []


# In[6]:


class Synthetic(object):
    def __init__(self, N=10000):
        
        DATAPATH = "/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00"
        CASEPATH = [DATAPATH + '/' + s for s in os.listdir(DATAPATH) if s != '.DS_Store']

        self.needlepaths = []
        self.casepaths = []
        for case in CASEPATH[:N]:
            print(case+'/needle-Manual_*')
            needlepath = glob.glob(case+'/needle-*-Manual_*')
            casepath = [case + '/' + s for s in os.listdir(case) if 'case' in s]
            self.needlepaths = self.needlepaths + needlepath
            self.casepaths = self.casepaths + casepath
        print('Number of Needles:',len(self.needlepaths))
        print("Number of Cases:",len(self.casepaths))
    
        #load needles
        self.needles = []
        for needlepath in self.needlepaths[:N]:
            needleimg = nrrd.read(needlepath)
            needlearray = needleimg[0]
            needlearray[needlearray != 0] = 1
            self.needles.append(needlearray)
        
        #load cases
        self.cases = []
        for casepath in self.casepaths[:N]:
            try:
                caseimg = nrrd.read(casepath)
                casearray = caseimg[0]
                self.cases.append(casearray)
            except:
                pass

            
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

    
    def get_array(self, num_needles, degree, shape, occult=True, std=None, mean=None):
        #################
        ## generate synthetic labelmap
        ##################
        random.shuffle(self.needles)
        labelmap_syn = np.zeros(shape,int)## int may need to be replaced by float
        labelmap_syn_morph = np.zeros(shape,int)## int may need to be replaced by float
        count_needle=0
        i=-1
        while count_needle < num_needles:
            i+=1
            needle = self.needles[np.random.randint(len(self.needles))]
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
            if np.sum(needle_crop)>700:
                count_needle+=1
                labelmap_syn = labelmap_syn + needle_crop   ## There will be chance of needle overlap
                # morphology
                if np.random.randint(2):
                    needle_crop = morphology.binary_dilation(needle_crop)
                    needle_crop = self.labelmap_resize(needle_crop, shape)

                    
                labelmap_syn_morph =  labelmap_syn_morph + needle_crop
            
        if occult:
            square = np.ones((40,40,40))
            squares = np.zeros(shape)
            for i in range(1):
                sq = np.where(square!=0)
#                 sqx = sq[0] + np.random.randint(0,148)
#                 sqy = sq[1] + np.random.randint(0,148)
#                 sqz = sq[2] + np.random.randint(0,148)
                sqx = sq[0] + shape[0]//2 -20
                sqy = sq[1] + shape[1]//2 -20
                sqz = sq[2] + shape[2]//2 -20
                squares[np.clip(sqx,0,shape[0]-1), np.clip(sqy,0,shape[1]-1), np.clip(sqz,0,shape[2]-1)] = 1
            
        #################
        ## generate synthetic case
        ##################
        if std == None:
            std = np.std(self.cases[0])
        if mean == None:
            mean = np.mean(self.cases[0])
        case_syn = np.random.normal(np.log(std), mean, shape)
        case_syn = gaussian_filter(case_syn, sigma=6)
        case_syn2 = np.random.normal(np.log(std), mean, shape)
        case_syn2 = gaussian_filter(case_syn, sigma=3)
        case_syn1 = np.random.normal(np.log(std), mean, shape)
        case_syn1 = gaussian_filter(case_syn, sigma=2)
        case_syn0 = gaussian_filter(case_syn, sigma=0.3)
        case_syn = case_syn0*case_syn1*case_syn2*case_syn
        case_syn /= np.mean(case_syn)
        case_syn *= mean
        
        
        case_syn = np.clip(case_syn,0,500)
        VAL_NEEDLE = np.random.randint(np.int(np.mean(case_syn)/3))
        VAL_NEEDLE = 0
        if occult:
            case_syn[(squares != 0)] = 0
            case_syn[labelmap_syn_morph != 0] = VAL_NEEDLE
        else:
            case_syn[labelmap_syn_morph != 0] = VAL_NEEDLE
        case_syn = gaussian_filter(case_syn, sigma=0.7)
    
        return labelmap_syn, case_syn


# In[7]:


synthetic = Synthetic(N=10000)


# In[8]:


from numpy import random
import sys
sys.path.append('../')
import os
import numpy as np
import random
from transforms3d.euler import euler2mat, mat2euler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
import skimage.transform
import nrrd
from scipy.ndimage.filters import gaussian_filter
import glob


# In[9]:


Num_Training = 1000
Num_Validation = 100
Num_Testing = 100

SAVEPATH = "/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_synthetic_occulted_new/"
NUMPYPATH = SAVEPATH + "numpy/"
NRRDPATH = SAVEPATH + "nrrd/"
try:
    os.mkdir(SAVEPATH)
    os.mkdir(NUMPYPATH)
    os.mkdir(NRRDPATH)
except:
    pass
                            
def generate_cases(numToDo=1, mode="training"):
    numberOfCases = len(glob.glob(NUMPYPATH + "%s_labelmap*"%mode))
    for num in trange(numToDo):
        while True:
            labelmap, case = synthetic.get_array(num_needles=20, degree=10, shape = (148, 148, 148))
            labelmap[labelmap != 0] = 1 
            unique, counts = np.unique(labelmap, return_counts = True)
            counts = np.sum(labelmap)
            print("Num: ", num, counts)
            if counts >= 100 and counts <= 30000:
                    training_array.append([case, labelmap])
                    np.save(NUMPYPATH + "{}_labelmap{}.npy".format(mode, numberOfCases + num+1), labelmap)
                    np.save(NUMPYPATH + "{}_case{}.npy".format(mode, numberOfCases + num+1), case)
                    nrrd.write(NRRDPATH + "{}_labelmap{}.nrrd".format(mode, numberOfCases + num+1), labelmap.astype(np.uint8))
                    nrrd.write(NRRDPATH + "{}_case{}.nrrd".format(mode, numberOfCases + num+1), case)
                    break

generate_cases(Num_Training, 'training')
generate_cases(Num_Testing, 'testing')
generate_cases(Num_Validation, 'validation')


# In[22]:


numberOfCases = len(glob.glob(NUMPYPATH + "training_labelmap*"))


# In[15]:





# In[11]:



# assert len(training_array) == Num_Training
# for num, array in enumerate(training_array):
#     case = array[0].astype(np.int32)
#     labelmap = array[1].astype(np.int32)
#     np.save(NUMPYPATH + "training_labelmap{}.npy".format(num+1), labelmap)
#     np.save(NUMPYPATH + "training_case{}.npy".format(num+1), case)
#     nrrd.write(NRRDPATH + "training_labelmap{}.nrrd".format(num+1), labelmap)
#     nrrd.write(NRRDPATH + "training_case{}.nrrd".format(num+1), case)
    
# assert len(validation_array) == Num_Validation
# for num, array in enumerate(validation_array):
#     case = array[0].astype(np.int32)
#     labelmap = array[1].astype(np.int32)
#     np.save(NUMPYPATH + "validation_labelmap{}.npy".format(num+1), labelmap)
#     np.save(NUMPYPATH + "validation_case{}.npy".format(num+1), case)
#     nrrd.write(NRRDPATH + "validation_labelmap{}.nrrd".format(num+1), labelmap)
#     nrrd.write(NRRDPATH + "validation_case{}.nrrd".format(num+1), case)
    
# assert len(testing_array) == Num_Testing
# for num, array in enumerate(testing_array):
#     case = array[0].astype(np.int32)
#     labelmap = array[1].astype(np.int32)
#     np.save(NUMPYPATH + "testing_labelmap{}.npy".format(num+1), labelmap)
#     np.save(NUMPYPATH + "testing_case{}.npy".format(num+1), case)
#     nrrd.write(NRRDPATH + "testing_labelmap{}.nrrd".format(num+1), labelmap)
#     nrrd.write(NRRDPATH + "testing_case{}.nrrd".format(num+1), case)


# In[4]:


# plt.hist(np.random.lognormal(0,1.3,1000))


# In[13]:


np.random.randint(2)


# In[ ]:




