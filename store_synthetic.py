
# coding: utf-8

# In[18]:

get_ipython().magic('matplotlib inline')
from fns import *
from syntheticdata import synthetic_generation


# In[3]:

training_array = []
validation_array = []
testing_array = []

synthetic = synthetic_generation.Synthetic(N=10000)


# In[ ]:

Num_Training = 300
Num_Validation = 10
Num_Testing = 10

SAVEPATH = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_synthetic_occulted/"
NUMPYPATH = SAVEPATH + "numpy/"
NRRDPATH = SAVEPATH + "nrrd/"
try:
    os.mkdir(SAVEPATH)
    os.mkdir(NUMPYPATH)
    os.mkdir(NRRDPATH)
except:
    pass

for num in trange(Num_Training):
    while True:
        labelmap, case = synthetic.get_array(num_needles=40, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 80000:
                training_array.append([case, labelmap])
                np.save(NUMPYPATH + "training_labelmap{}.npy".format(num+1), labelmap)
                np.save(NUMPYPATH + "training_case{}.npy".format(num+1), case)
                nrrd.write(NRRDPATH + "training_labelmap{}.nrrd".format(num+1), labelmap)
                nrrd.write(NRRDPATH + "training_case{}.nrrd".format(num+1), case)
                break
    
print('*'* 20)

for num in trange(Num_Validation):
    while True:
        labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 48000:
                validation_array.append([case, labelmap])
                np.save(NUMPYPATH + "validation_labelmap{}.npy".format(num+1), labelmap)
                np.save(NUMPYPATH + "validation_case{}.npy".format(num+1), case)
                nrrd.write(NRRDPATH + "validation_labelmap{}.nrrd".format(num+1), labelmap)
                nrrd.write(NRRDPATH + "validation_case{}.nrrd".format(num+1), case)
                break

print('*'*20)    
    
for num in trange(Num_Testing):
    while True:
        labelmap, case = synthetic.get_array(num_needles=10, degree=10, shape = (148, 148, 148))
        labelmap[labelmap != 0] = 1 
        unique, counts = np.unique(labelmap, return_counts = True)
        print("Num: ", num, unique, counts)
        if counts[1] >= 5000 and counts[1] <= 80000:
                testing_array.append([case, labelmap])
                np.save(NUMPYPATH + "testing_labelmap{}.npy".format(num+1), labelmap)
                np.save(NUMPYPATH + "testing_case{}.npy".format(num+1), case)
                nrrd.write(NRRDPATH + "testing_labelmap{}.nrrd".format(num+1), labelmap)
                nrrd.write(NRRDPATH + "testing_case{}.nrrd".format(num+1), case)
                break

                


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


# In[ ]:



