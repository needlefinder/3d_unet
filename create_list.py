
# coding: utf-8

# In[1]:

from fns import *
import glob


# In[5]:

rootPath = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
claheDataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00_clahe/"
trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")

files_training = [claheDataPath + name + '/case.nrrd' for name in trainingCases]
files_validation = [claheDataPath + name + '/case.nrrd' for name in validationCases]
files_testing = [claheDataPath + name + '/case.nrrd' for name in testingCases]


# In[21]:

inputs = [files_training, 
          files_validation, 
          files_testing]

outputs = ['training_subvolumes.txt', 'validation_subvolumes.txt', 'testing_subvolumes.txt']

for i in range(3):
    files = []

    for file in inputs[i]: 
        folder = file.replace('case.nrrd', '')
        files.append(glob.glob(folder+'case_*.nrrd'))

    files = np.concatenate(files)
    with open(outputs[i], 'w') as f:
        for file in files:
            
            f.write(file + '\n')



# In[22]:

inputs[2]


# In[30]:

with open('training_subvolumes.txt', 'r') as f:
    training_cases = f.read().splitlines()

print(training_cases)


# In[25]:

get_ipython().magic('pinfo f.readlines')


# In[ ]:



