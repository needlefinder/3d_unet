
# coding: utf-8

# In[31]:

from fns import *
import glob


# In[32]:

rootPath = "/home/gp1514/DATA/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
claheDataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")

files_training = [claheDataPath + name + '/case.nrrd' for name in trainingCases]
files_validation = [claheDataPath + name + '/case.nrrd' for name in validationCases]
files_testing = [claheDataPath + name + '/case.nrrd' for name in testingCases]


# In[33]:

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



# In[34]:

inputs[2]


# In[35]:

with open('training_subvolumes.txt', 'r') as f:
    training_cases = f.read().splitlines()

print(training_cases)


# In[25]:

get_ipython().magic('pinfo f.readlines')


# In[ ]:



