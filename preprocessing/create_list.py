
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
from fns import *
import glob


# In[2]:


rootPath = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
claheDataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")

files_training = [claheDataPath + name + '/case.nrrd' for name in trainingCases]
files_validation = [claheDataPath + name + '/case.nrrd' for name in validationCases]
files_testing = [claheDataPath + name + '/case.nrrd' for name in testingCases]


# In[5]:


off = 44 

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
            label_file = file.replace('case', 'needles')
            label = nrrd.read(label_file)[0].astype('uint16')
            sum_label = np.sum(label[off:-off,off:-off,off:-off])
            if sum_label > 400:
                f.write(file + '\n')



# In[6]:


inputs[2]


# In[7]:


with open('training_subvolumes.txt', 'r') as f:
    training_cases = f.read().splitlines()

print(training_cases)


# ## synthetic

# In[3]:


rootPath = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00_synthetic_occulted/"
cases = glob.glob(dataPath + 'nrrd/' + 'training_case*')


# In[4]:


nbTraining = int(len(cases)*0.8)
numberOfValidation = int(len(cases)*0.1)
training_cases = cases[:nbTraining]
validation_cases = cases[nbTraining:nbTraining+numberOfValidation]
testing_cases = cases[nbTraining+numberOfValidation:]

inputs = [training_cases, validation_cases, testing_cases]
outputs = ['training_subvolumes_synth.txt', 'validation_subvolumes_synth.txt', 'testing_subvolumes_synth.txt']
outputs_lab = ['training_subvolumes_synth_label.txt', 'validation_subvolumes_synth_label.txt', 'testing_subvolumes_synth_label.txt']


# In[5]:


for i in range(3):
    with open(outputs[i], 'w') as f:
        for file in inputs[i]: 
            f.write(file + '\n')
            
for i in range(3):
    with open(outputs_lab[i], 'w') as f:
        for file in inputs[i]: 
            file= file.replace('case', 'labelmap')
            f.write(file + '\n')
        


# In[6]:


print(len(training_cases))


# In[ ]:




