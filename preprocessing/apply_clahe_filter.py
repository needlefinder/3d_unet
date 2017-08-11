
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
from fns import *
from syntheticdata import synthetic_generation


# In[2]:

rootPath = "/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00/"
savePath = rootPath + "LabelMapsNEW2_1.00-1.00-1.00_clahe/"

trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")

files = [dataPath + name + '/case.nrrd' for name in trainingCases], [dataPath + name + '/case.nrrd' for name in
                                                                            validationCases], [
                   dataPath + name + '/case.nrrd' for name in testingCases]
files = np.concatenate(files).tolist()


# In[3]:

for i in trange(len(files)):
    file = files[i]
    data, options = nrrd.read(file)
    data = data.astype(np.uint16)
    nz = data.shape[2]
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(30, 30))
    data_res = []
    for k in range(nz):
        cl = data[..., k].copy()
        cl = cl.astype('uint16')
        data_res.append(clahe.apply(cl))
    data_res = np.swapaxes(data_res, 0, 1)
    data_res = np.swapaxes(data_res, 1, 2)
    savePath = file.replace("LabelMapsNEW2_1.00-1.00-1.00/", "LabelMapsNEW2_1.00-1.00-1.00_clahe/")
    print(savePath)
    try:
        os.mkdir( savePath[:-10])
    except:
        pass
    nrrd.write(savePath, data_res)


# In[19]:

os.mkdir(savePath[:-10])

