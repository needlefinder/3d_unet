# coding: utf-8
# In[1]:
import sys, sh
sys.path.append('../')
from fns import *
import os
import glob
import argparse
import numpy as np
import gc
import SimpleITK as sitk
USERPATH = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--k", help="case index", type=int)
args = parser.parse_args()
spacing = [1.0,1.0,1.0]

rootPath = "/Users/guillaume/Projects/github/NeedleFinder/data/"
dataPath = rootPath + "LabelMaps_GUILLAUME_03122017/"
savePath = rootPath + "LabelMaps/"

trainingCases = loadCases("training.txt")
validationCases = loadCases("validation.txt")
testingCases = loadCases("testing.txt")
files = np.concatenate([[dataPath + name + '/case.nrrd' for name in trainingCases], 
                [dataPath + name + '/case.nrrd' for name in validationCases], 
                [dataPath + name + '/case.nrrd' for name in testingCases]])
files_labels = np.concatenate([[dataPath + name + '/needles.nrrd' for name in trainingCases], 
                [dataPath + name + '/needles.nrrd' for name in validationCases], 
                [dataPath + name + '/needles.nrrd' for name in testingCases]])


def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return 0


def get_resized_img(k, files, string='case', data_type=sitk.sitkFloat32):
    """
    This function resizes an image to a fixed shape.
    If data type is sitkFloat32 a linear interpolation is used, otherwise nearest neighbor interpolation is used.
    """
    file = files[k]
    img = sitk.ReadImage(file)
    size = img.GetSize()
    ratio = [1.0/i for i in img.GetSpacing()]
    new_size = [int(size[i]/ratio[i]) for i in range(3)]

    rimage = sitk.Image(new_size, data_type)
    rimage.SetSpacing((spacing[0], spacing[1], max(spacing[2],1.0) ))
    rimage.SetOrigin(img.GetOrigin())
    tx = sitk.Transform()

    interp = sitk.sitkLinear
    if data_type == sitk.sitkInt16:
        interp = sitk.sitkNearestNeighbor

    new_image = sitk.Resample(img, rimage, tx, interp, data_type)
    K = file.split('/')[-2]
    print(file, K)
    dirPath = savePath + '%03d/' % int(K)
    createDir(dirPath)
    filename = dirPath + '%s.nrrd' % string
    sitk.WriteImage( new_image, filename )
    return filename

#--------------------------------------------------
## split cases
#--------------------------------------------------

file = get_resized_img(args.k, files, 'case')
data, options = nrrd.read(file)
data = data.astype(np.float32)

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

#--------------------------------------------------
## split labels
#--------------------------------------------------

file = get_resized_img(args.k, files_labels, 'needles', sitk.sitkUInt8)
data, options = nrrd.read(file)
data, options = nrrd.read(file)

data = data.astype(np.uint8)
arr_data = cutVolume(data)
file = file.replace(dataPath,savePath)
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
    
    
del img
del arr_data
gc.collect()
