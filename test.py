
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
import nrrd
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import matplotlib.pylab as plt
from PIL import Image


# In[9]:

channels = 1
n_class = 2
x = tf.placeholder("float", shape=[None, None, None, None, channels], name='data')
y = tf.placeholder("float", shape=[None, None, None, None,  n_class], name='target')

nx = tf.shape(x)[1]
ny = tf.shape(x)[2]
nz = tf.shape(x)[3]
print(nx,ny,nz)

x_image = tf.reshape(x, tf.stack([-1,nx,ny,nz,channels]), name='input_reshape')
in_node = x_image
batch_size = tf.shape(x_image)[0]
print(batch_size)


# In[ ]:



