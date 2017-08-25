
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
from fns import *


# In[ ]:


files_training, files_validation, files_testing = [], [], []

with open('../preprocessing/training_subvolumes_synth.txt', 'r') as f:
    files_training = f.read().splitlines()
with open('../preprocessing/validation_subvolumes_synth.txt', 'r') as f:
    files_validation = f.read().splitlines()
with open('../preprocessing/testing_subvolumes_synth.txt', 'r') as f:
    files_testing = f.read().splitlines()   

with open('../preprocessing/training_subvolumes.txt', 'r') as f:
    files_training += f.read().splitlines()
with open('../preprocessing/validation_subvolumes.txt', 'r') as f:
    files_validation += f.read().splitlines()
with open('../preprocessing/testing_subvolumes.txt', 'r') as f:
    files_testing += f.read().splitlines() 


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = '../data.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)


for i in tnrange(len(files_training)):
    image_path = files_training[i]
    
    label_name = image_path.replace('_case', '_labelmap')
    label_name = label_name.replace('case', 'needles')
    label_name = label_name.replace('_clahe', '')
    
    img = nrrd.read(img_path)[0]
    img = img.astype(np.float32)
    
    annotation = nrrd.read(label_name)[0]
    annotation = annotation.astype(np.uint16)
    annotation = crop_to_shape2(annotation, (60,60,60))
    
    height = img.shape[0]
    width = img.shape[1]
    deepth = img.shape[2]
    
    img_raw = img.tostring()
    annotation_raw = annotation.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'deepth': _int64_feature(deepth),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(annotation_raw)}))
    
    writer.write(example.SerializeToString())

writer.close()


# In[ ]:




