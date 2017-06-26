
# coding: utf-8

# In[3]:

#!/usr/bin/env python

import os
import numpy as np
import nrrd

main_dir = "/home/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMaps_1.00-1.00-1.00"

all_cases = [os.path.join(main_dir, folder) for folder in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, folder))]

for case in all_cases:

    print(case)
    
    needles = [needle for needle in os.listdir(case) if os.path.isfile(os.path.join(case, needle)) and 'needle' in needle]
    
    mri = nrrd.read('%s/case.nrrd' % case)[0]

    needles_sum = np.zeros_like(mri).astype(np.uint16)

    valid_needles = 0

    for needle in needles:
        needle = nrrd.read('%s/%s' % (case, needle))[0].astype(np.uint16)

        needle[needle!=0] = 1

        if needle.sum() > int(0.25 * needle.shape[0] * needle.shape[1] * needle.shape[2]):
            pass

        else:
            needles_sum += needle
            valid_needles += 1

    print(' %d valid needles on %d' % (valid_needles, len(needles)))
    
    needles_sum[needles_sum!=0] = 1

    nrrd.write('%s/needles.nrrd' % (case), needles_sum)

print("Done!")


# In[12]:

np.sum(nrrd.read(main_dir+'/002/needle-1.nrrd')[0])


# In[ ]:



