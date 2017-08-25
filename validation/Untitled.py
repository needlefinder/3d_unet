
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')


# In[2]:


import sys
sys.path.append('../')
from fns import *


# In[3]:


df_4layer = pd.read_csv('../unet_trained_bk/run_.,tag_loss.csv')
df_4layer.head(10)

df_3layer = pd.read_csv('../unet_trained/run_.,tag_loss.csv')
df_3layer.head(10)


# In[5]:


f(15,4)
smooth=10
plt.plot(mva(df_4layer.Value.values, smooth), label='4 layers')
plt.plot(mva(df_3layer.Value.values, smooth), label='3 layers')
plt.legend()


# In[ ]:




