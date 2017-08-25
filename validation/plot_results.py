
# coding: utf-8

# In[2]:


get_ipython().magic('matplotlib inline')
import tensorflow as tf


# In[6]:


import sys
sys.path.append('../')
from fns import *
from tensorboard.backend.event_processing import event_accumulator


# In[4]:


path = "/home/gp1514/CNL2/Dropbox/Projects/pw25/ziyang/3d_unet/models"


# In[29]:


df_4layer = pd.read_csv(path + '/unet_trained_bk/run_.,tag_loss.csv')
df_4layer.head(10)

df_3layer = pd.read_csv(path + '/unet_trained_3layers/run_.,tag_loss.csv')
df_3layer.head(10)

df_3layer = pd.read_csv(path + '/unet_trained_3layers/run_.,tag_loss.csv')
df_3layer.head(10)


# In[30]:


f(15,4)
smooth=10
plt.plot(mva(df_4layer.Value.values, smooth), label='4 layers')
plt.plot(mva(df_3layer.Value.values, smooth), label='3 layers')
plt.legend()
plt.title('Dice loss')


# In[57]:


path = "/home/gp1514/CNL2/Dropbox/Projects/pw25/ziyang/3d_unet/models/"
modelPath = path + "unet_trained_bk/"


# In[58]:


x = event_accumulator.EventAccumulator(path=modelPath)
x.Reload()


# In[44]:


x.Tags()


# In[54]:


loss = np.array(x.Scalars('loss_1'))
loss_t = loss[:,1]
loss_v = loss[:,2]


# In[56]:


f(15,4)
plt.plot(loss_t,loss_v)


# In[64]:


path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_real_dropout-025/"
xreal = event_accumulator.EventAccumulator(path=modelPath)
xreal.Reload()


# In[65]:


path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_mix_dropout-025/"
xmix = event_accumulator.EventAccumulator(path=modelPath)
xmix.Reload()


# In[9]:


def plot_loss(x, maxiter=400, label=''):
    loss = np.array(x.Scalars('loss_val'))
    loss_t = loss[:,0][:maxiter] - loss[0,0]
    loss_v = loss[:,2][:maxiter]
    loss_t2 = []
    for i in range(len(loss_t)):
        loss_t2.append(i)
    
    plt.plot(loss_t2,loss_v, label=label)


# In[107]:


f(15,4)
plot_loss(xreal, label='real')
plot_loss(xmix, label='real+synth')
plt.legend(fontsize=20)
plt.title('Dice Loss on testing data (4 layers, 16 features, dropout 0.75)')
plt.savefig('/home/gp1514/SSD/dice_loss.pdf')


# In[10]:


path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_mix_dropout-05-f3_l4_148/"
xmix = event_accumulator.EventAccumulator(path=modelPath)
xmix.Reload()

path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_syn_dropout-05-f3_l4_148/"
xsyn = event_accumulator.EventAccumulator(path=modelPath)
xsyn.Reload()

path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_real_dropout-025/"
xreal = event_accumulator.EventAccumulator(path=modelPath)
xreal.Reload()

path = "/home/gp1514/SSD/code/3d_unet/models/"
modelPath = path + "unet_trained_mix_dropout-025/"
xmix0 = event_accumulator.EventAccumulator(path=modelPath)
xmix0.Reload()


# In[12]:


f(15,4)
plot_loss(xsyn, label='syn')
plot_loss(xmix, label='real+synth')
plot_loss(xreal, label='real')
plot_loss(xmix0, label='real+synth 0')
plt.legend(fontsize=20)
plt.title('Dice Loss on testing data (4 layers, 16 features, dropout 0.75)')


# In[ ]:




