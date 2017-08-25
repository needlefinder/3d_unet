
# coding: utf-8

# In[72]:


get_ipython().magic('matplotlib inline')
import plotly
from plotly import tools
from fns import *
from syntheticdata import synthetic_generation
plotly.tools.set_credentials_file(username='gpernelle', api_key='4KIdNJBBApIeebdKxyN9')
import plotly.offline as offline

plotly.offline.init_notebook_mode()


# ## Training

# In[ ]:


trainer = Trainer(batch_size=2, optimizer="adam", filter_size=3, layers=4)
path = trainer.train("./models/unet_trained_mix_dropout-05-clahe30_f32",
                     synth=0,
                     freeze_deep_layers=0,
                     training_array = None,
                     validation_array = None,
                     testing_array = None,
                     training_iters=400, 
                     epochs=300, 
                     dropout=0.5, 
                     restore=True,
                     display_step=1)


# In[3]:


def dice(logits, labels):
        flat_logits = logits.flatten()
        flat_labels = labels.flatten()
        intersection = np.sum(flat_logits*flat_labels)
        union = np.sum(flat_logits) + np.sum(flat_labels)
        loss = 1 - 2 * intersection / union
        return loss


# ## Inference

# In[4]:


case='72'
image_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_split2/0%s/case.nrrd'%case
label_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/0%s/needles.nrrd'%case

data, options = nrrd.read(image_name)
data = data.astype(np.float32)
data1, options = nrrd.read(image_name.replace('_split2', ''))
data1 = data1.astype(np.float32)

label_data, options = nrrd.read(label_name)

prediction = predict([data1,data], "./models/unet_trained_mix_dropout-05-clahe30_f32")


# In[5]:


dice(prediction, label_data)


# In[30]:


islands = post_processing(prediction, min_area=int(100), max_residual=float(4))
# islands.shape


# In[95]:


import plotly.plotly as py
import plotly.graph_objs as go
x,y,z = np.where(label_data == 1)
xs,ys,zs = np.where(islands != 0)

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=3,
        line=dict(
            color='green',
            width=2
        ),
        opacity=0.3
    )
)

trace2 = go.Scatter3d(
    x=xs,
    y=ys,
    z=zs,
    mode='markers',
    marker=dict(
        color='red',
        size=3,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.3
    )
)
data = [trace1, trace2]
layout = go.Layout(
    title="72",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=100
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='72')


# In[81]:





# In[ ]:




