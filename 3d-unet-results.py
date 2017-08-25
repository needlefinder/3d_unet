
# coding: utf-8

# In[3]:


get_ipython().magic('matplotlib inline')
from fns import *
from syntheticdata import synthetic_generation
# import ipyvolume


# In[4]:


import plotly
plotly.tools.set_credentials_file(username='gpernelle', api_key='4KIdNJBBApIeebdKxyN9')
import plotly.plotly as py
import plotly.graph_objs as go


# In[5]:


def dice(logits, labels):
        flat_logits = logits.flatten()
        flat_labels = labels.flatten()
        intersection = np.sum(flat_logits*flat_labels)
        union = np.sum(flat_logits) + np.sum(flat_labels)
        loss = 1 - 2 * intersection / union
        return loss


# ### Predict

# In[24]:


def predict_case(caseNumber):
    case=str(caseNumber)
    image_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_split2/%s/case.nrrd'%case
    label_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/%s/needles.nrrd'%case

    data, options_ = nrrd.read(image_name)
    data = data.astype(np.float32)
    data1, options = nrrd.read(image_name.replace('_split2', ''))
    data1 = data1.astype(np.float32)

    label_data, options_ = nrrd.read(label_name)

    prediction = predict([data1,data], "./models/unet_trained_mix_dropout-05-clahe30_f32")
    return label_data, prediction, options


def save_case(caseNumber):
    case=str(caseNumber)
    image_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_split2/%s/case.nrrd'%case
    data1, options = nrrd.read(image_name.replace('_split2', ''))
    data1 = data1.astype(np.float32)
    nrrd.write('predictions/case_%s.nrrd'%str(caseNumber), data1, options=options)



# In[18]:


def predict_and_save(caseNumber):
    label_data, prediction, options = predict_case(caseNumber)
    dice(prediction, label_data)
    islands = post_processing(prediction, min_area=int(50), max_residual=float(4))
    nrrd.write('predictions/%s.nrrd'%str(caseNumber), islands, options=options)

def predict_and_plot3d(caseNumber):
    label_data, prediction, options = predict_case(caseNumber)
    dice(prediction, label_data)
    islands = post_processing(prediction, min_area=int(50), max_residual=float(4))
    nrrd.write('predictions/%s.nrrd'%str(caseNumber), islands, options=options)
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
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')


# In[8]:


validationCases = loadCases("preprocessing/validation.txt")
testingCases = loadCases("preprocessing/testing.txt")


# In[9]:


validationCases


# In[10]:


testingCases


# In[15]:


for case in validationCases:
    predict_and_save(case)


# In[16]:


for case in testingCases:
    predict_and_save(case)


# In[25]:


for case in validationCases:
    save_case(case)


# In[27]:


predict_and_plot3d('071')


# In[ ]:




