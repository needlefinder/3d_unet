
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
from fns import *
from syntheticdata import synthetic_generation
# import ipyvolume


# In[ ]:


import plotly
plotly.tools.set_credentials_file(username='gpernelle', api_key='4KIdNJBBApIeebdKxyN9')


# ## setting up the unet

# ## training

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


# ### Predict

# In[14]:


case='74'
image_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_split2/0%s/case.nrrd'%case
label_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/0%s/needles.nrrd'%case

data, options = nrrd.read(image_name)
data = data.astype(np.float32)
data1, options = nrrd.read(image_name.replace('_split2', ''))
data1 = data1.astype(np.float32)

label_data, options = nrrd.read(label_name)

prediction = predict([data1,data], "./models/unet_trained_mix_dropout-05-clahe30_f16")


# In[15]:


# image_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_synthetic_occulted_new/numpy/training_case831.npy'
# label_name = '/home/gp1514/SSD/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00_synthetic_occulted_new/numpy/training_labelmap831.npy'
# data = np.load(image_name)
# label_data = np.load(label_name)


# In[16]:


dice(prediction, label_data)


# In[12]:


islands = post_processing(prediction, min_area=int(100), max_residual=float(10))
# islands.shape


# In[13]:


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
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[ ]:


# %matplotlib notebook
# xs,ys,zs = np.where(prediction == 1)

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs, ys, zs, marker='o', alpha=0.1, s=1)
# plt.show()

# # fig = plt.figure(figsize=(6,6))
# # ax = fig.add_subplot(111, projection='3d')
# xs,ys,zs = np.where(label_data == 1)
# ax.scatter(xs, ys, zs, marker='o',color='g', alpha=0.1, s=5)
# plt.show()


# In[48]:


import ipyvolume.pylab as p3
p3.figure(figsize=(15,15))
ipyvolume.quickvolshow(islands, opacity=1)
ipyvolume.quickvolshow(label_data, opacity=1)


# In[35]:


x,y,z = np.where(label_data == 1)
ipyvolume.quickscatter(x,y,z, size=0.5, marker="sphere", opacity=0.2)

x,y,z = np.where(islands == 1)
ipyvolume.quickscatter(x,y,z, size=0.5, marker="sphere", opacity=0.2, color='blue')


# In[ ]:


np.random.seed(1)
islands = measure.label(prediction)
K = np.max(islands)
cp =sns.color_palette("Set2", K)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for j in range(1,K):
    xs,ys,zs = np.where(islands == j)
    ax.scatter(xs, ys, zs, marker='o',color=cp[j], alpha=0.9, s=2)
plt.show()


# In[ ]:


np.random.seed(1)
islands = measure.label(label_data[i,...,0][44:-44,44:-44,44:-44])
K = np.max(islands)
cp =sns.color_palette("Set2", K)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for j in range(1,K):
    xs,ys,zs = np.where(islands == j)
    ax.scatter(xs, ys, zs, marker='o',color=cp[j], alpha=0.9, s=2)
plt.show()


# In[ ]:


image_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/075/case.nrrd'
label_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/075/needles.nrrd'


# In[ ]:


# print(img.shape)
# print(data.shape)


# In[ ]:


# data.shape


# In[ ]:


tiles = (148,148,148)
tile = 148


# In[ ]:


data, options = nrrd.read(image_name)
data = data.astype(np.float32)
print(data.shape)
d = data.resize(max(data.shape[0],tile),
               max(data.shape[1],tile),
               max(data.shape[2],tile))
print(data.shape)
print(options)


# ## Inference pipeline

# In[ ]:


arr_data = cutVolume(data)
arr_pred = predict_full_volume(net, arr_data, model_path="./unet_trained/model 99.cpkt")
full_pred = recombine(arr_pred, data)


# In[ ]:


def post_processing(full_pred, min_area=150, max_residual=10):
    ''' Clustering + removing small clusters + keeping only line-looking clusters'''
    islands_ = measure.label(full_pred)
    regions = measure.regionprops(islands_)
    islands = np.zeros_like(full_pred, dtype=np.uint8)
    K = len(regions)
    print('Number of regions: %d' % K)
    i=0
    for k in range(K):
        region = regions[k]
        coords = region.coords
        if region.area > min_area:
            lm = measure.LineModelND()
            lm.estimate(coords)
            res = lm.residuals(coords)
            mean_res = np.mean(res)
            if mean_res < max_residual:
                i+=1
                print(k, i, mean_res, np.std(res), region.area)
                for x,y,z in coords:
                    islands[x,y,z] = i


# In[ ]:


islands = post_processing(full_pred)


# In[ ]:


islands_ = measure.label(full_pred)
regions = measure.regionprops(islands_)
region = regions[691]
lm = measure.LineModelND()
lm.estimate(region.coords)
res = lm.residuals(region.coords)
res
K = np.max(islands)
cp = sns.color_palette("Set2", K)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

xs,ys,zs = region.coords.T
ax.scatter(xs, ys, zs, marker='o',color=cp[j], alpha=0.9, s=2)
plt.show()


# In[ ]:


print(np.unique(islands, return_counts=True))


# In[ ]:


# islands = measure.label(full_pred)
K = np.max(islands)
cp = sns.color_palette("Set2", K)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for j in range(1,K):
    xs,ys,zs = np.where(islands == j)
    ax.scatter(xs, ys, zs, marker='o',color=cp[j], alpha=0.9, s=2)
plt.show()


# In[ ]:


nrrd.write('test75.nrrd', islands, options=options)


# In[ ]:


islands.shape
print(np.unique(islands, return_counts=True))


# In[ ]:


image_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/074/needles.nrrd'
data, options = nrrd.read(image_name)
data = data.astype(np.int8)


# In[ ]:


arr_labels = cutVolume(data)


# In[ ]:


# '''
# EXPORT
# '''

# with tf.Graph().as_default():
#         # Inject placeholder into the graph
#         serialized_tf_example = tf.placeholder(tf.string, name='input_image')
#         feature_configs = {'data': tf.FixedLenFeature(shape=[1,148,148,148,1], dtype=tf.float32),}
#         tf_example = tf.parse_example(serialized_tf_example, feature_configs)
#         x_test = tf_example['data']
#         # now the image shape is (1,148,148,148,1)

#         # Create UNET model
#         net = Unet(channels=1, n_class=1, layers=4, pool_size=2, features_root=16, summaries=True)
#         # Create saver to restore from checkpoints
#         saver = tf.train.Saver()

#         with tf.Session() as sess:
            
#             # Restore the model from last checkpoints
#             saver.restore(sess, "./unet_trained/model 99.cpkt")
            
            
#             y_dummy = np.empty((1,148,148,148,1))

#             # (re-)create export directory
#             export_path = './export/'
#             if os.path.exists(export_path):
#                 shutil.rmtree(export_path)

#             # create model builder
#             builder = tf.saved_model.builder.SavedModelBuilder(export_path)

#             # create tensors info
#             predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(x_test)
#             predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(net.predicter_label)

#             # build prediction signature
#             prediction_signature = (
#                 tf.saved_model.signature_def_utils.build_signature_def(
#                     inputs={'images': predict_tensor_inputs_info},
#                     outputs={'scores': predict_tensor_scores_info},
#                     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
#                 )
#             )

#             # save the model
#             legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#             builder.add_meta_graph_and_variables(
#                 sess, [tf.saved_model.tag_constants.SERVING],
#                 signature_def_map={
#                     'predict_images': prediction_signature
#                 },
#                 legacy_init_op=legacy_init_op)

#             builder.save()

# print("Successfully exported UNET model")


# In[ ]:




