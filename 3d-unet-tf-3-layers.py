
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
from fns import *
from syntheticdata import synthetic_generation


# ## setting up the unet

# In[2]:


net = Unet(channels=1, 
           n_class=1, 
           layers=3, 
           pool_size=2,
           features_root=16, summaries=True,
           #cost_kwargs=dict(regularizer=0.001)
          )


# ## training

# In[3]:


provider = ImageDataProvider(split_vol=True, check_vol=True)
provider.thresh = 400
provider._find_data_files()
count, vals = provider._count_valid_training(provider.training_data_files)


# In[4]:


# plt.hist(vals, 30)
count = np.sum(np.array(vals)>400)
# print(count)


# In[ ]:


data_provider = provider
trainer = Trainer(net, batch_size=3, optimizer="adam")
path = trainer.train(data_provider, 
                     "./unet_trained",
                     training_array = None,
                     validation_array = None,
                     testing_array = None,
                     training_iters=count, 
                     epochs=100, 
                     dropout=0.75, 
                     restore=False,
                     display_step=1)


# ### Predict

# In[8]:


provider = ImageDataProvider()
# testing_data, label_data = provider._load_data_and_label(provider.testing_data_files,3)
testing_data, label_data = provider._load_data_and_label(provider.testing_data_files,3)


# In[6]:


testing_data.shape


# In[7]:


provider.testing_data_files[:]


# In[18]:


i = 2
prediction = net.predict("./unet_trained/model 99.cpkt", testing_data[i][np.newaxis,...])[0][:,:,:,0]


# In[19]:


print(np.unique(prediction, return_counts=True))
print(prediction.shape)
print(label_data.shape)


# In[10]:


net.dice(prediction, label_data[i][44:-44,44:-44,44:-44])


# In[11]:


get_ipython().magic('matplotlib notebook')
xs,ys,zs = np.where(prediction == 1)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, marker='o', alpha=0.3, s=5)
plt.show()

# fig = plt.figure(figsize=(6,6))
# ax = fig.add_subplot(111, projection='3d')
xs,ys,zs = np.where(label_data[i, 44:-44,44:-44,44:-44,0] == 1)
ax.scatter(xs, ys, zs, marker='o',color='g', alpha=0.1, s=5)
plt.show()


# In[24]:


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


# In[25]:


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


# In[6]:


image_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/075/case.nrrd'
label_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/075/needles.nrrd'
# img = provider._load_file(image_name, np.float32, padding="noise")
label = provider._load_file(label_name, np.bool, padding="zero")


# In[7]:


# print(img.shape)
# print(data.shape)


# In[8]:


# data.shape


# In[9]:


tiles = (148,148,148)
tile = 148


# In[10]:


data, options = nrrd.read(image_name)
data = data.astype(np.float32)
print(data.shape)
d = data.resize(max(data.shape[0],tile),
               max(data.shape[1],tile),
               max(data.shape[2],tile))
print(data.shape)
print(options)


# ## Inference pipeline

# In[11]:


arr_data = cutVolume(data)
arr_pred = predict_full_volume(net, arr_data, model_path="./unet_trained/model 99.cpkt")
full_pred = recombine(arr_pred, data)


# In[111]:


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


# In[112]:


islands = post_processing(full_pred)


# In[113]:


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


# In[75]:


print(np.unique(islands, return_counts=True))


# In[76]:


# islands = measure.label(full_pred)
K = np.max(islands)
cp = sns.color_palette("Set2", K)
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for j in range(1,K):
    xs,ys,zs = np.where(islands == j)
    ax.scatter(xs, ys, zs, marker='o',color=cp[j], alpha=0.9, s=2)
plt.show()


# In[13]:


nrrd.write('test75.nrrd', islands, options=options)


# In[15]:


islands.shape
print(np.unique(islands, return_counts=True))


# In[13]:


image_name = '/mnt/DATA/gp1514/Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/074/needles.nrrd'
data, options = nrrd.read(image_name)
data = data.astype(np.int8)


# In[38]:


arr_labels = cutVolume(data)


# In[25]:


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




