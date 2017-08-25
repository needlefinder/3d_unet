
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from fns import *
from syntheticdata import synthetic_generation


# In[ ]:





# In[2]:


import tensorflow as tf

from tensorflow_serving.example import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS



  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print 'Please specify a positive value for training iteration.'
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print 'Please specify a positive value for version number.'
    sys.exit(-1)

  # Train model
  print 'Training model...'
  mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in xrange(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices))
  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  print 'training accuracy %g' % sess.run(
      accuracy, feed_dict={x: mnist.test.images,
                           y_: mnist.test.labels})
  print 'Done training!'

'''
EXPORT
'''
export_path_base = sys.argv[-1]
export_path = './export-%d/'%time.time()
print('Exporting trained model to', export_path)

builder = tf.saved_model.builder.SavedModelBuilder(export_path)

# Build the signature_def_map.
serialized_tf_example = tf.placeholder(tf.string, name='tf_example')

feature_configs = {'x': tf.FixedLenFeature(shape=[148,148,148], dtype=tf.float32),}
tf_example = tf.parse_example(serialized_tf_example, feature_configs)
feature_configs_output = {'y': tf.FixedLenFeature(shape=[60,60,60], dtype=tf.bool),}
tf_example_out = tf.parse_example(serialized_tf_example, feature_configs)

x = tf.identity(tf_example['x'], name='x')  # assign name

y = net.predict("./unet_trained/model 99.cpkt", x[np.newaxis,...])[0][:,:,:,0]
tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
tensor_info_y = tf.saved_model.utils.build_tensor_info(y)  # y = unet_predict

prediction_signature = (
  tf.saved_model.signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'prediction': tensor_info_y},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

builder.add_meta_graph_and_variables(
  sess, [tf.saved_model.tag_constants.SERVING],
  signature_def_map={
      'predict_volumes':
          prediction_signature,
      tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          segmentation_signature,
  })

builder.save()


# In[ ]:




