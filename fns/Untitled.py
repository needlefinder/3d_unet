
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
from fns.utils import *
from fns.functions import *
from fns.dataprovider import *
import time


# In[32]:


keep_prob = 0.75
channels = 1
n_class = 1
training_iters = 6
batch_size = 3
epochs = 10
with tf.Session() as sess:

    # training dataset
    features, labels = load_data(training_iters)
    features_placeholder = tf.placeholder(features.dtype, features.shape, name='features_placeholder')
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='features_placeholder')

    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    train_dataset = train_dataset.repeat()  # Repeat the input indefinitely.
    train_dataset = train_dataset.batch(batch_size)

    #testing
    features_test, labels_test = load_data(training_iters, mode='testing')
    features_test_placeholder = tf.placeholder(features_test.dtype, features_test.shape, name='features_test_placeholder')
    labels_test_placeholder = tf.placeholder(labels_test.dtype, labels_test.shape, name='labels_test_placeholder')

    test_dataset = tf.contrib.data.Dataset.from_tensor_slices((features_test_placeholder, labels_test_placeholder))
    test_dataset = test_dataset.repeat()  # Repeat the input indefinitely.
    test_dataset = test_dataset.batch(batch_size)


    # define iterators
    train_iterator = train_dataset.make_initializable_iterator()
    print("Feature shape:", features.shape)
    sess.run(train_iterator.initializer, feed_dict={features_placeholder: features,
                                  labels_placeholder: labels})
    train_iterator_handle = sess.run(train_iterator.string_handle())
    print(train_iterator_handle)

    test_iterator = test_dataset.make_initializable_iterator()
    sess.run(test_iterator.initializer, feed_dict={features_test_placeholder: features_test,
                                  labels_test_placeholder: labels_test})
    test_iterator_handle = sess.run(train_iterator.string_handle())
    print(test_iterator_handle)
    print(train_iterator.output_types)

    # define handles
    handle = tf.placeholder(tf.string, shape=[], name='handle_placeholder')
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_example, next_label = iterator.get_next()

    logging.info("label shape %s"%str(next_label.get_shape()))

    #---------------------------
    # unet
    predicter, variables, offset = create_conv_net(next_example, keep_prob, channels, n_class, 
                                                   layers=4, features_root=16, filter_size=3, 
                                                   pool_size=2, summaries=True)
    #---------------------------
    # prediction
    predicter_label = tf.cast(predicter >= 0.5, tf.float32)
    correct_pred = tf.cast(tf.equal(tf.reshape(predicter_label, [-1, n_class]), 
                                    tf.reshape(next_label, [-1, n_class])), tf.float32)
    accuracy = tf.reduce_mean(correct_pred)
    cost = _get_cost(predicter, next_label)

    #----------------------------
    # optimizer op
    training_op = tf.train.AdamOptimizer(learning_rate=tf.Variable(0.001)).minimize(cost,
                                                                                    global_step=tf.Variable(0))

    accuracy = accuracy
    predicter_label = predicter_label
    sess.run(tf.global_variables_initializer())

    #---------------------------
    # summary
    summary_writer = tf.summary.FileWriter('../test/', graph=sess.graph)
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

#     if restore:
#         ckpt = tf.train.get_checkpoint_state(output_path)
#         if ckpt and ckpt.model_checkpoint_path:
#             restore(sess, ckpt.model_checkpoint_path)

    logging.info("Start optimization")
    for epoch in range(epochs):
        total_loss = 0
        for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
            loss_step, _ = sess.run([cost,training_op], feed_dict={handle: train_iterator_handle})
            total_loss += loss_step
            logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss_step))
            summary_loss = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss_step),])
            summary_writer.add_summary(summary_loss, step)
            summary_writer.flush()
            if step % 10 == 0:
                #--------------------------
                # validation
                loss_val = 0
                for step_val in range(3):
                    loss_step_val = sess.run([cost], feed_dict={handle: test_iterator_handle})
                    logging.info("Iter {:}, Testing Loss= {:.4f}".format(step_val, np.mean(loss_step_val)))
                    loss_step += loss_step_val
                summary_loss_val = tf.Summary(value=[tf.Summary.Value(tag="loss_val", simple_value=loss_step/3),])
                summary_writer.add_summary(summary_loss_val, step)
                summary_writer.flush()
                    


        #output_epoch_stats(epoch, total_loss, training_iters, 0.001)

        #store_prediction(sess, val_x, val_y, "epoch_%s" % epoch)

        save_path = os.path.join(output_path, "model {}.cpkt".format(epoch))
        save_path = save(sess, save_path)

    logging.info("Optimization Finished!")


# In[21]:


def output_minibatch_stats(sess, summary_writer, step, summary_op, cost, accuracy, train_iterator_handle):
        # Calculate batch loss and accuracy
        summary_str = sess.run([summary_op], feed_dict={handle: train_iterator_handle})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

def _process_data(data):
        data = np.clip(np.fabs(data), -np.inf, np.inf)
        # normalization
        # data -= np.amin(data)
        # data /= np.amax(data)
        data -= np.mean(data)
        data /= np.std(data)
        return data
    
    
def _load_file(path, dtype=np.float32, padding=None):
        data = nrrd.read(path)[0].astype(dtype)
        return data

def load_data(min_iter=np.inf, mode='training', synth=""):
    '''
    mode: training, testing or validation
    '''
    files = []
    if synth:
        synth_str = '_synth'
    else:
        synth_str = ''

    # with open('preprocessing/training_subvolumes_synth.txt', 'r') as f:
    #     files_training = f.read().splitlines()
    # with open('preprocessing/validation_subvolumes_synth.txt', 'r') as f:
    #     files_validation = f.read().splitlines()
    # with open('preprocessing/testing_subvolumes_synth.txt', 'r') as f:
    #     files_testing = f.read().splitlines()   


    with open('../preprocessing/%s_subvolumes%s.txt' % (mode,synth), 'r') as f:
        files = f.read().splitlines()
    

    data, target = [], []
    for i in trange(min(min_iter,len(files))):
        image_path = files[i]

        label_name = image_path.replace('_case', '_labelmap')
        label_name = label_name.replace('case', 'needles')

        img = _load_file(image_path, np.float32, padding="noise")
        img = _process_data(img)[...,np.newaxis]

        annotation = _load_file(label_name, np.float32, padding="zero")
        annotation = crop_to_shape2(annotation, (60,60,60))[...,np.newaxis]

        data.append(img)
        target.append(annotation)
    return np.array(data), np.array(target)

def _get_cost(predicter, labels):
    """
    Constructs the cost function dice_coefficient.
    """
    with tf.name_scope('cost_function'):
        logging.info('*' * 50)
        logging.info('getting cost')
        logging.info("Logits: {}".format(predicter.get_shape()))
        logging.info("Y: {}".format(labels.get_shape()))
        flat_predicter = tf.reshape(predicter, [-1, 1], name='flat_predicter_reshape')
        flat_labels = tf.reshape(labels, [-1, 1], name='flat_labels_reshape')

        intersection = tf.reduce_sum(flat_predicter * flat_labels)
        union = tf.reduce_sum(flat_predicter) + tf.reduce_sum(flat_labels)
        loss = 1 - 2 * intersection / union
    return loss
            


# In[4]:


def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, 
                    pool_size=2, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,nz,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers: {layers}, FeaturesRoot: {features_root}, ConvolutionSize: {filter_size}*{filter_size}*{filter_size}, PoolingSize: {pool_size}*{pool_size}*{pool_size}"                 .format(layers=layers, features_root=features_root, filter_size=filter_size, pool_size=pool_size))
    
    in_node = x
    batch_size = tf.shape(x)[0] 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    in_size = 144
    size = in_size

   
    # down layers
    with tf.name_scope('going_down'):
        for layer in range(0, layers):
            with tf.name_scope('layer_down_%d'%layer):
                features = 2**layer*features_root
                stddev = 1 / (filter_size**3 * features)
                if layer == 0:
                    w1 = weight_variable([filter_size, filter_size, filter_size, channels, features], stddev)
                else:
                    w1 = weight_variable([filter_size, filter_size, filter_size, features//2, features], stddev)
                w2 = weight_variable([filter_size, filter_size, filter_size, features, features], stddev)
                b1 = bias_variable([features])
                b2 = bias_variable([features])
                
                conv1 = conv3d(in_node, w1, keep_prob)
                tmp_h_conv = tf.nn.elu(conv1 + b1)
                conv2 = conv3d(tmp_h_conv, w2, keep_prob)
                dw_h_convs[layer] = tf.nn.elu(conv2 + b2)
                
                logging.info("Down Convoltion Layer: {layer} Size: {size}".format(layer=layer,size=dw_h_convs[layer].get_shape()))
                
                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))

                size -= 4    
                if layer < layers-1:
                    pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                    in_node = pools[layer]
                    size /= 2    
        
    in_node = dw_h_convs[layers-1]
        
    # up layers
    with tf.name_scope('going_up'):
        for layer in range(layers-2, -1, -1):   
            with tf.name_scope('layer_up_%d'%layer):
                features = 2**(layer+1)*features_root
                stddev = 1 / (filter_size**3 * features)

                wd = weight_variable_devonc([pool_size, pool_size, pool_size, features//2, features], stddev)
                bd = bias_variable([features//2])
                h_deconv = tf.nn.elu(deconv3d(in_node, wd, pool_size) + bd)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)    
                deconv[layer] = h_deconv_concat

                w1 = weight_variable([filter_size, filter_size, filter_size, features, features//2], stddev)
                w2 = weight_variable([filter_size, filter_size, filter_size, features//2, features//2], stddev)
                b1 = bias_variable([features//2])
                b2 = bias_variable([features//2])

                conv1 = conv3d(h_deconv_concat, w1, keep_prob)
                h_conv = tf.nn.elu(conv1 + b1)
                conv2 = conv3d(h_conv, w2, keep_prob)
                in_node = tf.nn.elu(conv2 + b2)
                up_h_convs[layer] = in_node
                
                logging.info("Up Convoltion Layer: {layer} Size: {size}".format(layer=layer,
                                                                                size=tf.shape(dw_h_convs[layer])))
                
                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))

                size *= 2
                size -= 4

    # Output Map
    with tf.name_scope('output_map'):
        #stddev = 1 / (features_root)
        weight = weight_variable([1, 1, 1, features_root, 1], stddev)
        bias = bias_variable([1])
        conv = conv3d(in_node, weight, tf.constant(1.0))
        output_map = tf.nn.sigmoid(conv + bias)
        up_h_convs["out"] = output_map
        logging.info("Output map shape {size}, offset {offset}".format(size=output_map.get_shape(), offset=int(in_size-size)))

        if summaries:
            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%03d"%k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])

        variables = []
        for w1,w2 in weights:
            variables.append(w1)
            variables.append(w2)

        for b1,b2 in biases:
            variables.append(b1)
            variables.append(b2)
        
        variables.append(weight)
        variables.append(bias)
        
    return output_map, variables, int(in_size - size)



# In[ ]:




