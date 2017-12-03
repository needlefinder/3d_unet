from fns.utils import *
from fns.functions import *
from fns.dataprovider import *
import time
import cv2

def _process_data(data, clahe_filter=True):
        data = np.clip(np.fabs(data), -np.inf, np.inf)
        # normalization
        data -= np.amin(data)
        data /= np.amax(data)
        
        # data -= np.mean(data)
        # data /= np.std(data)
        return data
    
    
def _load_file(path, dtype=np.float32):
        data = np.load(path).astype(dtype)[..., np.newaxis]
        return data
    
    
def load_data(min_iter=np.inf, mode='training', synth=0, tile=60):
    '''
    mode: training, testing or validation
    '''
    if synth:
        synth_str = '_synth'
    else:
        synth_str = ''


    with open('preprocessing/%s_subvolumes%s.txt' % (mode, synth_str), 'r') as f:
        files = f.read().splitlines()
        
    if mode=='training':
        with open('preprocessing/%s_subvolumes_synth.txt' % (mode), 'r') as f:
            files += f.read().splitlines()
        
    
        
    # with open('preprocessing/%s_subvolumes%s.txt' % (mode,synth_str), 'r') as f:
    #     files = f.read().splitlines()
    

    data, target = [], []
    for i in trange(min(min_iter,len(files))):
        image_path = files[i]

        label_name = image_path.replace('_case', '_labelmap')
        label_name = label_name.replace('case', 'needles')

        img_clahe = _load_file(image_path, np.float32)
        img_clahe = _process_data(img_clahe)
        
        image_path = image_path.replace('_split2', '')
        img = _load_file(image_path, np.float32)
        img = _process_data(img)
        
        # print(img.shape, img_clahe.shape)
        imgs = np.concatenate([img, img_clahe], axis=3)
        # print(imgs.shape)


        annotation = _load_file(label_name, np.float32)
        annotation = crop_to_shape2(annotation, (tile,tile,tile))[...,np.newaxis]

        data.append(imgs)
        target.append(annotation)
    return np.array(data), np.array(target), len(files)

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
            

class Trainer(object):
    """
    Trains a unet instance

    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    def __init__(self, batch_size=1, filter_size=3, layers=4, optimizer="momentum", opt_kwargs={}):
        self.prediction_path = "prediction"
        self.batch_size = batch_size
        self.filter_size = filter_size
        self.opt_kwargs = opt_kwargs
        self.layers = layers
        self.n_class = 1

    def _initialize(self, training_iters, output_path, restore):
        init = tf.global_variables_initializer()
        prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(prediction_path))
            shutil.rmtree(prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(prediction_path):
            logging.info("Allocating '{:}'".format(prediction_path))
            os.makedirs(prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

        ########################################################################################################



    def train(self, output_path, training_array=None, validation_array=None, testing_array=None, \
              training_iters=50, epochs=100, dropout=0.75, display_step=1, restore=False, synth=0, freeze_deep_layers=0):
        """
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        """

        keep_prob = 0.5
        channels = 2
        n_class = 1
        
        with tf.Session() as sess:
            init = self._initialize(training_iters, output_path, restore)
            
            # training dataset
            if self.layers == 3:
                tile = 108
            elif self.layers == 4:
                tile = 60
            features, labels, nfiles = load_data(training_iters, synth=synth, tile=tile)
            features_placeholder = tf.placeholder(features.dtype, features.shape, name='features_placeholder')
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name='features_placeholder')
            
            train_dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
            train_dataset = train_dataset.shuffle(buffer_size=1000)
            train_dataset = train_dataset.repeat()  # Repeat the input indefinitely.
            train_dataset = train_dataset.batch(self.batch_size)
            
            #testing
            features_test, labels_test, nfiles_testing = load_data(100, mode='testing', synth=synth)
            features_test_placeholder = tf.placeholder(features_test.dtype, features_test.shape, name='features_test_placeholder')
            labels_test_placeholder = tf.placeholder(labels_test.dtype, labels_test.shape, name='labels_test_placeholder')
            
            test_dataset = tf.contrib.data.Dataset.from_tensor_slices((features_test_placeholder, labels_test_placeholder))
            test_dataset = test_dataset.repeat()  # Repeat the input indefinitely.
            test_dataset = test_dataset.batch(self.batch_size)


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
            # predicter, variables, offset = create_conv_net(tf.zeros((3,148,148,148,1)), keep_prob, channels, n_class, 
                                                           # layers=self.layers, features_root=16, filter_size=self.filter_size, 
                                                           # pool_size=2, summaries=True)
            
            # print(predicter.shape, offset)
            predicter, variables, offset = create_conv_net(next_example, keep_prob, channels, n_class, 
                                                           layers=self.layers, filter_size=self.filter_size, 
                                                           pool_size=2, summaries=True)
            
            #---------------------------
            # prediction
            predicter_label = tf.cast(predicter >= 0.5, tf.float32)
            correct_pred = tf.cast(tf.equal(tf.reshape(predicter_label, [-1, n_class]), 
                                            tf.reshape(next_label, [-1, n_class])), tf.float32)
            accuracy = tf.reduce_mean(correct_pred)
            self.cost = _get_cost(predicter, next_label)
            
            #----------------------------
            # freeze deeper layers after training on synthetic data
            tvars = tf.trainable_variables()
            tvars = tf.trainable_variables()
            if freeze_deep_layers:
                layer_down_0 = [var for var in tvars if 'going_up' in var.name]
            else:
                layer_down_0 = None
            
            
            #----------------------------
            # optimizer op
            training_op = tf.train.AdamOptimizer(learning_rate=tf.Variable(0.0001)).minimize(self.cost,
                                                                                            global_step=tf.Variable(0),
                                                                                           #var_list=layer_down_0,
                                                                                           )

            self.accuracy = accuracy
            self.predicter_label = predicter_label
            sess.run(tf.global_variables_initializer())

            #---------------------------
            # summary
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.restore(sess, ckpt.model_checkpoint_path)

            logging.info("Start optimization")
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    loss_step, _ = sess.run([self.cost,training_op], feed_dict={handle: train_iterator_handle})
                    total_loss += loss_step
                    logging.info("Iter {:}, Minibatch Loss= {:.4f}".format(step, loss_step))
                    summary_loss = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss_step),])
                    summary_writer.add_summary(summary_loss, step)
                    summary_writer.flush()
                    if step % 50 == 0:
                        #--------------------------
                        # validation
                        loss_val = 0
                        for step_val in range(nfiles_testing):
                            loss_step_val = sess.run([self.cost], feed_dict={handle: test_iterator_handle})
                            logging.info("Iter {:}, Testing Loss= {:.4f}".format(step_val, np.mean(loss_step_val)))
                            loss_step += loss_step_val
                        #summary_loss_val = tf.Summary(value=[tf.Summary.Value(tag="loss_val", simple_value=loss_step/nfiles_testing),])
                        #summary_writer.add_summary(summary_loss_val, step)
                        summary_writer.flush()

                save_path = os.path.join(output_path, "model {}.cpkt".format(epoch))
                save_path = self.save(sess, save_path)

            logging.info("Optimization Finished!")

            return save_path


    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.predicter)
        pred_shape = prediction.shape

        loss, acc = sess.run((self.cost, self.accuracy))

        logging.info("Validation Accuracy= %.4f, Validation Loss= %.4f" % (acc, loss))

        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, Learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step):
        # Calculate batch loss and accuracy
        summary_str, loss, acc = sess.run([self.summary_op, self.cost, self.accuracy])
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}".format(step,loss,acc))
        
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)
        
        
def predict(data, model_path="./unet_trained/model 6.cpkt"):
    print(50 * "-")
    print("Loading and Preparing the data")
    arr = []
    for d in data:
        arr.append(cutVolume(d)[..., np.newaxis])
    arr = np.concatenate(arr, axis=4)
    print(50 * "-")
    print("Starting the segmenter.")
    arr_pred = predict_full_volume2(arr, model_path=model_path)
    print("Merging subvolumes")
    full_pred = recombine(arr_out=arr_pred, data=data[0])
    return full_pred
    
    
def predict_full_volume2(arr_data, model_path="./unet_trained/model 6.cpkt", off=44):
    '''
    Perform inference on subvolumes
    '''
    arr_out = []
    imgs = []
    for i in trange(arr_data.shape[0]):
        data = arr_data[i,...]
        
        #input shape size required 1,148,148,148,1 
        # data = data[...,np.newaxis]
        
        imgs.append(data)
        
    imgs = np.array(imgs)
    print('imgs shape', imgs.shape)
    
    outs = predict_multiple2(imgs, model_path)
    
    for out in outs:
        out = out[0][:,:,:,0]
        out_p = np.pad(out,((off,off),(off,off),(off,off)), mode='constant', constant_values=[0])
        arr_out.append(out_p)
    return arr_out



def predict_multiple2(features, model_path, layers=4, filter_size=3):
    """
    Uses the model to create a prediction for the given data

    :param model_path: path to the model checkpoint to restore
    :param x_test: Data to predict on. Shape [n, nx, ny, nz, channels]
    :returns prediction: The unet prediction Shape [n, px, py, pz, labels] (px=nx-self.offset/2)
    """
    keep_prob = 0.5
    channels = 2
    n_class = 1
    tf.reset_default_graph()
    with tf.Session() as sess:
        #---------------------------
        # dataset
        features = np.array(features)
        labels = np.empty((features.shape[0],60,60,60,1), dtype=np.float32)
        print('Feature shape: ', features.shape)
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        dataset = dataset.batch(1)
        
        #---------------------------
        # iterator
        iterator = dataset.make_initializable_iterator()
        next_example, next_label = iterator.get_next()
        
        #---------------------------
        # unet
        predicter, variables, offset = create_conv_net(next_example, keep_prob, channels, n_class, 
                                                           layers=layers, filter_size=filter_size, 
                                                           pool_size=2, summaries=True)

        #---------------------------
        # prediction
        predicter_label = tf.cast(predicter >= 0.5, tf.float32)
        correct_pred = tf.cast(tf.equal(tf.reshape(predicter_label, [-1, n_class]), 
                                        tf.reshape(next_label, [-1, n_class])), tf.float32)
        cost = _get_cost(predicter, next_label)
        group_init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(group_init_ops)        
        sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                      labels_placeholder: labels})
        
        #---------------------------
        # restore weights
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            last_saved_model = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, last_saved_model)
            logging.info("Model restored from file: %s" % model_path)
        else:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            logging.info("Model restored from file: %s" % model_path)
            
                    
        #---------------------------
        # predictions
        predictions = []
        for i in trange(features.shape[0]):           
            pred = sess.run(predicter_label, feed_dict={next_example: features[i][np.newaxis, ...]})
            predictions.append(pred)
            
        print("prediction shape:", np.array(predictions).shape)

    return predictions        
        
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
    
    logging.info("Layers: {layers}, FeaturesRoot: {features_root}, ConvolutionSize: {filter_size}*{filter_size}*{filter_size}, PoolingSize: {pool_size}*{pool_size}*{pool_size}"\
                 .format(layers=layers, features_root=features_root, filter_size=filter_size, pool_size=pool_size))
    
    in_node = x
    batch_size = tf.shape(x)[0] 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    if filter_size==3:
        in_size = 144
    elif filter_size==5:
        in_size = 140
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
                                                                                size=dw_h_convs[layer].get_shape()))
                
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
