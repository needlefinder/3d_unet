
# coding: utf-8

# In[10]:

get_ipython().magic('matplotlib inline')
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import time, sys, yaml, itertools


# In[18]:

def dice_coef(y_true, y_pred):
    y_true_f = tf.flatten(y_true)
    y_pred_f = tf.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def deconv3d(x, W, stride=1):
    x_shape = tf.shape(x)
    print(x.get_shape())
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    print(output_shape.get_shape())
    print(W.get_shape())
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='VALID')

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


class LearningParameters:
    learningRate = 0
    lambda2 = 0
    epochs = 0
    batchSize = 0
    doBatchNorm = 0
    channels = 0
    loss = 0
    continueExp = 0
    tile = 0
    dropout = 0

filter_size=3 

def createUNet(x, filterNumStart, depth, lp):
    # Placeholder for the input image
    batch_size = tf.shape(x)[0]
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    nz = tf.shape(x)[3]
    x_image = tf.reshape(x, tf.stack([-1, nx, ny, nz, lp.channels]))
    in_node = tf.Variable(x_image)

    connections = []
    filterNumStart = 8
    '''
    Going down
    '''
    for d in range(depth):
        input_layer = tf.layers.batch_normalization(in_node)
        # conv1
        conv = tf.layers.conv3d(
            inputs=input_layer,
            filters=filterNumStart * 2 ** d,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu)
        connections.append(conv)

        conv = tf.layers.batch_normalization(conv)
        # conv2
        conv = tf.layers.conv3d(
            inputs=conv,
            filters=filterNumStart * 2 ** d * 2,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu)

        input_layer = tf.layers.batch_normalization(input_layer)

        if d < depth - 1:
            connections.append(conv)
            # pool1
            pool = tf.layers.max_pooling3d(
                inputs=conv,
                pool_size=[2, 2, 2],
                strides=2)
            input_layer = pool

    '''
    Going up
    '''
    ## start deconvoluion
    for d in range(depth - 1):
        features = 2**(depth-d+1)*4
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([2, 2, 2, features//2, features], stddev)
        
        deconv = deconv3d(conv, wd, filterNumStart * 2 ** (depth - d))
        deconv = tf.layers.batch_normalization(deconv)
        concat = tf.concat([connections[-1 - d], deconv])

        conv = tf.layers.conv3d(
            inputs=concat,
            filters=filterNumStart * 2 ** d * 2,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.relu)

        conv = tf.layers.batch_normalization(conv)

    network = tf.layers.conv3d(
        inputs=conv,
        filters=2,
        kernel_size=[1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    sh = network.get_shape()
    return [network, network.get_shape()[2], (lp.tile - sh[2]) // 2]


# In[19]:

def trainNet(train_fn, trainCases, path, savePath, e, lp, outdim, margin):
    print("training....")
    totalSum = [0, 0, 0, 0, 0]
    epochStartTime = time.clock()
    index = 0
    reportPath = savePath + "report.txt"
    path2 = path + trainCases[0] + "/"
    # lesions = sitk.GetArrayFromImage(sitk.ReadImage(path2+"needles.nrrd"))[22:-22,22:-22,22:-22]
    lesions = nrrd.read(path2 + "needles.nrrd")[0]
    sh = lesions.shape
    # fl = nib.load(path2+"fl_unet_3d.nii.gz").get_data()
    # sh_fl = fl.shape
    y = np.zeros((2, sh[0], sh[1], sh[2]), dtype='float32')
    totalD = 0
    featuresBatch = np.zeros(
        (lp.batchSize, len(lp.channels), outdim + 2 * margin, outdim + 2 * margin, outdim + 2 * margin),
        dtype='float32')
    labelsBatch = np.zeros((lp.batchSize, 2, outdim, outdim, outdim), dtype='float32')

    for c in trainCases:
        caseStartTime = time.clock()
        caseSum = [0, 0, 0, 0, 0]
        featureImages = []
        path2 = path + c + "/"
        print("{t})loading images...".format(t=index))
        # y[1] = sitk.GetArrayFromImage(sitk.ReadImage(path2+"needles.nrrd"))[22:-22,22:-22,22:-22]
        y[1] = nrrd.read(path2 + "needles.nrrd")[0]
        y[0] = np.ones_like(y[1]) - y[1]
        if np.count_nonzero(y[1]) == 0:
            print("skipping...")
            continue
        for cc in lp.channels:
            # featureImages.append(nib.load(path2+cc).get_data())
            mri = nrrd.read(path2 + cc)[0]
            mri = mri.astype(np.float32)
            mri -= mri.min()
            mri /= mri.max()
            mri = np.pad(mri, 44, "symmetric")
        featureImages.append(mri)

        print("training....")
        mask = np.ones_like(featureImages[0])

        caseD = 0
        batchIndex = 0
        coords = list(itertools.product(range(0, sh[0], outdim), range(0, sh[1], outdim), range(0, sh[2], outdim)))
        numIter = len(coords)
        shuffle(coords)
        xlim = sh[0] - outdim
        ylim = sh[1] - outdim
        zlim = sh[2] - outdim
        for coord in coords:
            xx = min(coord[0], xlim)
            yy = min(coord[1], ylim)
            zz = min(coord[2], zlim)
            labelsBatch[batchIndex, :, :, :, :] = y[:, xx:xx + outdim, yy:yy + outdim, zz:zz + outdim]
            for ch in range(len(lp.channels)):
                featuresBatch[batchIndex, ch, :, :, :] = featureImages[ch][xx:xx + outdim + 2 * margin,
                                                         yy:yy + outdim + 2 * margin, zz:zz + outdim + 2 * margin]
            batchIndex += 1
            if batchIndex == lp.batchSize:
                perf = train_fn(featuresBatch, labelsBatch)
                caseD += perf[5] * perf[3]
                totalD += perf[5] * perf[3]
                caseSum = [sum(x) for x in zip(perf[:-1], caseSum)]
                totalSum = [sum(x) for x in zip(perf[:-1], totalSum)]
                print(">>>", e, c, coord, ":", perf[0], perf[1], perf[2], perf[3], perf[4], perf[5])
                batchIndex = 0

        caseD = caseD / (caseSum[3] + 0.00001)
        losses = [x * lp.batchSize / numIter for x in caseSum]
        print("==========> patient time", (time.clock() - caseStartTime) / 60.0, losses, caseD)
        report = open(reportPath, "a")
        report.write(str(c) + "," + str(losses[0]) + "," + str(losses[1]) + "," + str(losses[2]) + "," + str(
            losses[3]) + "," + str(caseD) + "," + str(losses[4]) + "\n")
        report.close()
        index += 1
    totalD = totalD / (totalSum[3] + 0.00001)
    losses = [x * lp.batchSize / (numIter * len(trainCases)) for x in totalSum]
    print("epoch time", (time.clock() - epochStartTime) / 60.0, losses, totalD)
    report = open(reportPath, "a")
    report.write(str(e) + "train=====" + str(losses[0]) + "," + str(losses[1]) + "," + str(losses[2]) + "," + str(
        losses[3]) + "," + str(totalD) + "," + str(losses[4]) + "\n")
    report.close()


# In[20]:

def validateNet(valid_fn, validCases, path, savePath, e, lp, outdim, margin):
    print("validating....")
    totalSum = [0, 0, 0, 0, 0]
    epochStartTime = time.clock()
    index = 0
    reportPath = savePath + "report.txt"
    path2 = path + trainCases[0] + "/"
    # lesions = sitk.GetArrayFromImage(sitk.ReadImage(path2+"needles.nrrd"))[22:-22,22:-22,22:-22]
    lesions = sitk.GetArrayFromImage(sitk.ReadImage(path2 + "needles.nrrd"))
    sh = lesions.shape
    # fl = nib.load(path2+"fl_unet_3d.nii.gz").get_data()
    # sh_fl = fl.shape
    y = np.zeros((2, sh[0], sh[1], sh[2]), dtype='float32')
    totalD = 0
    featuresBatch = np.zeros(
        (lp.batchSize, len(lp.channels), outdim + 2 * margin, outdim + 2 * margin, outdim + 2 * margin),
        dtype='float32')
    labelsBatch = np.zeros((lp.batchSize, 2, outdim, outdim, outdim), dtype='float32')

    for c in validCases:
        caseStartTime = time.clock()
        caseSum = [0, 0, 0, 0, 0]
        featureImages = []
        path2 = path + c + "/"
        print("{t})loading images...".format(t=index))
        for cc in lp.channels:
            mri = nrrd.read(path2 + cc)[0]
            mri = mri.astype(np.float32)
            mri -= mri.min()
            mri /= mri.max()
            mri = np.pad(mri, 44, "symmetric")
            featureImages.append(mri)

        y[1] = nrrd.read(path2 + "needles.nrrd")[0]
        y[0] = np.ones_like(y[1]) - y[1]
        print("validating....")
        mask = np.ones_like(featureImages[0])

        caseD = 0
        batchIndex = 0
        coords = list(
            itertools.product(range(0, sh[0] - sh[0] % outdim, outdim), range(0, sh[1] - sh[1] % outdim, outdim),
                              range(0, sh[2] - sh[2] % outdim, outdim)))
        numIter = len(coords)
        shuffle(coords)
        for coord in coords:
            labelsBatch[batchIndex, :, :, :, :] = y[:, coord[0]:coord[0] + outdim, coord[1]:coord[1] + outdim,
                                                  coord[2]:coord[2] + outdim]
            for ch in range(len(lp.channels)):
                featuresBatch[batchIndex, ch, :, :, :] = featureImages[ch][coord[0]:coord[0] + outdim + 2 * margin,
                                                         coord[1]:coord[1] + outdim + 2 * margin,
                                                         coord[2]:coord[2] + outdim + 2 * margin]
            batchIndex += 1
            if batchIndex == lp.batchSize:
                perf = valid_fn(featuresBatch, labelsBatch)
                caseD += perf[5] * perf[3]
                totalD += perf[5] * perf[3]
                caseSum = [sum(x) for x in zip(perf[:-1], caseSum)]
                totalSum = [sum(x) for x in zip(perf[:-1], totalSum)]
                print(">>>", e, c, coord, ":", perf[0], perf[1], perf[2], perf[3], perf[4], perf[5])
                batchIndex = 0

        caseD = caseD / (caseSum[3] + 0.00001)
        losses = [x * lp.batchSize / numIter for x in caseSum]
        print("==========> patient time", (time.clock() - caseStartTime) / 60.0, losses, caseD)
        report = open(reportPath, "a")
        report.write(str(c) + "," + str(losses[0]) + "," + str(losses[1]) + "," + str(losses[2]) + "," + str(
            losses[3]) + "," + str(caseD) + "," + str(losses[4]) + "\n")
        report.close()
        index += 1
    totalD = totalD / (totalSum[3] + 0.00001)
    losses = [x * lp.batchSize / (numIter * len(trainCases)) for x in totalSum]
    print("epoch time", (time.clock() - epochStartTime) / 60.0, losses, totalD)
    report = open(reportPath, "a")
    report.write(str(e) + "validation=====" + str(losses[0]) + "," + str(losses[1]) + "," + str(losses[2]) + "," + str(
        losses[3]) + "," + str(totalD) + "," + str(losses[4]) + "\n")
    report.close()


# In[21]:


print("everything begins...")
rootPath = "/home/gp1514/Dropbox/2016-paolo/preprocessed_data/"
dataPath = rootPath+"LabelMaps_1.00-1.00-1.00/"

trainCases = loadCases("train.txt")
validCases = loadCases("valid.txt")

number = sys.argv[1]
netPath = rootPath+"networks/"+number+"/"
lp = loadLearningParameters("lp.txt")

# ftensor5 = T.TensorType('float32', (False,)*5)

X = tf.Variable(tf.zeros((1,1,1,1,1)))
Y = tf.Variable(tf.zeros((1,1,1,1,1)))

[network, outdim, margin] = createUNet(X, 32, 4, lp)
print("--", margin, outdim)
start = 0
if lp.continueExp !="-":
    print("loading to continue from ",lp.continueExp)
    network = load_parameters(network, netPath+lp.continueExp+".npz")
    print("loaded! :)")
    start = int(lp.continueExp)+1

weight = 0.000138 #findClassImbalance(trainCases, dataPath)
print("++++++++++++++++++++++++++", weight)
print("creating train function...")
save_parameters(network, "initial.npz")
valid_fn = loadValidFunction(X, Y, network, lp, weight)

for e in range(start, lp.epochs):
    print("===============================================now epoch", e)
    train_fn = loadTrainFunction(X, Y, network, lp, weight, e)
    trainNet(train_fn, trainCases, dataPath, netPath, e, lp, outdim, margin)    
    #validateNet(valid_fn, validCases, dataPath, netPath, e, lp, outdim, margin)
    save_parameters(network, netPath+str(e)+".npz")    


# In[22]:

trainCases


# In[15]:

def load_parameters(network, fname):
    with np.load(fname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values(network, param_values)
    return network


def save_parameters(network, fname):
    pass
    #np.savez(fname, *lasagne.layers.get_all_param_values(network))


def loadCases(p):
    f = open(p)
    res = []
    for l in f:
        l = l[:-1]
        if l == "":
            break
        if l[-1] == '\r':
            l = l[:-1]
        res.append(l)
    return res



def findClassImbalance(cases, path):
    totalPos = 0
    totalNeg = 0
    total = 0

    for c in cases:
        path2 = path + c+"/"
        lesions = nib.load(path2+"dawmPos_unet_2d.nii.gz").get_data()
        mask = nib.load(path2+"mask_unet_2d.nii.gz").get_data()
        pos = np.count_nonzero(lesions)
        vox = np.count_nonzero(mask)
        totalPos += pos
        total += vox
        print("Pos:", pos,"proportion:", float(pos)/vox, "totalProp:", float(totalPos)/total)
    return float(totalPos)/(total)


def loadLearningParameters(path):
    f = open(path)
    lp = LearningParameters()
    
    with open("lp.txt", 'r') as stream:
        try:
            par = (yaml.load(stream))
        except yaml.YAMLError as exc:
            print(exc)
    for k, v in par.items():
        setattr(lp, k, v)
    
    print("================================")
    for k,v in par.items():
        print(k + ":"+ str(v))
    
    lp.channels=1

    return lp


# In[ ]:

""

