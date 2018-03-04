import os
import scipy
import numpy as np
import scipy.io as spio
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



def load_notMNIST(batch_size, is_training=True):
    path = os.path.join('data', 'notMNIST')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        print("loaded")
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch



def load_affNIST(batch_size, is_training=True):
    
    if is_training:

        path = 'data/affNIST/train/training_and_validation.mat'
        dataset = loadmat(path)

        ans_set = dataset['affNISTdata']['label_int']
        train_set = dataset['affNISTdata']['image'].transpose()/255.0
        # print ('train_set',train_set.shape)# (60000, 1600)
        # print ('label_set',ans_set.shape)#(60000,)

        trX = train_set[:55000]
        print('输入的Tensor-shape',trX.shape)
        trX=trX.reshape((55000,40,40,1))
        trX=trX.astype(np.float32)

        trY = ans_set[:55000]
        trY=trY.astype(np.int32)

        valX = train_set[55000:, ]
        valX=valX.reshape((5000,40,40,1))
        valX=valX.astype(np.float32)

        valY = ans_set[55000:]
        valY=valY.astype(np.int32)

        num_tr_batch = 55000 // batch_size
        num_val_batch = 5000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        path = 'data/affNIST/test/1.mat'
        dataset = loadmat(path)

        ans_set = dataset['affNISTdata']['label_int']
        test_set = dataset['affNISTdata']['image'].transpose()/255.0
        teX=test_set.reshape((10000,40,40,1)).astype(np.float)
        
        # teX = loaded[16:].reshape((10000, 40, 40, 1)).astype(np.float)

        # fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        # loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = ans_set.reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'notMNIST':
        return load_notMNIST(batch_size, is_training)
    elif dataset == 'affNIST':
        return load_affNIST(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    if dataset == 'notMNIST':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_notMNIST(batch_size, is_training=True)
    elif dataset == 'affNIST':
        trX, trY, num_tr_batch, valX, valY, num_val_batch = load_affNIST(batch_size, is_training=True)
    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs
