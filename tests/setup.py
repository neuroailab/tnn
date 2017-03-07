import sys, os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('../../tfutils/master/')
from tfutils import data, model

SEED = 0


def get_mnist_data():
    # mnist variable is an instance of the DataSet class.
    mnist = input_data.read_data_sets('tutorials/MNIST_data/')
    assert mnist.train.images.shape == (55000, 784)
    assert mnist.train.labels.shape == (55000,)
    assert mnist.test.images.shape == (10000, 784)
    assert mnist.test.labels.shape == (10000,)
    assert mnist.validation.images.shape == (5000, 784)
    assert mnist.validation.labels.shape == (5000,)
    return mnist


def mnist_fc(images, labels):
    with tf.variable_scope('fc1'):
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights', shape=[784, 2048], initializer=init)
        init = tf.constant_initializer(.1)
        bias = tf.get_variable('bias', shape=[2048], initializer=init)
        fc1 = tf.nn.relu(tf.matmul(images, weights) + bias)

    with tf.variable_scope('fc2'):
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights', shape=[2048, 10], initializer=init)
        init = tf.constant_initializer(.1)
        bias = tf.get_variable('bias', shape=[10], initializer=init)
        fc2 = tf.matmul(fc1, weights) + bias

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=labels)
    return {'fc1': fc1, 'fc2': fc2, 'loss': tf.reduce_mean(loss)}


def mnist_conv(images, labels):
    with tf.variable_scope('conv1'):
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights', shape=[5, 5, 1, 32], initializer=init)
        init = tf.constant_initializer(0)
        bias = tf.get_variable('bias', shape=[32], initializer=init)
        conv1 = tf.nn.relu(tf.nn.conv2d(images, weights,
                                        strides=[1, 1, 1, 1], padding='SAME') + bias)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv2'):
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights', shape=[5, 5, 32, 64], initializer=init)
        init = tf.constant_initializer(.1)
        bias = tf.get_variable('bias', shape=[64], initializer=init)
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('fc1'):
        pool2_resh = tf.reshape(pool2, [pool2.get_shape().as_list()[0], -1])
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights',
                                    shape=[pool2_resh.get_shape().as_list()[-1], 512],
                                    initializer=init)
        init = tf.constant_initializer(.1)
        bias = tf.get_variable('bias', shape=[512], initializer=init)
        fc1 = tf.nn.relu(tf.matmul(pool2_resh, weights) + bias)

    with tf.variable_scope('fc2'):
        init = tf.truncated_normal_initializer(mean=0, stddev=.1, seed=SEED)
        weights = tf.get_variable(name='weights', shape=[512, 10], initializer=init)
        init = tf.constant_initializer(.1)
        bias = tf.get_variable('bias', shape=[10], initializer=init)
        fc2 = tf.matmul(fc1, weights) + bias

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=labels)
    return {'conv1': pool1, 'conv2': pool2, 'fc1': fc1, 'fc2': fc2, 'loss': tf.reduce_mean(loss)}


def alexnet(images, labels, scope, **kwargs):
    m = model.alexnet(images, seed=SEED, **kwargs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=m.output, labels=labels)
    graph = tf.get_default_graph()
    targets = {
            #    'conv1': graph.get_tensor_by_name('/'.join([scope, 'conv1/pool:0'])),
            #    'conv2': graph.get_tensor_by_name('/'.join([scope, 'conv2/pool:0'])),
            #    'conv3': graph.get_tensor_by_name('/'.join([scope, 'conv3/relu:0'])),
            #    'conv4': graph.get_tensor_by_name('/'.join([scope, 'conv4/relu:0'])),
            #    'conv5': graph.get_tensor_by_name('/'.join([scope, 'conv5/pool:0'])),
            #    'fc6': graph.get_tensor_by_name('/'.join([scope, 'fc6/relu:0'])),
            #    'fc7': graph.get_tensor_by_name('/'.join([scope, 'fc7/relu:0'])),
            #    'fc8': graph.get_tensor_by_name('/'.join([scope, 'fc8/fc:0'])),
               'loss': tf.reduce_mean(loss)
               }
    return targets


def get_imagenet():
    imagenet_path = os.environ.get('IMAGENET_PATH', '/data/imagenet_dataset/imagenet2012.hdf5')
    imagenet = data.ImageNet(imagenet_path, batch_size=256, crop_size=224)
    return imagenet