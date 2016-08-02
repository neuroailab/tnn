from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hdf5provider
import bypassrnn_params as params
import threading
import numpy as np
""" Instead of using tf's queue runner and their tfrecords, we use python threading and hdf5provider.
"""

def inputs(train):
    """ train = True: get from train_slice
        train = False: get from eval_slice """""
    if train:
        slice = np.zeros(params.TOTAL_IMGS_HDF5).astype(np.bool);
        slice[:params.TRAIN_SIZE] = True
    else:
        _N = params.EVAL_BATCH_SIZE * params.NUM_VALIDATION_BATCHES  # eval_batch_size is set same as batch_size for modeling ease
        slice = np.zeros(params.TOTAL_IMGS_HDF5).astype(np.bool)
        slice[params.TRAIN_SIZE: params.TRAIN_SIZE + _N] = True
    data = ImageNetSingle(subslice=slice)  # TRAINING data
    q = tf.FIFOQueue(capacity=params.BATCH_SIZE,  # having batch_size * n doesn't help interestingly (or not)
                     dtypes=[tf.float32, tf.int64],
                     shapes=[(params.IMAGE_SIZE, params.IMAGE_SIZE, params.NUM_CHANNELS), []])
    enqueue_op = q.enqueue([data.data_node, data.labels_node])
    images_batch, labels_batch = q.dequeue_many(params.BATCH_SIZE)
    return data, enqueue_op, images_batch, labels_batch #(with all these returns might as well just not use a separate method) ._.


class ImageNetSingle(hdf5provider.HDF5DataProvider):
    def __init__(self, subslice, *args, **kwargs):
        super(ImageNetSingle, self).__init__(params.DATA_PATH,
                                             ['data', 'labels'],
                                             1,
                                             subslice=subslice,
                                             preprocess={'labels': hdf5provider.get_unique_labels},
                                             postprocess={'data': self.postproc},
                                             pad=True)

        self.data_node = tf.placeholder(tf.float32,
                                        shape=(params.IMAGE_SIZE, params.IMAGE_SIZE, 3), name='data')
        self.labels_node = tf.placeholder(tf.int64, shape=[], name='labels')

    def postproc(self, ims, f):
        norml = lambda x: (x - (params.PIXEL_DEPTH / 2.0)) / params.PIXEL_DEPTH
        r = (params.IMAGE_SIZE_ORIG - params.IMAGE_SIZE) // 2
        if r == 0: # no cropping
            return norml(ims).reshape((ims.shape[0], params.NUM_CHANNELS, params.IMAGE_SIZE_ORIG, params.IMAGE_SIZE_ORIG)).swapaxes(1,2).swapaxes(2, 3)
        else:
            return norml(ims).reshape((ims.shape[0], params.NUM_CHANNELS, params.IMAGE_SIZE_ORIG, params.IMAGE_SIZE_ORIG)).swapaxes(1,2).swapaxes(2, 3)[:,
                               r:r + params.IMAGE_SIZE, r:r + params.IMAGE_SIZE]

    def next(self):

        #batch = super(ImageNetSingle, self).next()
        batch = super(ImageNetSingle, self).getNextBatch()
        feed_dict = {self.data_node: batch['data'][0].astype(np.float32),
                     self.labels_node: batch['labels'][0].astype(np.int64)}
        return feed_dict

    def load_and_enqueue(self, sess, enqueue_op, coord):
        while not coord.should_stop():
            batch = self.next()
            sess.run(enqueue_op, feed_dict=batch)
