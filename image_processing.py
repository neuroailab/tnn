""" Use python threading and hdf5provider to get Imagenet images and labels
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hdf5provider
import numpy as np


def inputs(train, params):
    """
    Creates tf.FIFOQueue to supply imgaes and labels.
    :param train: True: get inputs from train_slice, False: get from eval_slice
    :param params: Dictionary of model parameters
    :return:
    """
    if train:
        slice = np.zeros(params['total_imgs_hdf5']).astype(np.bool)
        slice[:params['train_size']] = True
    else:
        _N = params['eval_batch_size'] * params['num_validation_batches']
        slice = np.zeros(params['total_imgs_hdf5']).astype(np.bool)
        slice[params['train_size']: params['train_size'] + _N] = True
    data = ImageNetSingle(subslice=slice, params=params)  # TRAINING data
    q = tf.FIFOQueue(capacity=params['batch_size'],
                     dtypes=[tf.float32, tf.int64],
                     shapes=[(params['image_size_crop'],
                     params['image_size_crop'], params['num_channels']), []])
    enqueue_op = q.enqueue([data.data_node, data.labels_node])
    images_batch, labels_batch = q.dequeue_many(params['batch_size'])
    return data, enqueue_op, images_batch, labels_batch


class ImageNetSingle(hdf5provider.HDF5DataProvider):
    def __init__(self, subslice, params, *args, **kwargs):
        super(ImageNetSingle, self).__init__(params['data_path'],
                         ['data', 'labels'],
                         1,
                         subslice=subslice,
                         preprocess={'labels': hdf5provider.get_unique_labels},
                         postprocess={'data': self.postproc},
                         pad=True)

        self.data_node = tf.placeholder(tf.float32,
                                shape=(params['image_size_crop'],
                                params['image_size_crop'], 3), name='data')
        self.labels_node = tf.placeholder(tf.int64, shape=[], name='labels')
        self.params = params

    def postproc(self, ims, f):
        params = self.params
        norml = lambda x: (x - (params['pixel_depth'] /
                                2.0)) / params['pixel_depth']
        r = (params['image_size_orig'] - params['image_size_crop']) // 2
        if r == 0:  # no cropping
            return norml(ims).reshape((ims.shape[0], params['num_channels'],
                   params['image_size_orig'],
                   params['image_size_orig'])).swapaxes(1, 2).swapaxes(2, 3)
        else:
            return norml(ims).reshape((ims.shape[0], params['num_channels'],
                   params['image_size_orig'],
                   params['image_size_orig'])).swapaxes(1, 2).swapaxes(2, 3)[:,
                   r:r + params['image_size_crop'],
                   r:r + params['image_size_crop']]

    def next(self):
        # batch = super(ImageNetSingle, self).next()
        batch = super(ImageNetSingle, self).getNextBatch()
        feed_dict = {self.data_node: batch['data'][0].astype(np.float32),
                     self.labels_node: batch['labels'][0].astype(np.int64)}
        return feed_dict

    def load_and_enqueue(self, sess, enqueue_op, coord):
        while not coord.should_stop():
            batch = self.next()
            sess.run(enqueue_op, feed_dict=batch)
