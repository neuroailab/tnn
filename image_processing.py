""" Use python threading and hdf5provider to get Imagenet images and labels
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import hdf5provider
import numpy as np

# image information
TOTAL_IMGS_HDF5 = 1290129
IMAGE_SIZE_ORIG = 256
TRAIN_SIZE = 1000000
NUM_CHANNELS = 3
PIXEL_DEPTH = 255


def inputs(train, data_path, crop_size, batch_size, num_validation_batches=0):
    """
    Creates tf.FIFOQueue to supply images and labels. Only does center crop 
    to crop_sizexcrop_size and normalization to [-0.5, 0.5]
    :param train: True-get inputs from train_slice, False-get from eval_slice
    :param data_path: Path to Imagenet data
    :param crop_size: spatial size of images returned (crop_size <=
    IMAGE_SIZE_ORIG; if crop_size is less, then center crop is taken). If
    crop_size == None, then it is taken to be IMAGE_SIZE_ORIG
    :param batch_size: Training or evaluation batch size (int)
    :param num_validation_batches: If train==False, total number of batches
    to evaluate
    :return: data provider, enqueue_op (to start queue runners),
    images_batch and labels_batch, tf Tensors for input to model
    """

    if train:
        slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool)
        slice[:TRAIN_SIZE] = True
    else:
        _N = batch_size * num_validation_batches
        slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool)
        slice[TRAIN_SIZE: TRAIN_SIZE + _N] = True
    if crop_size is None:
        crop_size = IMAGE_SIZE_ORIG
    assert crop_size <= IMAGE_SIZE_ORIG
    data = ImageNetSingle(subslice=slice, data_path=data_path,
                          crop_size=crop_size)
    # center cropped
    q = tf.FIFOQueue(capacity=batch_size,
                     dtypes=[tf.float32, tf.int64],
                     shapes=[(crop_size,
                     crop_size, NUM_CHANNELS), []])
    enqueue_op = q.enqueue([data.data_node, data.labels_node])
    images_batch, labels_batch = q.dequeue_many(batch_size)

    return data, enqueue_op, images_batch, labels_batch


class ImageNetSingle(hdf5provider.HDF5DataProvider):
    def __init__(self, subslice, data_path, crop_size, *args, **kwargs):
        """
        :param subslice: np array for training or eval slice
        :param data_path: path to imagenet data
        :param crop_size: for center crop (crop_size x crop_size)
        """
        super(ImageNetSingle, self).__init__(data_path,
                         ['data', 'labels'],
                         1,
                         subslice=subslice,
                         preprocess={'labels': hdf5provider.get_unique_labels},
                         postprocess={'data': self.postproc},
                         pad=True)

        self.data_node = tf.placeholder(tf.float32,
                                shape=(crop_size,
                                crop_size, 3), name='data')
        self.labels_node = tf.placeholder(tf.int64, shape=[], name='labels')
        self.crop_size = crop_size

    def postproc(self, ims, f):
        norml = lambda x: (x - (PIXEL_DEPTH /
                                2.0)) / PIXEL_DEPTH
        r = (IMAGE_SIZE_ORIG - self.crop_size) // 2
        if r == 0:  # no cropping needed
            return norml(ims).reshape((ims.shape[0], NUM_CHANNELS,
                   IMAGE_SIZE_ORIG,
                   IMAGE_SIZE_ORIG)).swapaxes(1, 2).swapaxes(2, 3)
        else: # center crop
            return norml(ims).reshape((ims.shape[0], NUM_CHANNELS,
                   IMAGE_SIZE_ORIG,
                   IMAGE_SIZE_ORIG)).swapaxes(1, 2).swapaxes(2, 3)[:,
                   r:r + self.crop_size,
                   r:r + self.crop_size]

    def next(self):
        batch = super(ImageNetSingle, self).getNextBatch()
        feed_dict = {self.data_node: batch['data'][0].astype(np.float32),
                     self.labels_node: batch['labels'][0].astype(np.int64)}
        return feed_dict

    def load_and_enqueue(self, sess, enqueue_op, coord):
        while not coord.should_stop():
            batch = self.next()
            sess.run(enqueue_op, feed_dict=batch)

