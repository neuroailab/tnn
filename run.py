""" Use to generate model config files """

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tfutils

import model


host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    # data_path = '/mindhive/dicarlolab/common/imagenet/data.raw'
    data_path = '/om/user/qbilius/imagenet/data.raw'
    restore_var_file = '/mindhive/dicarlolab/u/qbilius/computed/tconvnet_test/'
else:  # agents
    data_path = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'
    restore_var_file = '/home/qbilius/mh17/computed/tconvnet_test/'

batch_size = 128
image_size_crop = 224
num_train_images = 2**20
num_valid_images = 2**14



def _get_data(train=False,
              data_path='',
              image_size_crop=224,
              num_train_images=2**20,
              num_valid_images=2**14,
              ):
    if train:
        subslice = range(num_train_images)
    else:
        subslice = range(num_train_images, num_train_images + num_valid_images)

    with tf.device("/gpu:0"):
        d = tfutils.data.ImageNet(data_path,
                                  subslice=subslice,
                                  crop_size=image_size_crop)
    return d


def imagenet_train(**params):
    return _get_data(train=True, **params)


def imagenet_validation(**params):
    return _get_data(train=False, **params)


def in_top_k(inputs, outputs, target, topn=1, **kwargs):
    return [tf.nn.in_top_k(out, inputs[target], topn) for out in outputs]


def exponential_decay(global_step,
                      learning_rate=.01,
                      decay_factor=.95,
                      decay_steps=1,
                      ):
    # Decay the learning rate exponentially based on the number of steps.
    if decay_factor is None:
        lr = learning_rate  # just a constant.
    else:
        # Calculate the learning rate schedule.
        lr = tf.train.exponential_decay(
            learning_rate,  # Base learning rate.
            global_step,  # Current index into the dataset.
            decay_steps,  # Decay step
            decay_factor,  # Decay rate.
            staircase=True)
    return lr


params = {
    'saver_params': {
        'host': 'localhost',
        'port': 31001,
        'dbname': 'bypass-test',
        'collname': 'test',
        'exp_id': 'bypass_15_test',

        'restore': False,
        'save': False,
        'save_initial': False,
        'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 3000,
        'save_filters_freq': 30000,
        'cache_filters_freq': 3000,
        'cache_dir': None,
        'tensorboard_dir': restore_var_file + 'output',  # save tensorboard graph
        'force_fetch': False
    },

    'model_params': {
        'func': model.get_model,
        'seed': 2,
        'model_base': model.alexnet,
        'bypasses': [],#(1,5)],  # bypasses: list of tuples (from, to)
        'init_weights': 'xavier',
        'weight_decay': .0005,  # None for no decay
        'dropout': .5,  # for training; config writes None for eval mode
        'memory_decay': None,  # just for Conv or ConvPool layers; float to use memory
                          # Note: default weights, strides, etc -> adjust in ConvRNN.py
                           # -1.1 initialize decay_factor t= sigmoid(-1.1) = 0.25
        'memory_trainable': False,

        'trim_top': True,
        'trim_bottom': True,  # might not need to use, since we don't expect that
            # bottom nodes that don't contribute to final loss will matter. And we also
            # may want to access all layer's outputs or states at time T.
        'features_layer': 'FC8',
        'bypass_pool_kernel_size': None,  # None => kernel size = stride size,
                                          # for pooling between bypass layers.
        'input_spatial_size': image_size_crop,
        'input_seq_len': 8,  # length of input sequence
        'target': 'data'
    },

    'train_params': {
        'data': {
            'func': imagenet_train,
            'data_path': data_path,  # path to image database
            'image_size_crop': image_size_crop,  # size after cropping an image
            'num_train_images': num_train_images,
            'num_valid_images': num_valid_images,
        },
        # 'targets': {
        # }
    },

    'loss_params': {
        'func': model.get_loss,
        'target': 'labels',
        'loss_per_case_func': tf.nn.sparse_softmax_cross_entropy_with_logits,
        'agg_func': tf.reduce_mean,
        'time_penalty': 1.2,  # 'gamma' time penalty as # time steps passed increases
    },

    'learning_rate_params': {
        'func': exponential_decay,
        'learning_rate': .01,  # .001 for Adam. initial learning rate.
        'decay_factor': .95,
        'decay_steps': num_train_images // batch_size,  # exponential decay each epoch
    },

    'optimizer_params': {
        'func': tfutils.optimizer.ClipOptimizer,
        'optimizer_class': tf.train.MomentumOptimizer,
        'clip': True,
        'momentum': .9
    },

    'validation_params': {
        'top1': {
            'data': {
                'func': imagenet_validation,
                'data_path': data_path,  # path to image database
                'image_size_crop': image_size_crop,  # size after cropping an image
                'num_train_images': num_train_images,
                'num_valid_images': num_valid_images,
            },
            'targets': {
                'func': in_top_k,
                'target': 'labels',
                'topn': 5
            }
        },
        'top5': {
            'data': {
                'func': imagenet_validation,
                'data_path': data_path,  # path to image database
                'image_size_crop': image_size_crop,  # size after cropping an image
                'num_train_images': num_train_images,
                'num_valid_images': num_valid_images,
            },
            'targets': {
                'func': in_top_k,
                'target': 'labels',
                'topn': 5
            }
        },
    },

    'queue_params': {
        'queue_type': 'random',
        'batch_size': batch_size,
        'n_threads': 4,
        'seed': 2,
    },
    'thres_loss': 1000,
    'num_steps': 90 * num_train_images,  # number of steps to train
    'log_device_placement': False,  # if variable placement has to be logged

}


if __name__ == '__main__':
    tfutils.base.get_params()
    tfutils.base.run_base(**params)