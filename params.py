""" Use to generate model config files """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json

host = os.uname()[1]
if host.startswith('node') or host == 'openmind7':  # OpenMind
    # DATA_PATH = '/mindhive/dicarlolab/common/imagenet/data.raw'
    data_path = '/om/user/qbilius/imagenet/data.raw'
    restore_var_file = '/mindhive/dicarlolab/u/qbilius/computed/bypass_test/'
else:  # agents
    data_path = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'
    restore_var_file = '/home/qbilius/mh17/computed/bypass_15_test/'

batch_size = 256
image_size_crop = 224
num_channels = 3
num_train_images = 2**20
num_valid_images = 2**14

params = {
    'seed': None,
    'thres_loss': 10000,
    # run configuration parameters
    'num_epochs': 90,  # number of epochs to train
    'log_device_placement': False,  # if variable placement has to be logged
    # 'train': True,
    'features_layer': None,

    # add evaluation parameters
    'eval': True,  # should evaluation be done during training
    # 'eval_avg_crop': False,  # only relevant during eval
    'num_train_batches': num_train_images // batch_size,
    'num_valid_batches': num_valid_images // batch_size,  # number of batches used in validation

    'saver': {
        'save': False,
        'dbname': 'bypass-test',
        'collname': 'test',
        'port': 31001,
        'exp_id': 'bypass_15_test',

        'save_metrics_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 3000,
        'cache_filters_freq': 3000,
        'cache_path': None,
        'save_filters_freq': 30000,
        'tensorboard_dir': None,
        # restoring variables from file
        'restore': False,  # If True, restores variables from RESTORE_VAR_FILE
        'force_fetch': False,
        'start_step': 0,  # to be used for step counter.
        'end_step': None,

        # tensorboard
        #'tensorboard': False,  # use tensorboard for graph visualization
                              # (not activations)
        'tensorboard_dir': restore_var_file + 'output',  # save tensorboard graph
    },

    'data': {  # image parameters
        'data_path': data_path,  # path to image database
        'image_size_crop': image_size_crop,  # size after cropping an image
        'random_crop': True,  # only relevant during training
        # 'num_labels': 1000,  # number of unique labels in the dataset
        'num_threads': 4,  # per tower
        'batch_size': batch_size,  # to be split among GPUs. For train, eval
        'num_train_images': num_train_images,
        'num_valid_images': num_valid_images,
    },

    'model': {
        # 'num_channels': num_channels,  # RGB, fixed number (see image_processing.py)
        # todo - specify alexnet?
        },
        # 'layer_sizes': LAYER_SIZES,
        'weight_decay': .0005,  # None for no decay
        'init_weights': 'xavier',
        'dropout': .5,  # for training; config writes None for eval mode
        'memory_decay': 0,  # just for Conv or ConvPool layers; float to use memory
                          # Note: default weights, strides, etc -> adjust in ConvRNN.py
                           # -1.1 initialize decay_factor t= sigmoid(-1.1) = 0.25

        # bypass parameters
        'T_tot': 8,  # Total number of time steps to run model
            # Note: if all of input is  desired to be run; then T = length(input) +
            # N_cells + 1. Otherwise, T >= shortest path is given and input is
            # truncated to length = T - shortest_path
        # 'layers': LAYERS,
        'bypasses': [(1,5)],  # bypasses: list of tuples (from, to)
        # 'initial_states': None,  # dictionary of initial states for cells
        # 'input_seq': None,  # Sequence of input images. If None, just repeats input image
        'trim_top': True,
        'trim_bottom': True,  # might not need to use, since we don't expect that
            # bottom nodes that don't contribute to final loss will matter. And we also
            # may want to access all layer's outputs or states at time T.
        'bypass_pool_kernel_size': None,  # None => kernel size = stride size,
                                          # for pooling between bypass layers.
    },
    'loss': {
        # loss function parameter
        'time_penalty': 1.2,  # 'gamma' time penalty as # time steps passed increases
    },
    'optimizer': {  # optimization parameters
        'grad_clip': False,
        'momentum': .9,  # for momentum optimizer
    },
    'learning_rate': {
        'learning_rate_base': .03,  # .001 for Adam. initial learning rate.
        'learning_rate_decay_factor': .85,
        'num_epochs_per_decay': 1,  # exponential decay each epoch
    }
}
