""" Use to generate model config files """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, json


if os.uname()[1].startswith('node'):  # OpenMind
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
    # run configuration parameters
    'num_epochs': 90,  # number of epochs to train
    'log_device_placement': False,  # if variable placement has to be logged
    'train': True,
    'features_layer': None,

    # add evaluation parameters
    'eval': True,  # should evaluation be done during training
    # 'eval_avg_crop': False,  # only relevant during eval
    'num_train_batches': num_train_images // batch_size,
    'num_valid_batches': num_valid_images // batch_size,  # number of batches used in validation

    'saver': {
        'dbname': 'bypass-test',
        'collname': 'test',
        'port': 31001,
        'exp_id': 'bypass_15_test',

        # saving path
        'save_path': restore_var_file + 'output/model',  # file name; # NOTE: if you use another directory make sure it exists first.
        'save_loss': True,  # Write training loss to file SAVE_PATH + '_loss.csv'
        'save_loss_freq': 5,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_vars': True,  # save variables if True
        'save_vars_freq': 3000,  # how often to save vars (divisble by 10)
        'max_to_keep': 10,

        # restoring variables from file
        'restore_vars': False,  # If True, restores variables from RESTORE_VAR_FILE
        'start_step': 27000,  # to be used for step counter.
        'restore_var_file': restore_var_file,  # if SAVE_PATH is given
                    #  through argparse, RESTORE_VAR_FILE is based on that save_path

        # tensorboard
        'tensorboard': True,  # use tensorboard for graph visualization
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
        'layer_sizes': {
            0: {'state': [batch_size, image_size_crop, image_size_crop, num_channels],
                'output': [batch_size, image_size_crop, image_size_crop, num_channels]},
            1: {'state': [batch_size, image_size_crop // 4, image_size_crop // 4, 96],
                'output': [batch_size, image_size_crop // 8, image_size_crop // 8, 96]},
            2: {'state': [batch_size, image_size_crop // 8, image_size_crop // 8, 256],
                'output': [batch_size, image_size_crop // 16, image_size_crop // 16, 256]},
            3: {'state': [batch_size, image_size_crop // 16, image_size_crop // 16, 384],
                'output': [batch_size, image_size_crop // 16, image_size_crop // 16, 384]},
            4: {'state': [batch_size, image_size_crop // 16, image_size_crop // 16, 384],
                'output': [batch_size, image_size_crop // 16, image_size_crop // 16, 384]},
            5: {'state': [batch_size, image_size_crop // 16, image_size_crop // 16, 256],
                'output': [batch_size, image_size_crop // 32, image_size_crop // 32, 256]},
            6: {'state': [batch_size, 4096],
                'output': [batch_size, 4096]},
            7: {'state': [batch_size, 4096],
                'output': [batch_size, 4096]},
            8: {'state': [batch_size, 1000],
                'output': [batch_size, 1000]}
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

outfile = 'params.json'
with open(outfile, 'w') as f:
    json.dump(params, f)




# LAYER_SIZES = {
#     0: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP, IMAGE_SIZE_CROP, NUM_CHANNELS],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP, IMAGE_SIZE_CROP,
#                    NUM_CHANNELS]},  # input
#     1: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP / 4, IMAGE_SIZE_CROP / 4, 96],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP / 8, IMAGE_SIZE_CROP / 8, 96]},
#     # stride2 conv AND pool!
#     2: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP / 8, IMAGE_SIZE_CROP / 8, 256],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16,
#                    256]},  # convpool
#     3: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16, 384],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16,
#                    384]},  # conv
#     4: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16, 384],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16,
#                    384]},  # conv
#     5: {'state': [BATCH_SIZE, IMAGE_SIZE_CROP / 16, IMAGE_SIZE_CROP / 16, 256],
#         'output': [BATCH_SIZE, IMAGE_SIZE_CROP / 32, IMAGE_SIZE_CROP / 32,
#                    256]},  # convpool
#     6: {'state': [BATCH_SIZE, 1, 1, 4096], 'output': [BATCH_SIZE, 4096]},  # fc
#     7: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 4096]},  # fc
#     8: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 1000]},  # fc
# }



# def get_layers(train):
#     """ LAYERS depends on whether we use training or eval (ex: for dropout)"""
#     if not train:
#         fc_keep_prob = None  # no dropout for eval mode
#     else:
#         fc_keep_prob = FC_KEEP_PROB

#     layers = {1: ['ConvPool', {'state_size': LAYER_SIZES[1]['state'],
#                                'output_size': LAYER_SIZES[1]['output'],
#                                'conv_size': 11,  # kernel size for conv
#                                'conv_stride': 4,  # stride for conv
#                                'weight_decay': WEIGHT_DECAY,  # None for none
#                                'lrn': True,  # use local response norm
#                                # Note: LRN parameters in ConvRNN.py
#                                'pool_size': 3,
#                                # kernel size for pool (defaults
#                                # to = stride determined by layer sizes.),
#                                'decay_param_init': DECAY_PARAM_INIT,
#                                # (relevant if you have memory)
#                                'memory': MEMORY}],
#               2: ['ConvPool', {'state_size': LAYER_SIZES[2]['state'],
#                                'output_size': LAYER_SIZES[2]['output'],
#                                'conv_size': 5,  # kernel size for conv
#                                'conv_stride': 1,  # stride for conv
#                                'weight_decay': WEIGHT_DECAY,  # None for none
#                                'lrn': True,  # use local response norm
#                                'pool_size': 3,  # kernel size for pool
#                                'decay_param_init': DECAY_PARAM_INIT,
#                                'memory': MEMORY}],
#               3: ['Conv', {'state_size': LAYER_SIZES[3]['state'],
#                            'conv_size': 3,  # kernel size for conv
#                            'conv_stride': 1,  # stride for conv
#                            'weight_decay': WEIGHT_DECAY,  # None for none
#                            # kernel size for pool
#                            'decay_param_init': DECAY_PARAM_INIT,
#                            # (relevant if you have memory)
#                            'memory': MEMORY}],
#               4: ['Conv', {'state_size': LAYER_SIZES[4]['state'],
#                            'conv_size': 3,  # kernel size for conv
#                            'conv_stride': 1,  # stride for conv
#                            'weight_decay': WEIGHT_DECAY,  # None for none
#                            'decay_param_init': DECAY_PARAM_INIT,
#                            'memory': MEMORY}],
#               5: ['ConvPool', {'state_size': LAYER_SIZES[5]['state'],
#                                'output_size': LAYER_SIZES[5]['output'],
#                                'conv_size': 3,  # kernel size for conv
#                                'conv_stride': 1,  # stride for conv
#                                'weight_decay': WEIGHT_DECAY,  # None for none
#                                'pool_size': 3,
#                                'decay_param_init': DECAY_PARAM_INIT,
#                                'memory': MEMORY}],
#               6: ['FC', {'state_size': LAYER_SIZES[6]['state'],
#                          'keep_prob': fc_keep_prob,
#                          'memory': False}],
#               7: ['FC', {'state_size': LAYER_SIZES[7]['state'],
#                          'keep_prob': fc_keep_prob,
#                          'memory': False}]
#               }
#     return layers
