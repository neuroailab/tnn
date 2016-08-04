from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ConvRNN import ConvRNNCell, ConvPoolRNNCell, FcRNNCell


# Image parameters
DATA_PATH = '/mindhive/dicarlolab/common/imagenet/data.raw' # for openmind runs
#DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw' # for agent runs

IMAGE_SIZE_ORIG = 256
IMAGE_SIZE = 256 # 224 What we crop it to.
NUM_CHANNELS = 3
PIXEL_DEPTH = 255 # from 0 to 255 (WHY NOT 256?)
NUM_LABELS = 1000 
TOTAL_IMGS_HDF5 = 1290129

# Training parameters
TRAIN_SIZE = 1000000 #1000000
NUM_EPOCHS = 80

BATCH_SIZE = 256 # to be split among GPUs. For train, eval
# Run configuration parameters (TODO REMOVE IF NOT NEEDED ANYMORE (with hdf5 guy))
NUM_PREPROCESS_THREADS = 4 # per tower. should be multiple of 4
NUM_READERS = 4 # parallel readers during training
INPUT_QUEUE_MEMORY_FACTOR = 4 # size of queue of preprocessed images. If 16 OOM, try 4, 2, 1
LOG_DEVICE_PLACEMENT = False # SO WE DONT GET SO MUCH OUTPUT, FOR NOW #whether to log device placement
NUM_GPUS = 1
TOWER_NAME = 'tower'

# Evaluation parameters
NUM_VALIDATION_BATCHES = 30
EVAL_BATCH_SIZE = BATCH_SIZE
EVAL_INTERVAL = 5 #60 * 5 # seconds between eval runs
EVAL_RUN_ONCE = False # run eval only once
#EVAL_DIR = '/om/user/mrui/tf_imagenet_val' # Where to write logs
EVAL_INTERMED = False # should set false if training is true. Set True if you care about looking at predictions at intermediate time points.

# Paths for saving things
CHECKPOINT_DIR = '/om/user/mrui/bypass/outputs/' # for eval to read
SAVE_FILE = CHECKPOINT_DIR + 'omind_trial' # file name base. NOTE: if you use another directory make sure it exists first.

# Save training loss and validation error to: SAVE_FILE + [_loss.csv, or _val.csv]
SAVE_LOSS = True
SAVE_LOSS_FREQ = 5 # keeps loss from every SAVE_LOSS_FREQ steps.

# Saving model parameters (variables)
SAVE_VARS = True # save variables if True
SAVE_VARS_FREQ = 3000 # save variables every SAVE_FREQ steps (Note: will need to coordinate with EVAL_INTERVAL
# (note cont). wrt timing and # steps ... .. # PLEASE MAKE DIVISIBLE BY 10!!
MAX_TO_KEEP = 5

# Restoring variables from file
RESTORE_VARS = False #If True, restores variables from VAR_FILE instead of initializing from scratch
START_STEP = 17500 # to be used for step counter. If RESTORE_VARS=False, we start with 1.
RESTORE_VAR_FILE = SAVE_FILE+ '-' + str(START_STEP) # location of checkpoint file

# Graph parameters. Note:  default weight sizes, strides, decay factor -> adjust in ConvRNN.py
GRAD_CLIP = False
KEEP_PROB= None # a hack for dropout-> applied to input to final fc/softmax layer during training.
# NOTE: Since we are only training for now, we can specify keep_prob with the graph structure. (see FC_KEEP_PROB)

# loss function parameters
TIME_PENALTY = 1.2 # 'gamma' time penalty as # time steps passed increases

# Optimization parameters
LEARNING_RATE_BASE = 0.05 # .001 for Adam. initial learning rate. (we'll use exponential decay) [0.05 in conv_img_cat.py tutorial, .01 elsewhere]
LEARNING_RATE_DECAY_FACTOR = 0.95
MOMENTUM = 0.9 # for momentum optimizer
NUM_EPOCHS_PER_DECAY = 1 # decay each epoch (just mult 0.95 factor)w

T = 8
# Graph structure
# sizes = [batch size, spatial, spatial, depth(num_channels)]

LAYER_SIZES = { 0: {'state': [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], 'output':  [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]}, # input
                1: {'state': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 96], 'output': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 96]}, # stride2 conv AND pool!
                2: {'state': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 256], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 256]}, # convpool
                3: {'state': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384]},# conv
                4: {'state': [BATCH_SIZE,  IMAGE_SIZE/16, IMAGE_SIZE/16, 384], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384]}, # conv
                5: {'state': [BATCH_SIZE,  IMAGE_SIZE/16, IMAGE_SIZE/16, 256], 'output': [BATCH_SIZE, IMAGE_SIZE/32, IMAGE_SIZE/32, 256]}, # convpool
                6: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 4096]}, # fc
                7: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 4096]},  # fc
                }
# Todo - for non-default values (strides, filter sizes, etc.) make nicer input format ..
####### template for entry in layers ########
# 1: [ConvPoolRNNCell, {'state_size': LAYER_SIZES[1]['state'], 'output_size': LAYER_SIZES[1]['output']
#                       'conv_size': 3,  # kernel size for conv
#                       'conv_stride': 1  # stride for conv
#                       # 'weight_init': use default (currently 'xavier')
#                       # 'weight_stddev': only relevant if you use 'trunc_norm' initialization
#                       # 'bias_init': use default (currently 0.1),
#                       'pool_size': 2,  # kernel size for pool (defaults to = stride determined by layer sizes.)
#                       # 'decay_param_init': relevant if you have memory
#                       'memory': False}] # defaults to True (then uses decay_param_init)
WEIGHT_DECAY = 0.0005
FC_KEEP_PROB = 0.5
LAYERS = {1: [ConvPoolRNNCell, {'state_size':LAYER_SIZES[1]['state'], 'output_size': LAYER_SIZES[1]['output'],
                               'conv_size': 11, # kernel size for conv
                                'conv_stride': 4, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0, # TODO Set factor to 0.5 (param = 0) first? (relevant if you have memory)
                                'memory': False}],
         2: [ConvPoolRNNCell, {'state_size': LAYER_SIZES[2]['state'], 'output_size': LAYER_SIZES[2]['output'],
                               'conv_size': 5, # kernel size for conv
                                'conv_stride': 2, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0, # TODO Set factor to 0.5 (param = 0) first? (relevant if you have memory)
                                'memory': False}],
         3: [ConvRNNCell, {'state_size': LAYER_SIZES[3]['state'],
                                'conv_size': 3,  # kernel size for conv
                                'conv_stride': 1,  # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                                'decay_param_init': 0, # TODO Set factor to 0.5 (param = 0) first? (relevant if you have memory)
                                'memory': False}],
         4: [ConvRNNCell, {'state_size': LAYER_SIZES[4]['state'],
                           'conv_size': 3,  # kernel size for conv
                           'conv_stride': 1,  # stride for conv
                            'weight_decay': WEIGHT_DECAY, # None for none
                           'decay_param_init': 0, # TODO Set factor to 0.5 (param = 0) first? (relevant if you have memory)
                           'memory': False}],
         5: [ConvPoolRNNCell, {'state_size':LAYER_SIZES[5]['state'], 'output_size': LAYER_SIZES[5]['output'],
                               'conv_size': 3, # kernel size for conv
                                'conv_stride': 1, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0, # TODO Set factor to 0.5 (param = 0) first? (relevant if you have memory)
                                'memory': False}],
         6: [FcRNNCell, {'state_size': LAYER_SIZES[6]['state'],
                         'keep_prob': FC_KEEP_PROB, # TODO change when evaluating
                            'memory': False}],
         7: [FcRNNCell, {'state_size': LAYER_SIZES[7]['state'],
                        'keep_prob': FC_KEEP_PROB,  # TODO change when evaluating
                          'memory': False}]
          # Note: Global Average Pooling (GAP) is included in final FC/softmax layer by setting GAP_FC = True
        }

BYPASSES = [] # bypasses: list of tuples (from, to)
