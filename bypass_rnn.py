from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import sys
import time
import math
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import hdf5provider
from ConvRNN import ConvRNNCell, ConvPoolRNNCell, FcRNNCell
import ConvRNN
import threading


# Save training loss and validation error to: SAVE_FILE + [_loss.csv, or _val.csv]
SAVE_LOSS = True
SAVE_LOSS_FREQ = 5 # keeps loss from every SAVE_LOSS_FREQ steps.

# Save model parameters
SAVE_VARS = True # save variables if True
SAVE_FREQ = 2000 # save variables every SAVE_FREQ steps
MAX_TO_KEEP = 5 # keep last MAX_TO_KEEP files
SAVE_FILE = './outputs28/outputs28wd/bypass_rnn28_part4' # file name. NOTE: if you use another directory make sure it exists first.
# NOTE: DON'T OVERWRITE EXISTING FILES THAT MATTER! (DON'T DO IT!)

# Restoring model parameters from file
RESTORE_VARS = True #If True, restores variables from VAR_FILE instead of initializing from scratch
START_STEP = 54000 # to be used for step counter. If RESTORE_VARS=False, we start with 1.
VAR_FILE = './outputs28/outputs28wd/bypass_rnn28wd_part3-'+ str(START_STEP)

# Tensorboard info (Graph only)
TBOARD = False # creates graph (doesn't save activation info) if True.
VERSION_NUMBER = '28' # for storing tensorboard summaries
SUMMARIES_DIR = './tensorboard/bypass_rnn_'+ str(VERSION_NUMBER) + '_weight_decay' # where to put tboard graph

# Processing params
GPU_MEMORY_FRACTION=0.4
NUM_PREPROCESS_THREADS = 4

# Training parameters
TRAIN_SIZE = 1000000 #1000000
BATCH_SIZE = 128 #64
NUM_EPOCHS = 80

# Evaluation parameters
EVAL_FREQUENCY = 200 # number of steps btwn evaluations
EVAL_BATCH_SIZE = BATCH_SIZE # for both validation and testing
NUM_VALIDATION_BATCHES = 16
NUM_TEST_BATCHES = 12

# Image parameters
# DATA_PATH = '/mindhive/dicarlolab/common/imagenet/data.raw' # for openmind runs
DATA_PATH = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw' # for agent runs

IMAGE_SIZE_ORIG = 256
IMAGE_SIZE = 224 # What we crop it to.
NUM_CHANNELS = 3
PIXEL_DEPTH = 255 # from 0 to 255 (WHY NOT 256?)
NUM_LABELS = 1000
TOTAL_IMGS_HDF5 = 1290129

# Graph parameters. Note:  default weight sizes, strides, decay factor -> adjust in ConvRNN.py
T = 8 # longest path for training; otherwise any T >= shortest_path.
GAP_FC = False # False # whether or not to take global average pool before applying linear transform in final_fc layer
TIME_PENALTY = 1.2 # 'gamma' time penalty on loss as # time steps passed increases
KEEP_PROB= 0.5 # for dropout, during training. None for no dropout
GRAD_CLIP = False # gradient clipping option. Can turn off for non-recurrent network, but should turn on otherwise

# Optimization parameters
LEARNING_RATE_BASE = 0.03 # .001 for Adam # .05 initial learning rate. (we'll use exponential decay) [0.05 in conv_img_cat.py tutorial, .01 elsewhere]
LEARNING_RATE_DECAY_FACTOR = 0.95
MOMENTUM = 0.9 # for momentum optimizer

# whether or not to collect data about loss/predictions at intermediate time points
INTERMED_LOSSES = False
INTERMED_PREDS = False

def _adjacency_list_creator(bypasses, N_cells):
    """ bypasses = list of tuples of bypasses
    adjacency_list = (a reversed version): dictionary {TO state #: FROM [state, #s]} defined by bypasses
    first, create regular connections
    """
    adjacency_list = {}
    for item in range(1, N_cells + 1): # [1, ... N_cells] (doesn't include linearsoftmax layer)
        adjacency_list[item] = [item-1]
    # now add in bypass connections
    for item in bypasses: # for each tuple
        adjacency_list[item[1]] = adjacency_list[item[1]] + [item[0]] # return new updated list
    for _, adj in adjacency_list.iteritems(): # SORT (asc.) every list in value entry of adjacency_list
        adj.sort() # (modifies original list.)
    return adjacency_list

def get_shortest_path_length(adjacency_list): # using breadth first search
    # note: our adj_list is 'reversed'. So we search (equivalently) for shortest path from state N_state -> 1
    # lists to keep track of distTo and prev
    N_cells = len(adjacency_list)
    distTo = [1000000] * (N_cells+1) # adj_list + 1 so we index with 1, 2, 3.. N_cells
    prev = [-1] * (N_cells+1) # holds state that feeds into your state (prev in path)
    distTo[N_cells] = 0 # starting point
    for node in xrange(N_cells, 0, -1): # N_cells, ..., 2, 1
        # look at adjacency list
        for target in adj_list[node]:
            # update the target if your new distance is less than previous
            if (distTo[target] > distTo[node] + 1):
                distTo[target] = distTo[node] + 1 # new smaller distance
                prev[target] = node # keep track of previous node! # (actually irrelevant for now)
    # now check the shortest path length to 0 (input)
    shortest_path_length_conv = distTo[0]
    shortest_path = shortest_path_length_conv + 1 # add 1 for last linear/softmax layer that's always included
    return shortest_path

# to use with evaluation: get giant array of all your predictions and compare with eval_labels
def error_rate(predictions, labels):
    # returns batch error rate (# correct/ batch size). 100% - (top-1 accuracy)%
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])


def _make_cells_dict(layers):
    """ returns a dict : {#layer: initialized RNN cell}"""
    cells_list = {}
    for k, fun_list in layers.iteritems():
        fun = fun_list[0]
        kwarg_things = fun_list[1]
        cells_list[k] = fun(**kwarg_things)
    return cells_list

def maxpool(input, in_spatial, out_spatial, name='pool'):
    stride = in_spatial / out_spatial  # how much to pool by
    pool = tf.nn.max_pool(input, ksize=[1, stride, stride, 1],  # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool


def _graph_initials(input, N_cells):
    """ Returns prev_out dict with all zeros (tensors) but first input, curr_states with all Nones
    # input should be a TENSOR!! """
    prev_out = {}; curr_states = {}
    for i in range(1, N_cells + 1): # 1, ... N_cells
        prev_out[i] = tf.zeros(shape=layer_sizes[i]['output'], dtype=tf.float32, name='zero_out') # initially zeros of correct size
        curr_states[i] = None # note: can also initialize state to random distribution, etc...
        # if we use tf.variable (non-trainable) and then set curr_states[i]=theVariable.eval() [with interactive session]
    return prev_out, curr_states

def _make_curr_in(input, prev_out, adj_list, last=None, next_time=None): # for a given time step
    """ Gathers inputs based on adj_list. Maxpools down to right spatial size if needed
    curr_in keys: 1, .. N_cells + 1 (last one is for input to linear softmax layer)
    input = output of layer 0
    prev_out = dictionary; keys:1,..., N_cells
    last = dictionary of last time point that a cell matters
    next_time = current time point t
    If given last and next_time (not None), inputs to the trimmed off bottom-half of graph will be
    not have a value (so we don't waste concatenation operations). This is okay because curr_in of these cells should
    never be called.
    """
    curr_in = {}
    # concatenation along channel dimension for all conv and first FC input
    # the input to layer 1 (j=1) is solely from input
    curr_in[1] = input
    N_cells = len(prev_out)
    # trimming time
    if not (last is None and next_time is None):
        # determine which layers to skip
        relevant_cells = [c for c in last if last[c] >= next_time] # if the cell is still relevant
    else:
        relevant_cells = range(2, N_cells + 1)
    for j in relevant_cells: # 2, 3, ... N_cells (corresponding with adjacency_list keys 1, .. N_cells)
        # check if desired spatial size is 4d
        incoming_size = layer_sizes[j-1]['output']# the proper incoming size as taken from prev layer's output
        if len(incoming_size) == 4: # to concatenate in channel dimension and do pooling and all
            in_tensors = []
            for index in adj_list[j]:
                desired_spatial_size = incoming_size[1]
                incoming = prev_out[index]
                incoming_spatial_size = incoming.get_shape().as_list()[1]
                if (desired_spatial_size > incoming_spatial_size):
                    raise ValueError('Incoming spatial size', incoming_spatial_size, '\n is less than desired size,',
                                 desired_spatial_size, 'Strange.')
                elif (desired_spatial_size < incoming_spatial_size): # maxpool to get to right shape
                    #print('POOLING FROM', index, 'TO', j)
                    incoming = maxpool(incoming, incoming_spatial_size, desired_spatial_size, name='bypass_pool')
                in_tensors.append(incoming)
            incoming_tot = tf.concat(3, in_tensors) # concatenate along the channel dimension
            curr_in[j] = incoming_tot
        else: # you are after some FC layer. No need to concatenate. Your adj list should only have one entry
            if len(adj_list[j]) > 1:
                raise ValueError('Do you have bypass to FC layers that take only flattened outputs? (1st FC layer is ok)? No good.')
            else:
                curr_in[j] = prev_out[adj_list[j][0]]
    # add final linearsoftmax layer
    curr_in[N_cells + 1] = prev_out[N_cells] # from last layer before smax
    return curr_in


def final_fc(input, use_gap=GAP_FC, gap_keep_prob=None, pre_keep_prob=None):
    """flattens input (if not already flattened) and returns W(input)+b
    We don't include this with the other operations is because there is no activation function,
    unlike with regular FCs. It is purely part of a linear readout
    use_gap = True => global average pooling before final linear transform and softmax. (can also specify gap_keep_prob)
    gap_keep_prob = placeholder or None. If None, the graph should not have a keep_prob placeholder defined at all.
    pre_keep_prob = placeholder or None. Applies dropout to input to this final fc layer.
    """
    with tf.variable_scope('final_fc') as scope:
        # flatten input to [#batches, input_size]
        ## Note: new!
        if pre_keep_prob is not None:
            input = tf.nn.dropout(input, pre_keep_prob)
        if use_gap: # global average pooling
            input_shape = input.get_shape().as_list()
            if (len(input_shape) < 4): # GAP shouldn't come after any FCs, etc.
                raise ValueError('Input shape is not valid. Should be of form [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]')
            input = tf.nn.avg_pool(input, ksize=[1, input_shape[1], input_shape[2], 1], # kernel = spatial size (global pool)
                                   strides=[1, 1, 1, 1], padding='VALID')
            # use 'VALID' pooling to reduce to 1x1 spatial ('SAME' won't work here since stride = 1)
            if gap_keep_prob is None:
                print('no dropout used! Post GAP:', input)
            else:
                input = tf.nn.dropout(input, gap_keep_prob)  # apply dropout
                print('using dropout! keep_prob tensor:', gap_keep_prob, 'post GAP:', input)
        input_shape = input.get_shape().as_list()
        if len(input_shape) > 2: # needs flattening. assumed to have dimension 4.
            input_size = input_shape[1] * input_shape[2] * input_shape[3]
            input = tf.reshape(input, [-1, input_size])
        logits = ConvRNN.linear(input, output_size=NUM_LABELS)
    return logits

def _make_inputs(static_in, T, effect=None):
    """ effect=None: returns list of T copies of static_in
    effect=exponential: returns list of T versions of static_in, with an applied exponential mask
    """
    if effect is None:
        input_list = [static_in for t in range(T)]
    else: # todo - can add other effects.
        pass
    return input_list

def _first(bypasses, N_cells):
    """ bypasses = list of tuples of bypasses
    Returns dictionary first[i] = time t where layer matters"""
    # construct a (forward) adjacency 'list' (dictionary)
    fwd_adj_list = _fwd_adj_list_creator(bypasses, N_cells)
    first = {} # first {cell#, first t}
    curr_ind = [1]
    t = 1
    while len(first) < N_cells: # while we have not hit every cell
        next_ind = []
        for ind in curr_ind: # for current indices, check if already accounted for in first
            if not ind in first:
                first[ind] = t
                # then add adjacency list onto next_ind
                next_ind.extend(fwd_adj_list[ind])
        curr_ind = next_ind
        t += 1
    return first


def _last(adj_list, T):
    """ adj_list = (bkwd) adjacency list {TO: [FROM, ..]}
     T = total time steps for model
     Basically does the backwards version of _first
     Returns dictionary last[i] = time t where layer last matters"""
    last = {}  # last {cell#, last t}
    N_cells = len(adj_list)
    curr_ind = [N_cells]
    t = T - 1 # last time the N_cell'th cell will matter. since at T we get softmax readout
    while len(last) < N_cells:  # while we have not hit every cell
        next_ind = []
        for ind in curr_ind:  # for current indices, check if already accounted for in first
            if not ind in last and ind > 0:
                last[ind] = t
                # then add adjacency list onto next_ind
                next_ind.extend(adj_list[ind])
        curr_ind = next_ind
        t -= 1
    return last

def _fwd_adj_list_creator(bypasses, N_cells):
    # bypasses = list of tuples of bypasses
    # fwd adjacency_list = dictionary {FROM state #: TO [state1, state2, ...]} defined by bypasses
    # first, create regular connections
    fwd_adj_list = {}
    for item in range(1, N_cells + 1):  # [1, ... N_cells] (doesn't include linearsoftmax layer)
        fwd_adj_list[item] = [item + 1]
    # now add in bypass connections
    for item in bypasses:  # for each tuple (from, to)
        fwd_adj_list[item[0]].append(item[1]) # update TO list
    for _, adj in fwd_adj_list.iteritems():  # SORT (asc.) every list in value entry of fwd_adj_list
        adj.sort()  # (modifies original list.)
    return fwd_adj_list

# Graph structure
# sizes = [batch size, spatial, spatial, depth(num_channels)]
layer_sizes = { 0: {'state': [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], 'output':  [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]}, # input
                1: {'state': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 96], 'output': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 96]}, # stride2 conv AND pool!
                2: {'state': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 256], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 256]}, # convpool
                3: {'state': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384]},# conv
                4: {'state': [BATCH_SIZE,  IMAGE_SIZE/16, IMAGE_SIZE/16, 384], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 384]}, # conv
                5: {'state': [BATCH_SIZE,  IMAGE_SIZE/16, IMAGE_SIZE/16, 256], 'output': [BATCH_SIZE, IMAGE_SIZE/32, IMAGE_SIZE/32, 256]}, # convpool
                6: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 4096]}, # fc
                7: {'state': [BATCH_SIZE, 4096], 'output': [BATCH_SIZE, 4096]},  # fc
                }

####### template for entry in layers [can modify method default parameters in ConvRNN] ########
# 1: [ConvPoolRNNCell, {'state_size': layer_sizes[1]['state'], 'output_size': layer_sizes[1]['output']
#                       'conv_size': 3,  # kernel size for conv
#                       'conv_stride': 1  # stride for conv
#                       # 'weight_init': use default (currently 'xavier')
#                       # 'weight_stddev': only relevant if you use 'trunc_norm' initialization
#                       # 'bias_init': use default (currently 0.1),
#                       'pool_size': 2,  # kernel size for pool (defaults to = stride determined by layer sizes.)
#                       # 'decay_param_init': relevant if you have memory
#                       'memory': False}] # defaults to True (then uses decay_param_init)
WEIGHT_DECAY = 0.0005
layers = {1: [ConvPoolRNNCell, {'state_size':layer_sizes[1]['state'], 'output_size': layer_sizes[1]['output'],
                               'conv_size': 11, # kernel size for conv
                                'conv_stride': 4, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0,
                                'memory': False}],
         2: [ConvPoolRNNCell, {'state_size': layer_sizes[2]['state'], 'output_size': layer_sizes[2]['output'],
                               'conv_size': 5, # kernel size for conv
                                'conv_stride': 2, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0,
                                'memory': False}],
         3: [ConvRNNCell, {'state_size': layer_sizes[3]['state'],
                                'conv_size': 3,  # kernel size for conv
                                'conv_stride': 1,  # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                                'decay_param_init': 0,
                                'memory': False}],
         4: [ConvRNNCell, {'state_size': layer_sizes[4]['state'],
                           'conv_size': 3,  # kernel size for conv
                           'conv_stride': 1,  # stride for conv
                            'weight_decay': WEIGHT_DECAY, # None for none
                           'decay_param_init': 0,
                           'memory': False}],
         5: [ConvPoolRNNCell, {'state_size':layer_sizes[5]['state'], 'output_size': layer_sizes[5]['output'],
                               'conv_size': 3, # kernel size for conv
                                'conv_stride': 1, # stride for conv
                                'weight_decay': WEIGHT_DECAY, # None for none
                               'pool_size': 3, # kernel size for pool (defaults to = stride determined by layer sizes.),
                                'decay_param_init': 0,
                                'memory': False}],
         6: [FcRNNCell, {'state_size': layer_sizes[6]['state'],
                            'memory': False}],
         7: [FcRNNCell, {'state_size': layer_sizes[7]['state'],
                          'memory': False}]
          # Note: Global Average Pooling (GAP) is included in final FC/softmax layer by setting GAP_FC = True
        }

bypasses = [] # bypasses: list of tuples (from, to)

N_cells = len(layers)
longest_path = N_cells + 1 # includes final linear-softmax layer.
adj_list = _adjacency_list_creator(bypasses, N_cells) # dictionary {TO state #: FROM [states, #, ...]} defined by bypasses
shortest_path = get_shortest_path_length(adjacency_list=adj_list) # which time point outputs start to matter [accounts for linsoftmax layer]
cells_dict = _make_cells_dict(layers)

# TODO: change _model to only output logits and handle softmax, xentropy outside
def _model(cells, inputs, label, keep_prob=None, initial_states=None):
    """
    cells = dictionary of cells {1: tf.rnncell, 2: tf.rnncell, etc..}
    inputs = list of T inputs (placeholders). T >= 1
    label = ONE tf.placeholder for your label (batch) (corresp to img for inputs). single integer [sparse]
    initial_states=None or dictionary of initial_states

    RETURNS:
    fetch_dict = dictionary of things to run [use with sess.run, but also add other stuff you might want to run too]
    """
    # losses_dict {time#: loss output at that time}
    losses_dict = {}
    # predictions_dict {time#: softmax predictions}
    predictions_dict = {}
    prev_out, curr_states = _graph_initials(inputs[0], len(cells)) # len(cells) = N_cells
    if not initial_states is None:
        curr_states = initial_states  # Note: can input different initial states of cells.
    first = _first(bypasses, N_cells) # dict of first times that cell matters (nontrivial input reaches cell)
    last = _last(adj_list, T) # dict of last times that cell matters (affects latest output)
    for t, input_ in enumerate(inputs, 1): # start from t = 1 for consistency with notion of time step
        # get curr_in from adjacency list and prev_out
        curr_in = _make_curr_in(input=input_, prev_out=prev_out, adj_list=adj_list)
        next_out = {} # keep from contaminating current run
        print('--------------t = ', t, '-----------------')
        # print('current vars: ', [x.name for x in tf.trainable_variables()]) # (ensure vars reused)
        if t == 1: # only cell 1 should actually be updated/have output, but we initialize the rest.
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                # need to set unique variable scopes for each cell to use variable sharing correctly
                with tf.variable_scope(layers[cell_num][0].__name__ + '_' + str(cell_num)) as varscope: #op name
                    if cell_num == 1:
                        out, curr_states[cell_num] = tf.nn.rnn(cell,
                                                           inputs=[curr_in[cell_num]],  # has to be list
                                                           initial_state=curr_states[cell_num],
                                                           dtype=tf.float32)
                        next_out[cell_num] = out[0] # out is a list, so we extract its element
                    else: # don't update state nor output
                        tf.nn.rnn(cell,
                                  inputs=[curr_in[cell_num]],  # has to be list
                                  initial_state=curr_states[cell_num],
                                  dtype=tf.float32)
                        next_out[cell_num] = prev_out[cell_num]# keep zero output
                        print('next_out=prev_out for cell', cell_num, ': ', next_out[cell_num])
            with tf.variable_scope('final') as varscope:# just to initialize vars once.
                logits = final_fc(input=curr_in[cell_num + 1], use_gap=GAP_FC, pre_keep_prob=keep_prob)  # of input to softmax
        else:  # t > 1
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                # note- 13 modification. if first[cell_num] > t, you don't matter yet so don't call your cell/change output,state
                # note- 14 modification. if t > last[cell_num], you don't matter anymore
                if first[cell_num] > t or last[cell_num] < t: # cell doesn't matter yet/anymore
                    next_out[cell_num] = prev_out[cell_num]  # don't call cell, nor change output, state
                    print('next_out=prev_out for cell', cell_num, ': ', next_out[cell_num])
                else:
                    with tf.variable_scope(layers[cell_num][0].__name__ + '_' + str(cell_num)) as varscope:
                        varscope.reuse_variables()
                        out, curr_states[cell_num] = tf.nn.rnn(cell,
                                                               inputs=[curr_in[cell_num]],
                                                               initial_state=curr_states[cell_num],
                                                               dtype=tf.float32)
                        next_out[cell_num] = out[0]
            if t >= shortest_path:
                with tf.variable_scope('final') as varscope:
                    varscope.reuse_variables()
                    logits = final_fc(input=curr_in[cell_num + 1], use_gap=GAP_FC, pre_keep_prob = keep_prob)   # of input to softmax
                # loss and predictions
                with tf.variable_scope('loss') as varscope:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label),
                                          name='xentropy_loss_t' + str(t)) # averaged over batch.
                    if INTERMED_LOSSES:
                        losses_dict['loss_' + str(t)] = loss  # add to losses_dict so we can compare raw losses wrt time
                    loss_term = tf.mul(loss, math.pow(TIME_PENALTY, # TODO can modify (ex: multiply by a factor gamma^t) before adding to total loss
                                                      t - shortest_path))
                    tf.add_to_collection('losses', loss_term)  # add loss to collection to be summed up
                    if INTERMED_PREDS or t == T:
                        predictions = tf.nn.softmax(logits)  # softmax predictions for current minibatch
                        predictions_dict['pred_' + str(t)] = predictions
        # after running through network, update prev_out. (curr_states already updated)
        prev_out = next_out

    # add all loss contributions (check that T >= shortest_path)
    if T < shortest_path:
        raise ValueError('T < shortest path through graph. No good.')
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss') # NOTE: weight decay is automatically added to this collection!
    losses_dict['tot_loss'] = total_loss
    fetch_dict = {} # 'concatenate' predictions_dict and losses_dict
    fetch_dict.update(losses_dict)
    fetch_dict.update(predictions_dict)
    return fetch_dict


class ImageNetSingle(hdf5provider.HDF5DataProvider):
    def __init__(self, subslice, *args, **kwargs):
        super(ImageNetSingle, self).__init__(DATA_PATH,
                                             ['data', 'labels'],
                                             1,
                                             subslice=subslice,
                                             preprocess={'labels': hdf5provider.get_unique_labels},
                                             postprocess={'data': self.postproc},
                                             pad=True)

        self.data_node = tf.placeholder(tf.float32,
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data')
        self.labels_node = tf.placeholder(tf.int64, shape=[], name='labels')

    def postproc(self, ims, f):
        norml = lambda x: (x - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        r = (IMAGE_SIZE_ORIG - IMAGE_SIZE) // 2
        if r == 0: # no cropping
            return norml(ims).reshape((ims.shape[0], NUM_CHANNELS, IMAGE_SIZE_ORIG, IMAGE_SIZE_ORIG)).swapaxes(1,2).swapaxes(2, 3)
        else:
            return norml(ims).reshape((ims.shape[0], NUM_CHANNELS, IMAGE_SIZE_ORIG, IMAGE_SIZE_ORIG)).swapaxes(1,2).swapaxes(2, 3)[:,
                               r:r + IMAGE_SIZE, r:r + IMAGE_SIZE]

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

def get_validation_data():
    # create hdf5 datasources for training, validation, and testing
    sourcelist = ['data', 'labels']
    preprocess = {'labels': hdf5provider.get_unique_labels}
    # rescale pixel vals from [0, 255] down to [-0.5, 0.5]
    norml = lambda x: (x - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    postprocess = {'data': lambda x, _: norml(x).reshape((x.shape[0], NUM_CHANNELS,
                                                          IMAGE_SIZE, IMAGE_SIZE)).swapaxes(1,2).swapaxes(2, 3)}
    _N = EVAL_BATCH_SIZE * NUM_VALIDATION_BATCHES # eval_batch_size is set same as batch_size for modeling ease
    validation_slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool); validation_slice[TRAIN_SIZE: TRAIN_SIZE + _N] = True
    validation_dp = hdf5provider.HDF5DataProvider(DATA_PATH, sourcelist, BATCH_SIZE,
                                                preprocess=preprocess,
                                                postprocess=postprocess,
                                                subslice = validation_slice,
                                                pad=True)
    validation_data = []
    validation_labels = []
    for i in range(NUM_VALIDATION_BATCHES):
        # add all the batches to giant data/label arrays
        b = validation_dp.getBatch(i)
        validation_data.append(b['data'])
        validation_labels.append(b['labels'])
    validation_data = np.row_stack(validation_data)
    validation_labels = np.concatenate(validation_labels)

    return validation_data, validation_labels

def eval_in_batches(eval_data, eval_labels, tot_loss, pred_t, img_ph, labels,keep_prob, sess):
    """ Get all predictions for a dataset by running it in small batches.
    Evaluate fetch_dict containing: tot_loss, pred_t aka last predictions output (at T)
    img_ph, labels- images input and labels placeholder

    Returns predictions matrix (tot_imgs x num_labels) and avg_loss (scalar)
    (for future: if dropout placeholder exists, just automatically add in this method)
    """
    tot_imgs = eval_data.shape[0] # num imgs in eval set
    if tot_imgs < EVAL_BATCH_SIZE:
        raise ValueError('batch size > # eval dataset: %d' % tot_imgs)
    predictions = np.zeros(shape=(tot_imgs, NUM_LABELS), dtype=np.float32) # to store results of all batches
    fetch_dict = {'tot_loss': tot_loss, 'pred': pred_t}
    loss = np.zeros(shape=int(math.ceil(tot_imgs/EVAL_BATCH_SIZE))) # to collect losses before returning avg over data
    loss_index = 0 # for storing loss.
    for begin in xrange(0, tot_imgs, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= tot_imgs:
            # need to run and get output for evaluation
            if KEEP_PROB is None:
                results= sess.run(fetch_dict, feed_dict={img_ph: eval_data[begin:end, ...], labels: eval_labels[begin:end]})
            else:
                results = sess.run(fetch_dict,
                                   feed_dict={img_ph: eval_data[begin:end, ...],
                                              labels: eval_labels[begin:end],
                                              keep_prob: 1.0}) # no dropout during eval
            predictions[begin:end, :] = results['pred']
            loss[loss_index] = results['tot_loss']
        else:
            if KEEP_PROB is None: # NOTE: MODIFIED (obviously)
                results = sess.run(fetch_dict, feed_dict={img_ph: eval_data[-EVAL_BATCH_SIZE:, ...], labels:eval_labels[-EVAL_BATCH_SIZE:]})
            else:
                results = sess.run(fetch_dict, feed_dict={img_ph: eval_data[-EVAL_BATCH_SIZE:, ...],
                                                          labels:eval_labels[-EVAL_BATCH_SIZE:],
                                                          keep_prob: 1.0}) # no dropout during eval
            predictions[begin:, :] = results['pred'][begin - tot_imgs:, :]
            loss[loss_index] = results['tot_loss']
        # compute average tot_loss
        loss_index += 1 # increment loss index
    avg_loss = np.mean(loss)
    return predictions, avg_loss

def load_vars(sess, saver, var_file):
    """ sess = current tf.Session with graph created
    saver = your tf.train.Saver
    var_file = file from which to load variables"""""
    # graph should be created already
    saver.restore(sess, var_file)  # restore variables from var_file to graph


def run_and_process(sess):
    # Get data
    train_slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool);
    train_slice[:TRAIN_SIZE] = True
    data = ImageNetSingle(subslice=train_slice)  # TRAINING data
    q = tf.FIFOQueue(capacity=BATCH_SIZE,
                     dtypes=[tf.float32, tf.int64],
                     shapes=[(IMAGE_SIZE, IMAGE_SIZE, 3), []])
    enqueue_op = q.enqueue([data.data_node, data.labels_node])
    images_batch, labels_batch = q.dequeue_many(BATCH_SIZE) # are tensors already, not placeholders.

    input_list = _make_inputs(static_in=images_batch, T=T)  # list of input, repeated T times
    if KEEP_PROB is not None:
        keep_prob = tf.placeholder("float")  # following tutorial using 'float'. tf.float32 probably equiv.
        fetch_dict = _model(cells_dict, input_list, labels_batch, initial_states=None, keep_prob=keep_prob) # can be used for any dropout purposes
    else:  # keep_prob should not even be created! (it is None)
        fetch_dict = _model(cells_dict, input_list, labels_batch, initial_states=None)

    if RESTORE_VARS:
        start_step = START_STEP + 1  # to keep consistent count (of epochs passed, etc.)
    else:
        start_step = 1  # start from the very beginning, a very good place to start.

    # training step (optimizer)
    batch_count = tf.Variable(start_step - 1, trainable=False)  # to count steps (inc. once per batch by optimizer) AKA global_step
    if LEARNING_RATE_DECAY_FACTOR is None:
        learning_rate = LEARNING_RATE_BASE  # just a constant.
    else:
        learning_rate = tf.train.exponential_decay(
                            LEARNING_RATE_BASE,  # Base learning rate.
                            batch_count * BATCH_SIZE,  # Current index into the dataset.
                            TRAIN_SIZE,  # Decay step (aka once every EPOCH)
                            LEARNING_RATE_DECAY_FACTOR,  # Decay rate.
                            staircase=True)

    vars_before_optimizer = tf.all_variables()  # relevant if we restore model but switch optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # Note: Switched from adam to momentum
    gvs = optimizer.compute_gradients(fetch_dict['tot_loss']) # .minimize = compute gradients and apply them.
    if GRAD_CLIP:
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if not grad is None] # gradient clipping.
        # Note: some gradients returned are 'None' because there is no relation btwn that var and tot loss; so we skip those.
        optimizer = optimizer.apply_gradients(capped_gvs, global_step = batch_count) # returns an Operation to run
        print('Gradients clipped')
    else:
        optimizer = optimizer.apply_gradients(gvs, global_step = batch_count)
        print('Gradients not clipped')

    fetch_dict.update({'opt': optimizer, 'lr': learning_rate})

    if TBOARD:
        tf.train.SummaryWriter(SUMMARIES_DIR, sess.graph)

    # collect info at times shortest_path, shortest_path + 1, ... longest_path = N_cells + 1
    confidences_dict = {} # {time #: confidence of top1 prediction}
    loss_dict = {}  # yes, this is not losses_dict(which holds tf.Tensors)- loss_dict holds scalars
    for t in range(shortest_path, T + 1):
        #errors_dict[t] = []  # list, so we can append batch error rates wrt each output time
        confidences_dict[t] = []
        loss_dict[t] = []

    # Initialize and restore variables
    saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)  # defaults to save all variables.
    init = tf.initialize_all_variables()  # initialize variables
    sess.run(init)
    print("We've initialized!")
    if RESTORE_VARS:
        #saver_old = tf.train.Saver(vars_before_optimizer, max_to_keep=MAX_TO_KEEP)  # use if switching optimizer
        saver.restore(sess, VAR_FILE)  # restore variables from var_file to graph
        print("We've loaded!")

    print('Running for: ', int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE, 'iterations', 'batch_size:', BATCH_SIZE)
    tot_losses_len = int(math.floor(EVAL_FREQUENCY/SAVE_LOSS_FREQ)) # to save a few re-computations. FLOOR since we start with step = 1
    tot_losses = np.zeros([tot_losses_len, 2]) # to write to file

    # prepare file to save losses to
    outfile_loss = SAVE_FILE + '_loss.csv'
    for x in [outfile_loss]: # clear files
        f = open(x, "w+")
        f.close()

    start_time = start_time_step = time.time()  # start timer

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord) # start queue runners
    try:
        threads = []
        for i in range(NUM_PREPROCESS_THREADS):
            thread = threading.Thread(target=data.load_and_enqueue, args=(sess, enqueue_op, coord))
            thread.start()
            thread.daemon = True # thread closes when parent quits
            threads.append(thread)

        # num training batches = total number of imgs (including epoch repeats)/batch size
        for step in xrange(start_step, int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE + 1):  # start with step=START_STEP or 1.

            if not KEEP_PROB is None:
                feed_dict = {keep_prob: KEEP_PROB}  # if dropout implemented

            results = sess.run(fetch_dict, feed_dict)  # results as a dictionary- {'loss_2': loss, 'lr': lr, etc..}

            for t in range(shortest_path, T + 1):  # extract loss/predictions per time
                if INTERMED_PREDS or t == T:
                    predictions = results['pred_' + str(t)]  # batch size x num_labels array
                    #err_rate = error_rate(predictions, labels_batch)  # batch error rate # note: comment out since it doesn't take tensors right now..
                    #errors_dict[t].append(err_rate)  # add to errors_dict[t] so we can plot & compare later
                    top_confidence = np.mean(np.amax(predictions, 1), axis=0)
                    confidences_dict[t].append(top_confidence)  # see how 'prob' of top prediction varies with time/batch (averaged over batch)
                if INTERMED_LOSSES:
                    losses = results['loss_' + str(t)]  # batch_size array
                    loss_dict[t].append(losses)
            print('Step %d total_loss: %.6f, top_conf: %.6f, lr: %.6f' % (
                step, results['tot_loss'], top_confidence, results['lr']))

            if SAVE_LOSS and step%SAVE_LOSS_FREQ == 0:
                tot_losses[step//SAVE_LOSS_FREQ%tot_losses_len - 1, :] = [step, results['tot_loss']] # save so we can write to file later

            elapsed_time_step = time.time() - start_time_step
            start_time_step = time.time()
            print('step %d, %.1f ms' % (step, 1000 * elapsed_time_step))

            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time  # time between evaluations.
                print('Step %d (epoch %.2f), avg step time:  %.1f ms' % (
                    step, float(step) * BATCH_SIZE / TRAIN_SIZE,  # num imgs/total in an epoch
                    1000 * elapsed_time / EVAL_FREQUENCY))  # average step time
                sys.stdout.flush()  # flush the stdout buffer

                # Write to file. Note: we only write every EVAL_FREQUENCY to limit I/O bottlenecking.
                if SAVE_LOSS:
                    with open(outfile_loss, 'ab') as f_handle:
                        np.savetxt(
                            f_handle,  # file name
                            tot_losses,  # array to save
                            fmt='%.3f',  # formatting, 3 digits in this case
                            delimiter=',',  # column delimiter
                            newline='\n')  # new line character

                start_time = time.time()  # reset timer.

            if step % SAVE_FREQ == 0: # how often to SAVE OUR VARIABLES
                if SAVE_VARS == True:
                    saver.save(sess, save_path=SAVE_FILE, global_step=batch_count)

    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    sess.close()


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) # check which gpus/cpus used.
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=GPU_MEMORY_FRACTION)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
run_and_process(sess)
