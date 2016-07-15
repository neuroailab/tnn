# VGG layers != actual vgg layers. they are just default convpool layers with 3x3 kernel.
# uses img size 256 but we should change so input is a 224 cropped image for vgg.
# flexible so only IMAGE_SIZE needs to be changed
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# todo - uncomment if any are needed. delete if not.
#import gzip
#import os
import sys
import time
import math
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import hdf5provider

# image parameters (num_channels, img_width/height)
IMAGE_SIZE = 256 # todo- are we going to crop it to 224? [256 from conv_img_cat.py tutorial]
NUM_CHANNELS = 3
PIXEL_DEPTH = 255 # from 0 to 255
NUM_LABELS = 1000 # [999 in conv_img_cat.py tutorial] todo- why?
TOTAL_IMGS_HDF5 = 1290129 # todo- check if this is correct? num images supplied thru hdf5provider from imagenet set

# graph parameters
WEIGHT_STDDEV = 0.1 # for truncated normal distribution
BIAS_INIT = 0.1 # initialization for bias variables
KERNEL_SIZE = 3 
DECAY_PARAM_INITIAL = 0.0# for self loops- p_j initializations (so actual decay parameter is initialized to ~ sigmoid(p_j) = 0.1)
KEEP_PROB= 0.5 # for dropout, during training.
LEARNING_RATE_BASE = 0.01 # initial learning rate. (we'll use exponential decay) [0.05 in conv_img_cat.py tutorial]
LEARNING_RATE_DECAY_FACTOR = 0.95 
TIME_PENALTY = 1.2 # 'gamma' time penalty as # time steps passed increases

# training parameters
TRAIN_SIZE = 100000 #1000000
BATCH_SIZE = 28 #64
NUM_EPOCHS = 10 
# validation/testing parameters
EVAL_BATCH_SIZE = 28 #64 # for both validation and testing
NUM_VALIDATION_BATCHES = 12
EVAL_FREQUENCY = 100 # number of steps btwn evaluations
NUM_TEST_BATCHES = 12

# tensorboard info
SUMMARIES_DIR = './tensorboard/default-256_variations' # where to write summaries

sess = tf.InteractiveSession()
# out_dict: dictionary to hold state outputs (which will be concatenated or used later)
# out_dict[(j, t)] = tf.Tensor output y of node j at time t
#     you determine which entry you want to access by looking at adj_list
out_dict = {}
# kernel_dict: to allow sharing of weights/biases
# kernel_dict[name_of_funct] = {'weights': tf.Variable..., 'bias': tf.Variable...}
kernel_dict = {}
# state_dict: state_dict[j, t] = state l of 'layer' j at time t. (then you can take a decay on this)
state_dict = {}


# Vgg functions
def vgg_1(input):
    return default_convpool(input, output_shape=[IMAGE_SIZE/2, 64], name='vgg_1')
def vgg_2(input):
    return default_convpool(input, output_shape=[IMAGE_SIZE/4, 128], name='vgg_2')
def vgg_3(input):
    return default_convpool(input, output_shape=[IMAGE_SIZE/8, 256], name='vgg_3')
def vgg_4(input):
    return default_convpool(input, output_shape=[IMAGE_SIZE/16, 512], name='vgg_4')
def vgg_5(input):
    return default_convpool(input, output_shape=[IMAGE_SIZE/32, 512], name='vgg_5')
def default_fc(input, output_size, dropout, name='fc'):
    # a general fully connected layer, with dropout placeholder (specified at runtime)
    # input = flattened tf.Tesnor; output_size = integer
    # check that you have already flattened your input
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        raise ValueError('your input shape, ', input_shape, 'is not 2-D. FLATTEN IT!!')
    input_size = input_shape[1]
    with tf.variable_scope(name) as scope:
        if not name in kernel_dict: # make and add
            weights = _weights([input_size, output_size])
            bias = _bias([output_size])
            kernel_dict[name] = {'weights': weights, 'bias': bias}
        else: # retrieve
            weights = kernel_dict[name]['weights']
            bias = kernel_dict[name]['bias']
        fc = tf.nn.relu(tf.add(tf.matmul(input, weights), bias), name=name)
        fc_dropout = tf.nn.dropout(fc, dropout, name='fc_dropout')
    _activation_summary(fc) # add to tf summary
    _activation_summary(fc_dropout)
    return fc_dropout
def fc_final(input): # flattens input and applies FC to get into [batchsize, numlabels] vector for softmax/xentropy,etc.
    with tf.variable_scope('classifier') as scope:
        # flatten input to [#batches, input_size]
        input_shape = input.get_shape().as_list()
        input_size = input_shape[1] * input_shape[2] * input_shape[3]
        input_flat = tf.reshape(input, [-1, input_size])
        name = 'fc_final'
        if not name in kernel_dict: # make and add
            weights = _weights([input_size, NUM_LABELS])
            bias = _bias([NUM_LABELS])
            kernel_dict[name] = {'weights': weights, 'bias': bias}
        else: # retrieve
            weights = kernel_dict[name]['weights']
            bias = kernel_dict[name]['bias']
        fc_final = tf.add(tf.matmul(input_flat, weights), bias, 'fc_final')
    return fc_final
def default_convpool(input, output_shape, name='convpool'):
    # output_shape = [SPATIAL_DIM, NUM_CHANNELS] determine pooling stride and # conv filters
    # name = unique transition (j) identifier (ex: convpool_j, vgg_j, etc.) so we can share weights between times.
    # get shape as a list [#batches, width, height, depth/channels]
    input_shape = input.get_shape().as_list()
    # shape = [spatial, spatial, num_input_channels, num_output_channels]
    weights_shape = [KERNEL_SIZE, KERNEL_SIZE, input_shape[3], output_shape[1]]
    with tf.variable_scope(name) as scope:
        # conv
        if not name in kernel_dict: # make and add
            weights = _weights(weights_shape)
            bias = _bias([output_shape[1]])
            kernel_dict[name] = {'weights': weights, 'bias': bias}
        else: # retrieve
            weights = kernel_dict[name]['weights']
            bias = kernel_dict[name]['bias']
        conv = conv_relu(input, weights, bias, name + '_conv')
        _activation_summary(conv)
        # pool
        pool = maxpool(input=conv, in_spatial=input_shape[1], 
                           out_spatial=output_shape[0], name=name+'pool')
    return pool

def _adjacency_list_creator(bypasses, N_states):
    # bypasses = list of tuples of bypasses
    # adjacency_list = (a reversed version): dictionary {TO state #: FROM [state, #s]} defined by bypasses
    # N_states = num states = num convpool blocks
    # first, create regular connections
    adjacency_list = {}
    for item in range(1, N_states + 1): # [1, ... N_states]
        adjacency_list[item] = [item-1]
    # now add in bypass connections
    for item in bypasses: # for each tuple
        adjacency_list[item[1]] = adjacency_list[item[1]] + [item[0]] # return new updated list
    for _, adj in adjacency_list.iteritems(): # SORT (asc.) every list in value entry of adjacency_list
        adj.sort() # (modifies original list.)
    return adjacency_list
# decay params creator - returns dictionary of p_js (trainable vars). Take sigmoid for decay factor. 
def _decay_params_creator(N_states):
    # N_states = num states = num convpool blocks
    decay_params = {}
    for item in range(1, N_states + 1):
        decay_params[item] = tf.Variable(tf.constant(DECAY_PARAM_INITIAL))
        tf.scalar_summary('decay_' + str(item), decay_params[item])
    return decay_params
# convert decay parameter to decay factor for actual usage
def to_decay_factor(decay_parameter):
    # decay_parameter is tf.Variable
    return tf.sigmoid(decay_parameter)
def add_to_out_dict(y, j, t):
    # y = output of convpoolj (aka from state j)
    # j = layer/state number
    # t = time step
    out_dict[(j,t)] = y
def add_to_state_dict(l, j, t):
    # l = state of layer j at time t
    state_dict[(j, t)] = l
    
# other graph creation helpers
def _weights(shape): # weights for convolution
    # shape = [spatial, spatial, num_input_channels, num_output_channels]
    # initialized with truncated normal distribution
    return tf.Variable(tf.truncated_normal(shape, stddev=WEIGHT_STDDEV), name='weights')
def _bias(shape): # bias variable for convolution
    return tf.Variable(tf.constant(BIAS_INIT, shape=shape), name='bias')
def _activation_summary(x): # helper function to create tf summaries for conv, fc, etc.
    # x = Tensor [aka output after convs, FCs, Softmax], returns nothing
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x)) # cool! measure sparsity too!
def conv_relu(input, weights, bias, name):
    # relu o (1x1 stride conv + bias relu)
    conv2d = tf.nn.conv2d(input, weights, strides=[1,1,1,1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(conv2d, bias), name=name)
def maxpool(input, in_spatial, out_spatial, name='pool'):
    stride = in_spatial/out_spatial # how much to pool by
    pool = tf.nn.max_pool(input, ksize=[1, stride, stride, 1],  # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool
# calculate state sizes based on output_sizes (wrt concatenation, bypasses)
def _calculate_state_sizes(output_sizes, adj_list):
    state_sizes = {} # dictionary{state#: [SPATIAL_DIM, NUM_CHANNELS]}
    for target, incoming in adj_list.iteritems():
        # retrieve spatial dim from output of immediate previous layer.
        spatial_dim = output_sizes[target-1][0] # same spatial dimension
        tot_channel_size = 0
        for inc in incoming: # for each incoming state
            # get channel size of inc and add to tot_channel_size
            tot_channel_size += output_sizes[inc][1]
        state_sizes[target] = [spatial_dim, tot_channel_size]
    return state_sizes

def get_shortest_path_length(adjacency_list): 
    # note that our adj_list is 'reversed'. So we search (equivalently) for shortest path from state N_state -> 1 
    # N_states = len(adj_list)
    distTo = [1000000] * (N_states+1) # adj_list + 1 so we index with 1, 2, 3.. N_states
    prev = [-1] * (N_states+1) # holds state that feeds into your state (prev in path)
    distTo[N_states] = 0 # starting point
    for node in xrange(N_states, 0, -1): # N_states, ..., 2, 1
        # look at adjacency list
        for target in adj_list[node]:
            # update the target if your new distance is less than previous
            if (distTo[target] > distTo[node] + 1):
                distTo[target] = distTo[node] + 1 # new smaller distance
                prev[target] = node # keep track of previous node
    # now check the shortest path length to 0, the input
    shortest_path_length_conv = distTo[0]
    shortest_path = shortest_path_length_conv + N_fc # add fc layers to get total shortest path
    return shortest_path
def error_rate(predictions, labels): # to use with evaluation: get array of all your predictions and compare with eval_labels
    # returns batch error rate (# correct/ batch size). 100% - (top-1 accuracy)%
    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

# run tensorflow with dictionary. NOTE: Inspired by Styrke's github: https://github.com/tensorflow/tensorflow/issues/1941
def tf_run_dict(session, fetches, feed_dict):
    # session = your tf session, feed_dict = feed_dict as usual
    # fetches = dictionary {'name1': fetch1, 'name2': fetch2}
    # returns dict {'name1': fetch1.eval(), 'name2': fetch2.eval()}
    keys, values = fetches.keys(), list(fetches.values()) 
    results = session.run(values, feed_dict) # get results out (as a list)
    return {key: value for key, value in zip(keys, results)} # reform as a dictionary and return



# INPUTS
# state 0 is considered input.
# convpool functs dictionary = {transition to layer: function}
# example: 1: vgg1 means to get from state 0 (img) -> state 1 we do vgg1.
convpool_functs = {1: vgg_1, 
                   2: vgg_2, 
                   3: vgg_3, 
                   4: vgg_4, 
                   5: vgg_5, 
                   'fc_final': fc_final}
bypasses = [] # list of tuples of bypasses (ON TOP of regular connections)
# N_states = N_conv
N_states = 5 
N_fc = 1 # includes last linear layer input to softmax

# output_sizes = dict {state #: shape = [SPATIAL DIMENSION, CHANNEL DEPTH]}
output_sizes = {0: [IMAGE_SIZE, NUM_CHANNELS],
                1: [IMAGE_SIZE/2, 64], 
                2: [IMAGE_SIZE/4, 128], 
                3: [IMAGE_SIZE/8, 256], 
                4: [IMAGE_SIZE/16, 512], 
                5: [IMAGE_SIZE/32, 512], 
                'fc': [1, 4096], 
                'fc_final': [1, 1000]}

# dictionary {TO state #: FROM [states, #, ...]} defined by bypasses
adj_list = _adjacency_list_creator(bypasses, N_states) 
# state sizes (resulting from concatenation of multiple or single outputs of prev layers)
state_sizes = _calculate_state_sizes(output_sizes, adj_list)
# losses_dict {time#: loss output at that time}
losses_dict = {}
# predictions_dict {time#: top 1 prediction}
predictions_dict = {}

# decay_parameters [not decay_factors] (p_j) for each layer, take the sigmoid of these to get your decay_factor
decay_params = _decay_params_creator(N_states)

shortest_path = get_shortest_path_length(adjacency_list=adj_list) # shortest path. aka what time point outputs start to matter
longest_path = N_states + N_fc


# get data
# create hdf5 datasources for training, validation, and testing
hdf5source = '/data/imagenet_dataset/hdf5_cached_from_om7/data.raw'
sourcelist = ['data', 'labels']
preprocess = {'labels': hdf5provider.get_unique_labels} 
# rescale pixel vals from [0, 255] down to [-0.5, 0.5]
norml = lambda x: (x - (PIXEL_DEPTH/2.0)) / PIXEL_DEPTH
postprocess = {'data': lambda x, _: norml(x).reshape((x.shape[0], NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).swapaxes(1, 2).swapaxes(2, 3)}
train_slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool); train_slice[:TRAIN_SIZE] = True
# TODO: WHY NOT EVAL_BATCH_SIZE here? (for _N and _M) -- or make 1 batch_size for train AND eval
_N = BATCH_SIZE * NUM_VALIDATION_BATCHES
validation_slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool); validation_slice[TRAIN_SIZE: TRAIN_SIZE + _N] = True
_M = BATCH_SIZE  * NUM_TEST_BATCHES
test_slice = np.zeros(TOTAL_IMGS_HDF5).astype(np.bool); test_slice[TRAIN_SIZE + _N: TRAIN_SIZE + _N + _M] = True
train_data = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                         preprocess=preprocess,
                                         postprocess=postprocess,
                                         subslice = train_slice,
                                         pad=True)
validation_dp = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                            preprocess=preprocess,
                                            postprocess=postprocess,
                                            subslice = validation_slice,
                                            pad=True)
validation_data = []
validation_labels = []
for i in range(NUM_VALIDATION_BATCHES):
    # add all the batches to giant data/label arrays 
    # todo Q: because we run ENTIRE validation set each time we validate??
    b = validation_dp.getBatch(i)
    validation_data.append(b['data'])
    validation_labels.append(b['labels'])
validation_data = np.row_stack(validation_data)
validation_labels = np.concatenate(validation_labels)

test_dp = hdf5provider.HDF5DataProvider(hdf5source, sourcelist, BATCH_SIZE,
                                        preprocess=preprocess,
                                        postprocess=postprocess,
                                        subslice = test_slice, pad=True)
test_data = []
test_labels = []
for i in range(NUM_TEST_BATCHES):
    b = test_dp.getBatch(i)
    test_data.append(b['data'])
    test_labels.append(b['labels'])
test_data = np.row_stack(test_data)
test_labels = np.concatenate(test_labels)


# placeholders
images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='images')
labels = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob') # for dropout


# create graph
for t in range(longest_path + 1): # 0 = input, N_states + N_fc = last softmax layer
    for j in range(0, N_states+1): #for each layer block, 1, 2, ... N_states
        print('t', t, 'j', j)
        if j == 0:
            with tf.name_scope('input_layer'):  
                if t == 0:
                    # initialize state variables and add to state_dict
                    state = images # your state variable is just the image. 
                else: # t > 0
                    # todo: need to decide if we blank the input or keep giving in images:
                    # CHOICE 1: blank input
                    state = tf.Variable(tf.zeros(images.get_shape()), trainable=False, dtype = tf.float32, name='input_' + str(t))
                    # CHOICE 2: keep feeding in same input
                    # state = images 
                    # CHOICE 3: other
                output = state # output is same as state for input layer
        else: # j > 0
            with tf.name_scope(convpool_functs[j].__name__):
                if t == 0:
                    spatial_dim = state_sizes[j][0]
                    num_channels = state_sizes[j][1]
                    # just a default initialization to zeros. Can use feed_dict to change this value with name: statej_0 
                    state = tf.Variable(tf.zeros([BATCH_SIZE, spatial_dim, spatial_dim, num_channels]),
                                                    trainable=False, dtype = tf.float32, name='state' + str(j) + '_' + str(t))
                else: # t>0
                    in_tensors = [] # will hold list of tf.Tensors that are incoming, we want to concatenate
                    for index in adj_list[j]: # for each of your incoming inputs
                        # find corresponding out_vector (tf.Tensor) from out_dict
                        incoming = out_dict[(index, t-1)] 
                        desired_spatial_size = state_sizes[j][0] 
                        incoming_spatial_size = incoming.get_shape().as_list()[1]
                        if (desired_spatial_size > incoming_spatial_size):
                            raise ValueError('Incoming spatial size is less than desired size. Strange.')
                        elif (desired_spatial_size < incoming_spatial_size): # maxpool to get to right shape
                            incoming = maxpool(incoming, incoming_spatial_size, desired_spatial_size, name='bypass_pool')
                        in_tensors.append(incoming)        
                    incoming_state = tf.concat(3, in_tensors) # concatenate along the channel dimension
                    # add decayed version of the previous you to this incoming, concatenated state
                    decay_factor = to_decay_factor(decay_params[j])
                    old_state = state_dict[(j, t-1)]
                    state = tf.add(incoming_state, tf.mul(decay_factor, old_state))
                # create output of this 'layer' (convolutions on these states) 
                output = convpool_functs[j](state) # (thank goodness for modularization).
        add_to_state_dict(state, j, t)
        add_to_out_dict(output, j, t) 
    if (t >= shortest_path):
        logits = convpool_functs['fc_final'](output) # output will always be last layer's output
        with tf.name_scope('loss'):
            print('LOSS!! AT T = ', t)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels), name='xentropy_loss_t'+str(t))
            tf.scalar_summary('loss_t'+str(t), loss)
            losses_dict['loss_' + str(t)] = loss # add to losses_dict so we can compare raw losses wrt time
            loss_term = tf.mul(loss, math.pow(TIME_PENALTY, t - shortest_path + 1)) # can modify (ex: multiply by a factor gamma^t) before adding to total loss
            tf.add_to_collection('losses', loss_term) # add loss to collection to be summed up 
            predictions = tf.nn.softmax(logits) # softmax predictions for current minibatch
            predictions_dict['pred_' + str(t)] = predictions # todo - will memory be an issue wrt to this stuff? :(
            

# add all loss contributions
total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
tf.scalar_summary('total_loss',total_loss)
# training step (optimizer)
batch_count = tf.Variable(0, trainable=False) # to count steps (inc. once per batch)
learning_rate = tf.train.exponential_decay(
      LEARNING_RATE_BASE,                # Base learning rate.
      batch_count * BATCH_SIZE,  # Current index into the dataset.
      TRAIN_SIZE,          # Decay step (aka once every EPOCH)
      LEARNING_RATE_DECAY_FACTOR,                # Decay rate.
      staircase=True)
tf.scalar_summary('learning_rate', learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

# softmax/accuracy or 1-acc = error/loss vector - collect at times N+1, N, ..., N + 1 - largest_bypass_hop for analysis
# errors_dict {time#: error at that time} ... changes for each batch input, of course
errors_dict = {} # dictionary of errors for different time steps. 
# confidences_dict {time #: confidence of top1 prediction}
confidences_dict = {}
loss_dict = {} # yes, this is not losses_dict(which holds tf.Tensors)- loss_dict holds scalars
for t in range(shortest_path, longest_path + 1):
    errors_dict[t] = [] # list, so we can append batch error rates wrt each output time
    confidences_dict[t] = [] 
    loss_dict[t] = []
# todo- can do same thing for accuracy, but accuracy is just 1 - error

# summary stuff
merged_summaries = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(SUMMARIES_DIR, sess.graph) 

# run 
start_time = start_time_step = time.time()
tf.initialize_all_variables().run()
print("We've initialized!")

numTrainingBatches = 500
for step in range(numTrainingBatches):
    batchd = train_data.getNextBatch()
    batch_data = batchd['data']
    batch_labels = batchd['labels']
    fetch_dict = {'opt': optimizer, 'summary': merged_summaries, 'lr': learning_rate, 'tot_loss': total_loss}
    fetch_dict.update(losses_dict) # concatenate all the dictionaries so we can tf.run
    fetch_dict.update(predictions_dict)
    feed_dict = {images:batch_data, labels: batch_labels, keep_prob: KEEP_PROB}
    results = tf_run_dict(sess, fetch_dict, feed_dict) # results as a dictionary- {'summary': summary, 'lr': lr, etc..}
    for t in range(shortest_path, longest_path + 1): # extract loss/predictions per time and do things
        predictions = results['pred_' + str(t)] # batch size x num_labels array        
        err_rate = error_rate(predictions, batch_labels) # batch error rate
        errors_dict[t].append(err_rate) # add to errors_dict[t] so we can plot & compare later 
        confidences_dict[t].append(np.mean(np.amax(predictions, 1), axis=0)) # see how 'prob' of top prediction varies with time/batch (averaged over batch)
        losses = results['loss_' + str(t)] # batch_size array
        loss_dict[t].append(losses)
    print('Step %d total_loss: %.6f, err_rate: %.6f, lr: %.6f' % (step, results['tot_loss'], err_rate, results['lr'])) # last error rate
    elapsed_time_step = time.time() - start_time_step
    start_time_step = time.time()
    print('step %d, %.1f ms' % (step, 1000 * elapsed_time_step))
    train_writer.add_summary(results['summary'], step) # write info to tensorboard!!
    if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time # todo - reinstate to eval instead of train, after we get things going
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / TRAIN_SIZE, # num imgs/total in an epoch
                                                1000 * elapsed_time / EVAL_FREQUENCY)) # average step time
        #print('Validation error: %.6f' % error_rate(
        #       eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush() # flush the stdout buffer

