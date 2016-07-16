from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

VERSION_NUMBER = 10 # for storing tensorboard summaries

# image parameters (num_channels, img_width/height)
T = 6 # longest path
IMAGE_SIZE = 256 # todo- are we going to crop it to 224?
NUM_CHANNELS = 3
PIXEL_DEPTH = 255 # from 0 to 255
NUM_LABELS = 1000 # [999 in conv_img_cat.py tutorial] todo- why?
TOTAL_IMGS_HDF5 = 1290129 # todo- check if this is correct? num images supplied thru hdf5provider from imagenet set

# graph parameters (adjust in ConvRNN.py)
# WEIGHT_STDDEV = 0.1 # for truncated normal distribution
# BIAS_INIT = 0.1 # initialization for bias variables
# KERNEL_SIZE = 3
# DECAY_PARAM_INITIAL = -2.197 # for self loops- p_j initializations (so actual decay parameter is initialized to ~ sigmoid(p_j) = 0.1)
KEEP_PROB= 0.5 # for dropout, during training. # todo - implement dropout
LEARNING_RATE_BASE = 0.05 # initial learning rate. (we'll use exponential decay) [0.05 in conv_img_cat.py tutorial]
LEARNING_RATE_DECAY_FACTOR = 0.95
TIME_PENALTY = 1.2 # 'gamma' time penalty as # time steps passed increases

# training parameters
TRAIN_SIZE = 1000000 #1000000
BATCH_SIZE = 32 #64
NUM_EPOCHS = 80
# validation/testing parameters
EVAL_BATCH_SIZE = 32 #64 # for both validation and testing
NUM_VALIDATION_BATCHES = 12
EVAL_FREQUENCY = 100 # number of steps btwn evaluations
NUM_TEST_BATCHES = 12
# tensorboard info
SUMMARIES_DIR = './tensorboard/bypass_rnn_'+ str(VERSION_NUMBER) # where to write summaries
SAVE_FREQ = 250 # save variables every SAVE_FREQ steps

def _adjacency_list_creator(bypasses, N_cells):
    # bypasses = list of tuples of bypasses
    # adjacency_list = (a reversed version): dictionary {TO state #: FROM [state, #s]} defined by bypasses
    # first, create regular connections
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
    # note that our adj_list is 'reversed'. So we search (equivalently) for shortest path from state N_state -> 1
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

# run tensorflow with dictionary. NOTE: Inspired by Styrke's github: https://github.com/tensorflow/tensorflow/issues/1941
def tf_run_dict(session, fetches, feed_dict):
    # session = your tf session, feed_dict = feed_dict as usual
    # fetches = dictionary {'name1': fetch1, 'name2': fetch2}
    # returns dict {'name1': fetch1.eval(), 'name2': fetch2.eval()}
    keys, values = fetches.keys(), list(fetches.values()) # not in order of putting, but in corresponding order.
    results = session.run(values, feed_dict) # get results out (as a list)
    return {key: value for key, value in zip(keys, results)} # reform as a dictionary =D and return

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
    """
    Returns prev_out dict with all zeros (tensors) but first input, curr_states with all Nones
    # input should be a TENSOR!! """
    prev_out = {}; curr_states = {}
    for i in range(1, N_cells + 1): # 1, ... N_cells
        prev_out[i] = tf.zeros(shape=layer_sizes[i]['output'], dtype=tf.float32, name='zero_out') # initially zeros of correct size
        curr_states[i] = None # note: can also initialize state to random distribution, etc...
        # if we use tf.variable (non-trainable) and then set curr_states[i]=theVariable.eval() [with interactive session]
    return prev_out, curr_states

def _make_curr_in(input, prev_out, adj_list): # for a given time step
    """ Gathers inputs based on adj_list. Maxpools down to right spatial size if needed
    curr_in keys: 1, .. N_cells + 1 (last one is for input to linear softmax layer)
    input = output of layer 0
    prev_out = dictionary; keys:1,..., N_cells
    """
    curr_in = {}
    # concatenation along channel dimension for all conv and first FC input
    # the input to layer 1 (j=1) is solely from input
    curr_in[1] = input
    N_cells = len(prev_out)
    for j in range(2, N_cells + 1): # 2, 3, ... N_cells (corresponding with adjacency_list keys 1, .. N_cells)
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

def final_fc(input): # flattens input (if not already flattened) and returns W(input)+b
    # We don't include this with the other operations is because there is no activation function,
    # unlike with regular FCs. It is purely part of a linear readout
    with tf.variable_scope('final_fc') as scope:
        # flatten input to [#batches, input_size]
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
    if effect == None:
        input_list = [static_in for t in range(T)]
    else: # todo - can add other effects.
        pass
    return input_list

# Graph structure
# sizes = [batch size, spatial, spatial, depth(num_channels)]
# Todo - for non-default values (strides, filter sizes, etc.) make nicer input format
layer_sizes = { 0: {'state': [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], 'output':  [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS]}, # input
                1: {'state': [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 32], 'output': [BATCH_SIZE, IMAGE_SIZE/2, IMAGE_SIZE/2, 32]},
                2: {'state': [BATCH_SIZE, IMAGE_SIZE/2, IMAGE_SIZE/2, 64], 'output': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 64]},
                3: {'state': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 128], 'output': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 128]},
                4: {'state': [BATCH_SIZE, IMAGE_SIZE/4, IMAGE_SIZE/4, 128], 'output': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 128]},
                5: {'state': [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 256], 'output': [BATCH_SIZE, IMAGE_SIZE/16, IMAGE_SIZE/16, 256]}#,
                #6: {'state': [BATCH_SIZE, 1024], 'output': [BATCH_SIZE, 1024]}
              }
layers = {1: [ConvPoolRNNCell, {'state_size':layer_sizes[1]['state'], 'output_size': layer_sizes[1]['output']}],
         2: [ConvPoolRNNCell, {'state_size': layer_sizes[2]['state'], 'output_size': layer_sizes[2]['output']}],
         3: [ConvRNNCell, {'state_size': layer_sizes[3]['state']}],
         4: [ConvPoolRNNCell, {'state_size':layer_sizes[4]['state'], 'output_size': layer_sizes[4]['output']}],
         5: [ConvPoolRNNCell, {'state_size':layer_sizes[5]['state'], 'output_size': layer_sizes[5]['output']}],
         #6: [FcRNNCell, {'state_size':layer_sizes[6]['state']}]
        }
N_cells = len(layers)
longest_path = N_cells + 1 # includes final linear-softmax layer.
bypasses = [] # bypasses: list of tuples (from, to)
adj_list = _adjacency_list_creator(bypasses, N_cells) # dictionary {TO state #: FROM [states, #, ...]} defined by bypasses
shortest_path = get_shortest_path_length(adjacency_list=adj_list) # which time point outputs start to matter [accounts for linsoftmax layer]

cells_dict = _make_cells_dict(layers)


def _model(cells, inputs, label, initial_states=None):
    """
    cells = dictionary of cells {1: tf.rnncell, 2: tf.rnncell, etc..}
    inputs = list of T inputs (placeholders). T >= 1
    label = ONE tf.placeholder for your label (batch) (corresp to img for inputs). single integer [sparse]
    initial_states=None or dictionary of initial_states

    RETURNS:
    fetch_dict = dictionary of things to run [use with tf_run_dict, but also add other stuff you might want to run too]
    """
    # losses_dict {time#: loss output at that time}
    losses_dict = {}
    # predictions_dict {time#: softmax predictions} # TODO - UNLESS WE ONLY CARE ABOUT TOP 1, THEN LET'S SAVE SOME MEMORY
    predictions_dict = {}
    prev_out, curr_states = _graph_initials(inputs[0], len(cells)) # len(cells) = N_cells
    if not initial_states == None:
        curr_states = initial_states  # Note: can input different initial states of cells.
    for t, input_ in enumerate(inputs, 1): # start from t = 1 for consistency with notion of time step
        # get curr_in from adjacency list and prev_out
        curr_in = _make_curr_in(input=input_, prev_out=prev_out, adj_list=adj_list)
        next_out = {}
        next_states = {}  # keep from contaminating current run
        print('--------------t = ', t, '-----------------')
        # print('current vars: ', [x.name for x in tf.trainable_variables()]) # (ensure vars reused)
        if t == 1:
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                # need to set unique variable scopes for each cell to use variable sharing correctly
                with tf.variable_scope(layers[cell_num][0].__name__ + '_' + str(cell_num)) as varscope: #op name
                    out, next_states[cell_num] = tf.nn.rnn(cell,
                                                           inputs=[curr_in[cell_num]],  # has to be list
                                                           initial_state=curr_states[cell_num],
                                                           dtype=tf.float32)
                    next_out[cell_num] = out[0] # out is a list, so we extract its element
            with tf.variable_scope('final') as varscope:# just to initialize vars once.
                logits = final_fc(input=curr_in[cell_num + 1])  # of input to softmax
        else:  # t > 1
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                with tf.variable_scope(layers[cell_num][0].__name__ + '_' + str(cell_num)) as varscope:
                    varscope.reuse_variables()
                    out, next_states[cell_num] = tf.nn.rnn(cell,
                                                           inputs=[curr_in[cell_num]],
                                                           initial_state=curr_states[cell_num],
                                                           dtype=tf.float32)
                    next_out[cell_num] = out[0]
            if t >= shortest_path:
                with tf.variable_scope('final') as varscope:
                    varscope.reuse_variables()
                    logits = final_fc(input=curr_in[cell_num + 1])  # of input to softmax
                # loss and predictions
                with tf.variable_scope('loss') as varscope:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label),
                                          name='xentropy_loss_t' + str(t))
                    losses_dict['loss_' + str(t)] = loss  # add to losses_dict so we can compare raw losses wrt time
                    loss_term = tf.mul(loss, math.pow(TIME_PENALTY,
                                                      t - shortest_path))  # TODO can modify (ex: multiply by a factor gamma^t) before adding to total loss
                    tf.add_to_collection('losses', loss_term)  # add loss to collection to be summed up
                    predictions = tf.nn.softmax(logits)  # softmax predictions for current minibatch
                    predictions_dict['pred_' + str(t)] = predictions  # todo - will memory be an issue wrt to this stuff? :(
        # after running through network, update curr_stuff
        prev_out = next_out
        curr_states = next_states
    # add all loss contributions (check that T >= shortest_path)
    if T < shortest_path:
        raise ValueError('T < shortest path through graph. No good.')
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    losses_dict['tot_loss'] = total_loss
    fetch_dict = {} # 'concatenate' predictions_dict and losses_dict
    fetch_dict.update(losses_dict)
    fetch_dict.update(predictions_dict)
    return fetch_dict

def get_data():
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

    return train_data, validation_data, validation_labels, test_data, test_labels


# todo - add other stuff to fetch_dict [decay_param?]
def run_and_process(sess):
    # placeholders
    labels = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='labels')
    img_ph = tf.placeholder(tf.float32, shape=layer_sizes[0]['output'], name='images')
    input_list = _make_inputs(static_in=img_ph, T=T)  # list of input, repeated T times
    fetch_dict = _model(cells_dict, input_list, labels, initial_states=None)

    # training step (optimizer)
    batch_count = tf.Variable(0, trainable=False)  # to count steps (inc. once per batch) aka the global_step
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # Base learning rate.
        batch_count * BATCH_SIZE,  # Current index into the dataset.
        TRAIN_SIZE,  # Decay step (aka once every EPOCH)
        LEARNING_RATE_DECAY_FACTOR,  # Decay rate.
        staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(fetch_dict['tot_loss'])  # TODO - can also use MOMENTUM.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(fetch_dict['tot_loss']) # .minimize = compute gradients and apply them.
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]    # gradient clipping.
    optimizer = optimizer.apply_gradients(capped_gvs, global_step = batch_count) # returns an Operation to run

    fetch_dict.update({'opt': optimizer, 'lr': learning_rate})
    # collect info at times shortest_path, shortest_path + 1, ... longest_path = N_cells + 1
    # errors_dict {time#: error at that time} ... changes for each batch input, of course
    errors_dict = {}  # dictionary of list of errors [by batch] for different time steps.
    acc_dict = {}  # dictionary of list of accuracies [by batch] (acc is just 1 - error)
    # confidences_dict {time #: confidence of top1 prediction}
    confidences_dict = {}
    loss_dict = {}  # yes, this is not losses_dict(which holds tf.Tensors)- loss_dict holds scalars
    for t in range(shortest_path, max(min(T + 1, longest_path + 1), T + 1)):
        errors_dict[t] = []  # list, so we can append batch error rates wrt each output time
        confidences_dict[t] = []
        loss_dict[t] = []
        acc_dict[t] = []  # accuracy is just 1 - error. but it might help in future.

    # run
    start_time = start_time_step = time.time()  # start timer
    saver = tf.train.Saver() # defaults to save all variables.
    init = tf.initialize_all_variables()  # initialize variables
    sess.run(init)
    print("We've initialized!")
    #numTrainingBatches = 500  # temp- for testing purposes
    train_data, validation_data, validation_labels, test_data, test_labels = get_data()
    print('WE ARE GOING TO RUN FOR: ', int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE, 'ITERATIONS')
    for step in xrange(int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE): # num training batches = total number of imgs (including epoch repeats)/batch size

        batchd = train_data.getNextBatch()
        batch_data = batchd['data']
        batch_labels = batchd['labels']
        feed_dict = {img_ph: batch_data, labels: batch_labels}  # TODO!! , keep_prob: KEEP_PROB}
        results = tf_run_dict(sess, fetch_dict,
                              feed_dict)  # results as a dictionary- {'loss_2': loss, 'lr': lr, etc..}
        for t in range(shortest_path, max(min(T + 1, longest_path + 1), T + 1)):  # extract loss/predictions per time
            predictions = results['pred_' + str(t)]  # batch size x num_labels array
            err_rate = error_rate(predictions, batch_labels)  # batch error rate
            errors_dict[t].append(err_rate)  # add to errors_dict[t] so we can plot & compare later
            top_confidence = np.mean(np.amax(predictions, 1),axis=0)
            confidences_dict[t].append(top_confidence)  # see how 'prob' of top prediction varies with time/batch (averaged over batch)
            losses = results['loss_' + str(t)]  # batch_size array
            loss_dict[t].append(losses)
        print('Step %d total_loss: %.6f, err_rate: %.6f, top_conf: %.6f, lr: %.6f' % (
        step, results['tot_loss'], err_rate, top_confidence, results['lr']))  # last error rate
        elapsed_time_step = time.time() - start_time_step
        start_time_step = time.time()
        print('step %d, %.1f ms' % (step, 1000 * elapsed_time_step))
        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time  # todo - reinstate to eval instead of train, after we get things going
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' % (
            step, float(step) * BATCH_SIZE / TRAIN_SIZE,  # num imgs/total in an epoch
            1000 * elapsed_time / EVAL_FREQUENCY))  # average step time
            # print('Validation error: %.6f' % error_rate(
            #       eval_in_batches(validation_data, sess), validation_labels))
            sys.stdout.flush()  # flush the stdout buffer
        if step % 250 == 0: # how often to SAVE OUR VARIABLES
            # Append the step number to the checkpoint name:
            saver.save(sess, save_path='bypass_rnn10_saver', global_step=batch_count)


#run_and_process(tf.Session(config=tf.ConfigProto(log_device_placement=True))) # check which gpus/cpus used.
run_and_process(tf.Session())
