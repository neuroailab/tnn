from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
""" use _model(...) to create a graph based on parameters specified in bypassrnn_params.py""" 

import math

#from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import ConvRNN
import bypassrnn_params as params

def _graph_initials(input, layer_sizes, N_cells):
    """
    Returns prev_out dict with all zeros (tensors) but first input, curr_states with all Nones
    # input should be a TENSOR!! """
    prev_out = {}; curr_states = {}
    for i in range(1, N_cells + 1): # 1, ... N_cells
        prev_out[i] = tf.zeros(shape=layer_sizes[i]['output'], dtype=tf.float32, name='zero_out') # initially zeros of correct size
        curr_states[i] = None # note: can also initialize state to random distribution, etc...
        # if we use tf.variable (non-trainable) and then set curr_states[i]=theVariable.eval() [with interactive session]
    return prev_out, curr_states

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


def _make_curr_in(input, prev_out, adj_list, layer_sizes, last=None, next_time=None): # for a given time step
    """ Gathers inputs based on adj_list. Maxpools down to right spatial size if needed
    curr_in keys: 1, .. N_cells + 1 (last one is for input to linear softmax layer)
    input = output of layer 0
    prev_out = dictionary; keys:1,..., N_cells
    TODO: if given the last (dict) & next_time, t, inputs to the trimmed off bottom-half of graph will be
    not have a value (so we don't waste concatenation operations). This is okay because curr_in of these cells should
    never be called.
    """
    curr_in = {}
    # concatenation along channel dimension for all conv and first FC input
    # the input to layer 1 (j=1) is solely from input
    curr_in[1] = input
    N_cells = len(prev_out)
    # trimming time
    if not (last == None and next_time == None):
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



# # Regular, no GAP final_fc layer
# def final_fc(input):
#     """flattens input (if not already flattened) and returns W(input)+b
#     We don't include this with the other operations is because there is no activation function,
#     unlike with regular FCs. It is purely part of a linear readout"""
#     with tf.variable_scope('final_fc') as scope:
#         # flatten input to [#batches, input_size]
#         input_shape = input.get_shape().as_list()
#         if len(input_shape) > 2: # needs flattening. assumed to have dimension 4.
#             input_size = input_shape[1] * input_shape[2] * input_shape[3]
#             input = tf.reshape(input, [-1, input_size])
#         logits = ConvRNN.linear(input, output_size=params.NUM_LABELS)
#     return logits

def final_fc(input, pre_keep_prob=None):  # Note: can add dropout BEFORE Fc/softmax
    """flattens input (if not already flattened) and returns W(input)+b
    We don't include this with the other operations is because there is no activation function,
    unlike with regular FCs. It is purely part of a linear readout
    pre_keep_prob = placeholder or None. Applies dropout to input to this final fc layer.
    # TODO- figure out how we can add dropout AFTER fc layers so we can remove this hack.
    """""
    with tf.variable_scope('final_fc') as scope:
        # flatten input to [#batches, input_size]
        if pre_keep_prob is not None:
            input = tf.nn.dropout(input, pre_keep_prob)
        input_shape = input.get_shape().as_list()
        if len(input_shape) > 2: # needs flattening. assumed to have dimension 4.
            input_size = input_shape[1] * input_shape[2] * input_shape[3]
            input = tf.reshape(input, [-1, input_size])
        logits = ConvRNN.linear(input, output_size=params.NUM_LABELS)
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
    # note: our adj_list is 'reversed'. So we search (equivalently) for shortest path from state N_state -> 1
    # lists to keep track of distTo and prev
    N_cells = len(adjacency_list)
    distTo = [1000000] * (N_cells+1) # adj_list + 1 so we index with 1, 2, 3.. N_cells
    prev = [-1] * (N_cells+1) # holds state that feeds into your state (prev in path)
    distTo[N_cells] = 0 # starting point
    for node in xrange(N_cells, 0, -1): # N_cells, ..., 2, 1
        # look at adjacency list
        for target in adjacency_list[node]:
            # update the target if your new distance is less than previous
            if (distTo[target] > distTo[node] + 1):
                distTo[target] = distTo[node] + 1 # new smaller distance
                prev[target] = node # keep track of previous node! # (actually irrelevant for now)
    # now check the shortest path length to 0 (input)
    shortest_path_length_conv = distTo[0]
    shortest_path = shortest_path_length_conv + 1 # add 1 for last linear/softmax layer that's always included
    return shortest_path

def _make_cells_dict(layers):
    """ returns a dict : {#layer: initialized RNN cell}"""
    cells_list = {}
    for k, fun_list in layers.iteritems():
        fun = fun_list[0]
        kwarg_things = fun_list[1]
        cells_list[k] = fun(**kwarg_things)
    return cells_list

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

def _model(layers, layer_sizes, bypasses, images, labels, train=True, keep_prob = None, initial_states=None):  
    """
    Creates a graph for training Should be called only once (to create a graph)
    Returns the fetch_dict that includes final predictions and losses.

    cells = dictionary of cells (layers) {1: tf.rnncell, 2: tf.rnncell, etc..}
    images = * batch of single images. We will create a list of T images. T >= 1
    labels = ONE tf.placeholder for your labels (batch) (corresp to img for images). single integer [sparse]
    initial_states=None or dictionary of initial_states
    train = True => training model; otherwise returns evaluation model. See fetch_dict (output).
    RETURNS:
    fetch_dict = dictionary of tf.Tensors to run [use with sess.run, but also add other stuff you might want to run too]
        - loss_T, pred_T, tot_loss
        - also, loss_t, pred_t for t >= shortest_path if eval == True
    """""
    # get all preliminary graph data structures
    N_cells = len(layers)
    T = params.T
    longest_path = N_cells + 1  # includes final linear-softmax layer.
    adj_list = _adjacency_list_creator(bypasses,
                                       N_cells)  # dictionary {TO state #: FROM [states, #, ...]} defined by bypasses
    shortest_path = get_shortest_path_length(
                        adjacency_list=adj_list)  # which time point outputs start to matter [accounts for linsoftmax layer]

    print('shortest path', shortest_path, 'T: ', T)
    cells = _make_cells_dict(layers)


    if not train: # eval
        keep_prob = None # just in case it's accidentally defined. not train prevails
    losses_dict = {} # {time#: loss output at that time}
    predictions_dict = {} # {time#: softmax predictions}
    input_list = _make_inputs(static_in=images, T=T, effect=None)  # list of input, repeated T times
    prev_out, curr_states = _graph_initials(input_list[0], layer_sizes, N_cells=len(cells)) # len(cells) = N_cells
    if not initial_states == None:
        curr_states = initial_states  # Note: can input different initial states of cells.
    first = _first(bypasses, N_cells) # dict of first times that cell matters (nontrivial input reaches cell)
    last = _last(adj_list, T) # dict of last times that cell matters (affects latest output)
    for t, input_ in enumerate(input_list, 1): # start from t = 1 for consistency with notion of time step
        # get curr_in from adjacency list and prev_out
        curr_in = _make_curr_in(input=input_, prev_out=prev_out, adj_list=adj_list, layer_sizes=layer_sizes)
        next_out = {} # keep from contaminating current run
        print('--------------t = ', t, '-----------------')
        # print('current vars: ', [x.name for x in tf.trainable_variables()]) # (ensure vars reused)
        if t == 1: # only cell 1 should actually be updated/have output, but we initialize the rest.
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                # need to set unique variable scopes for each cell to use variable sharing correctly
                with tf.variable_scope(layers[cell_num][0].__name__ + '_' + str(cell_num)) as varscope:  # op name
                    if cell_num == 1:
                        out, curr_states[cell_num] = tf.nn.rnn(cell,
                                                               inputs=[curr_in[cell_num]],  # has to be list
                                                               initial_state=curr_states[cell_num],
                                                               dtype=tf.float32)
                        next_out[cell_num] = out[0]  # out is a list, so we extract its element
                    else:  # don't update state nor output
                        tf.nn.rnn(cell,
                                  inputs=[curr_in[cell_num]],  # has to be list
                                  initial_state=curr_states[cell_num],
                                  dtype=tf.float32)
                        next_out[cell_num] = prev_out[cell_num]  # keep zero output
                        print('next_out=prev_out for cell', cell_num, ': ', next_out[cell_num])
            with tf.variable_scope('final') as varscope:  # just to initialize vars once.
                logits = final_fc(input=curr_in[cell_num + 1], pre_keep_prob=keep_prob) # of input to softmax
        else:  # t > 1
            # do your rnn stuff here
            for cell_num, cell in cells.iteritems():  # run through network
                # if first[cell_num] > t, you don't matter yet so don't call your cell/change output,state
                # if t > last[cell_num], you don't matter anymore
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
                    logits = final_fc(input=curr_in[cell_num + 1], pre_keep_prob=keep_prob)   # of input to softmax
                # loss and predictions
                with tf.variable_scope('loss') as varscope:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels),
                                          name='xentropy_loss_t' + str(t)) # averaged over batch.
                    #if t == T: # Note: we currently don't care about intermediate losses.
                        #losses_dict['loss_' + str(t)] = loss  # add to losses_dict so we can compare raw losses wrt time
                    loss_term = tf.mul(loss, math.pow(params.TIME_PENALTY,
                                            t - shortest_path))  # TODO can modify (ex: multiply by a factor gamma^t) before adding to total loss
                    tf.add_to_collection('losses', loss_term)  # add loss to collection to be summed up
                    if (params.EVAL_INTERMED) or t == T: # not train: add intermediate t; or always add
                        predictions = tf.nn.softmax(logits)  # softmax predictions for current minibatch
                        predictions_dict['pred_' + str(t)] = predictions  
        # after running through network, update prev_out. (curr_states already updated)
        prev_out = next_out

    # add all loss contributions (check that T >= shortest_path)
    if T < shortest_path:
        raise ValueError('T < shortest path through graph. No good.')
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    losses_dict['tot_loss'] = total_loss
    fetch_dict = {} # 'concatenate' predictions_dict and losses_dict
    if train:
        fetch_dict.update(losses_dict) # Note: Only care about losses during training. Can always adjust if we care
        # for evaluation too!
    else:
        fetch_dict.update(predictions_dict) # Note: only care about predictions during evaluation.
    return fetch_dict
