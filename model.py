"""
The "model" function is used to create a TF graph given layers and bypasses
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import networkx as nx
import ConvRNN

TRIM_TOP = True
TRIM_BOTTOM = True  # might not need to use, since we don't expect that
# bottom nodes that don't contribute to final loss will matter. And we also
# may want to access all layer's outputs or states at time T.


def _model(layers, layer_sizes, bypasses, input_seq,
           T_tot, initial_states=None, num_labels=1000):
    """
    Creates model graph and returns logits.
    :param layers: Dictionary to construct cells for each layer of the form
    {layer #: ['cell type', {arguments}] Does not include the final linear
    layer used to get logits.
    :param layer_sizes: Dictionary of dictionaries containing state and
    output sizes for each layer
    :param bypasses: list of tuples (from, to)
    :param input_seq: list for sequence of input images as tf Tensors
    :param T_tot: total number of time steps to run the model.
    :param initial_states: optional; dict of initial state {layer#: tf Tensor}
    :param num_labels: Size of logits to output [1000 for ImageNet]

    :return: Returns a dictionary logits (output of a linear FC layer after
    all layers). {time t: logits} for t >= shortest_path and t < T_total}
    """

    # get dictionary of RNNCells
    cells = _layers_to_cells(layers)

    # create networkx graph with layer #s as nodes
    N_cells = len(layers)  # number of layers without final FC/logits
    graph = _graph(bypasses, N_cells=N_cells)

    # ensure that graph is acyclic
    assert (nx.is_directed_acyclic_graph(graph)), 'graph not acyclic'

    # ensure that T_tot >= shortest_path through graph
    shortest_path = nx.shortest_path_length(graph, source=0,
                                            target=N_cells + 1)
    assert (T_tot >= shortest_path), \
        'T_tot (%d) < shortest path length (%d)' % (T_tot, shortest_path)

    # get first and last time points where layers matter
    if TRIM_TOP:
        first = _first(graph)
    else:
        first = {0: 0} # input matters at t = 0, rest starting t = 1
        first.update({j: 1 for j in range(1, N_cells + 1 + 1)})

    if TRIM_BOTTOM:
        last = _last(graph, T_tot)
    else:
        last = {j: T_tot for j in range(0, N_cells + 1 + 1)}

    # check inputs: Compares input sequence length with the input length
    # that is needed for output at T_tot. Zero pads or truncates as needed
    if len(input_seq) > last[0] + 1:  # more inputs than needed => truncate
        print('truncating input sequence to length', last[0] + 1)
        del input_seq[last[0] + 1:]
    elif len(input_seq) < last[0] + 1:  # too short => pad with zero inputs
        print('zero-padding input sequence to length', last[0] + 1)
        num_needed = (last[0] + 1) - len(input_seq)  # inputs to add
        if not input_seq:  # need input length of at least one
            raise ValueError('input sequence should not be empty')
        padding = [tf.zeros_like(input_seq[0]) for i in range(0, num_needed)]
        input_seq.extend(padding)

    # add inputs to outputs dict for layer 0
    # outputs = {layer#: {t1:__, t2:__, ... }}
    outputs = {0: {t: input_ for t, input_ in enumerate(input_seq)}}
    final_states = {}  # holds final states from the last time state changes

    # create zero initial states if none specified
    if initial_states is None:
        # zero state returns zeros (tf.float32) based on state size.
        initial_states = {j: cells[j].zero_state(None, None) for j in
                          range(1, N_cells + 1)}

    # create graph layer by layer
    for j in range(1, N_cells + 1):
        with tf.variable_scope(layers[j][0] + '_' + str(j)):
            print('------------j:', j, ' - ', layers[j][0], '-----------')
            # create inputs list for layer j, each element is an input in time
            inputs = []  # list of inputs to layer j in time
            incoming = graph.predecessors(j)  # list of incoming nodes

            # for relevant time points: gather inputs, pool, and concatenate
            for t in range(first[j], last[j] + 1):
                # concatenate inputs (pooled to right spatial size) at time t
                incoming_shape = layer_sizes[j - 1]['output']  # with no bypass
                if len(incoming_shape) == 4:
                    inputs_t = [_maxpool(input=outputs[i][t - 1],
                                         out_spatial=incoming_shape[1],
                                         name='bypass_pool') for i in
                                sorted(incoming)]
                    # concat in channel dim
                    inputs.append(tf.concat(3, inputs_t))

                else:  # if input is 2D (ex: after FC layer)
                    print('FC')
                    # no bypass to FC layers beyond first FC
                    assert (len(incoming) == 1), 'No bypass to FC layers ' \
                                                 'beyond first FC allowed'
                    inputs_t = outputs[incoming[0]][t - 1]
                    inputs.append(inputs_t)
                print('inputs at t = ', t, ':', inputs_t)

            # run tf.nn.rnn and get list of outputs
            # Even if initial_states[j] is None, tf.nn.rnn will just set
            # zero initial state (given dtype)
            if inputs:
                outputs_list, final_state = tf.nn.rnn(cell=cells[j],
                                                      inputs=inputs,
                                                      initial_state=
                                                      initial_states[j],
                                                      dtype=tf.float32)
            else:  # if empty, layer j doesn't contribute to output t<= T_tot
                final_state = initial_states[j]

            # trim graph- fill in empty outputs with zeros
            outputs[j] = {t: tf.zeros(shape=layer_sizes[j]['output'],
                                      dtype=tf.float32) for t in
                          range(0, first[j])}

            # update outputs and state dictionary
            for i, t in enumerate(range(first[j], last[j] + 1)):
                outputs[j][t] = outputs_list[i]
            final_states[j] = final_state  # todo - return if desired.

    # last FC to get logits = {t: logits(t)} for t in [shortest path, T]
    logits = {}
    for t in range(shortest_path, T_tot + 1):
        print('--Final FC---t', t, '-----------')
        input_ = _flatten_input(outputs[N_cells][t - 1])
        with tf.variable_scope('final_fc') as varscope:
            if t > shortest_path:  # share weights across time
                varscope.reuse_variables()
            logits[t] = ConvRNN.linear(input_, output_size=num_labels)

    return logits


def _first(graph):
    """
    Returns dictionary with times of when each layer first matters, that is,
    receives information from input layer.
    :param graph: networkx graph representing model
    :return: dictionary first[j] = time t where layer j matters. j ranges
    from 0 to N_cells + 1
    """
    first = {}
    curr_ind = [0]
    t = 0
    while len(first) < graph.number_of_nodes():  # = N_cells + 2 to account for
        # input and final logits layer
        next_ind = []  # layers reached at next time point
        # for current indices, check if already accounted for in first
        for ind in curr_ind:
            if ind not in first:
                first[ind] = t
                # then add all neighbors to next_ind
                next_ind.extend(graph.successors(ind))
        curr_ind = next_ind
        t += 1
    return first


def _last(graph, T_tot):
    """
    Returns dictionary with times of when each layer last matters, that is,
    last time t where layer j information reaches output layer at, before T_tot
    Note last[0] >= 0, for input layer
    :param graph: networkx graph representing model
    :param T_tot: total number of time steps to run the model.
    :return: dictionary {layer j: last time t}
    """
    last = {}
    curr_ind = [graph.number_of_nodes() - 1]  # start with output layer
    t = T_tot
    while len(last) < graph.number_of_nodes():
        next_ind = []  # layers at prev time point
        for ind in curr_ind:
            if ind not in last:
                last[ind] = t
                # then add adjacency list onto next_ind
                next_ind.extend(graph.predecessors(ind))
        curr_ind = next_ind
        t -= 1
    return last


def _graph(bypasses, N_cells):
    """
    Constructs networkx DiGraph based on bypass connections
    :param bypasses: list of tuples (from, to)
    :param N_cells: number of layers not including last FC for logits
    :return: Returns a networkx DiGraph where nodes are layer #s, starting
    from 0 (input) to N_cells + 1 (last FC layer for logits)
    """
    graph = nx.DiGraph()
    regular_connections = [(j, j + 1) for j in range(0, N_cells + 1)]

    # add regular connections (aka 1->2, 2->3...)
    graph.add_edges_from(regular_connections)

    #  adjacent layers
    graph.add_edges_from(bypasses)  # add bypass connections

    # check that bypasses don't add extraneous nodes
    assert (graph.number_of_nodes() == N_cells + 2), \
        'bypasses list is not valid'
    return graph


def _flatten_input(input_):
    """
        Flattens input (if not already flattened). Use for input to FC layers
        :param input_: tf Tensor
        :return: flattened input of size [batch_size, spatial*spatial*channels]
    """
    # flatten input to [#batches, input_size]
    input_shape = input_.get_shape().as_list()
    if len(input_shape) > 2:  # needs flattening. assumed to have dim. 4
        input_size = input_shape[1] * input_shape[2] * input_shape[3]
        input_ = tf.reshape(input_, [-1, input_size])
    return input_


def _layers_to_cells(layers):
    """
    Converts dictionary of layers and parameters to actual RNNCell objects.
    Note: the RNNCell's operations have not yet been created (so no weights,
    biases, etc. exist yet)
    :param layers: dictionary {layer #: ['cell type', {arguments}]
    :return: dictionary of cells {layer #: RNN Cell}
    """
    cells = {}
    for k, fun_list in layers.iteritems():
        fun_name = fun_list[0]  # 'fun' = 'function'
        cell_cases = {'conv': ConvRNN.ConvRNNCell,
                      'convpool': ConvRNN.ConvPoolRNNCell,
                      'fc': ConvRNN.FcRNNCell}  # switch-case equivalent
        fun = cell_cases[fun_name.lower()]  # case insensitive search
        kwarg_items = fun_list[1]
        cells[k] = fun(**kwarg_items)
    return cells


def _maxpool(input, out_spatial, name='pool'):
    """ returns a tf operation for maxpool of input, with stride determined
    by the spatial size ratio of output and input"""
    in_spatial = input.get_shape().as_list()[1]
    stride = in_spatial / out_spatial  # how much to pool by
    if stride < 1:
        raise ValueError('spatial dimension of output should not be greater '
                         'than that of input')
    pool = tf.nn.max_pool(input, ksize=[1, stride, stride, 1],
                          # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME',
                          name=name)
    return pool


def _graph_initials(layer_sizes, N_cells):
    """
    Returns prev_out dict with all zeros (tensors) but first input,
    curr_states with all Nones input should be a TENSOR!! """
    prev_out = {}
    curr_states = {}
    for i in range(1, N_cells + 1):  # 1, ... N_cells
        prev_out[i] = tf.zeros(shape=layer_sizes[i]['output'],
                               dtype=tf.float32,
                               name='zero_out')  # init. zeros of correct size
        curr_states[i] = None
        # note: can also initialize state to random distribution, etc...
    return prev_out, curr_states
