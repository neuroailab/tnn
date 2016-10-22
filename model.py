"""
The "model" function is used to create a TF graph given layers and bypasses
"""

from __future__ import absolute_import, division, print_function

import networkx as nx
import tensorflow as tf

from conv_rnn_cell import ConvRNNCell


def alexnet(weight_decay=.0005, memory_decay=None, dropout=.5,
            init_weights='xavier', train=True):
    dropout = dropout if train else None
    image_size_crop = 227

    ## make initial image size also a parameter? so we can define
    # layer spatial sizes in terms of it.

    class Conv1(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(Conv1, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, image_size_crop // 4,
                                image_size_crop // 4, 96]
            self._output_size = [self.batch_size, image_size_crop // 8,
                                 image_size_crop // 8, 96]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            conv = self.conv(inputs, 96, 11, 4, stddev=.1, bias=.1,
                             init=init_weights, weight_decay=weight_decay)
            norm = self.lrn(conv)
            new_state = self.memory(norm, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            pool = self.pool(relu, 3, 2)
            return pool, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class Conv2(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(Conv2, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, image_size_crop // 8,
                                image_size_crop // 8, 256]
            self._output_size = [self.batch_size, image_size_crop // 16,
                                 image_size_crop // 16, 256]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            conv = self.conv(inputs, 256, 5, 1, stddev=.1, bias=.1,
                             init=init_weights, weight_decay=weight_decay)
            norm = self.lrn(conv)
            new_state = self.memory(norm, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            pool = self.pool(relu, 3, 2)
            return pool, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class Conv3(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(Conv3, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, image_size_crop // 16,
                                image_size_crop // 16, 384]
            self._output_size = [self.batch_size, image_size_crop // 16,
                                 image_size_crop // 16, 384]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            conv = self.conv(inputs, 384, 3, 1, stddev=.1, bias=.1,
                             init=init_weights, weight_decay=weight_decay)
            new_state = self.memory(conv, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            return relu, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class Conv4(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(Conv4, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, image_size_crop // 16,
                                image_size_crop // 16, 384]
            self._output_size = [self.batch_size, image_size_crop // 16,
                                 image_size_crop // 16, 384]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            conv = self.conv(inputs, 384, 3, 1, stddev=.1, bias=.1,
                             init=init_weights, weight_decay=weight_decay)
            new_state = self.memory(conv, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            return relu, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class Conv5(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(Conv5, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, image_size_crop // 16,
                                image_size_crop // 16, 256]
            self._output_size = [self.batch_size, image_size_crop // 32,
                                 image_size_crop // 32, 256]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            conv = self.conv(inputs, 256, 3, 1, stddev=.1, bias=.1,
                             init=init_weights, weight_decay=weight_decay)
            new_state = self.memory(conv, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            pool = self.pool(relu, 3, 2)
            return pool, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class FC6(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(FC6, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, 4096]
            self._output_size = [self.batch_size, 4096]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            resh = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1])
            fc = self.fc(resh, 4096, dropout=dropout, stddev=.1, bias=.1,
                         init=init_weights)
            new_state = self.memory(fc, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            drop = self.dropout(relu, dropout=dropout)
            return drop, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class FC7(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(FC7, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, 4096]
            self._output_size = [self.batch_size, 4096]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            fc = self.fc(inputs, 4096, dropout=dropout, stddev=.1, bias=.1,
                         init=init_weights)
            new_state = self.memory(fc, state, memory_decay=memory_decay)
            relu = self.relu(new_state)
            drop = self.dropout(relu, dropout=dropout)
            return drop, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    class FC8(ConvRNNCell):
        def __init__(self, *args, **kwargs):
            super(FC8, self).__init__(*args, **kwargs)
            self._state_size = [self.batch_size, 1000]
            self._output_size = [self.batch_size, 1000]

        def __call__(self, inputs, state):
            # with tf.variable_scope(type(self).__name__, reuse=True):
            fc = self.fc(inputs, 1000, dropout=None, stddev=.1, bias=.1,
                         init=init_weights)
            new_state = self.memory(fc, state, memory_decay=memory_decay)
            # relu = self.relu(new_state) # no relu in final layer
            return new_state, new_state

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._output_size

    layers = [Conv1, Conv2, Conv3, Conv4, Conv5, FC6, FC7, FC8]

    return layers


def get_model(input_seq,
              model_base=None,
              train=False,
              bypasses=[],
              T_tot=8,
              init_weights='xavier',
              weight_decay=None,
              dropout=None,
              memory_decay=None,
            #   initial_states=None,
              num_labels=1000,
              trim_top=True,
              trim_bottom=True,
              features_layer=None,
              bypass_pool_kernel_size=None,
              batch_size=None):
    """
    Creates model graph and returns logits.
    :param layers: Dictionary to construct cells for each layer of the form
     {layer #: ['cell type', {arguments}] Does not include the final linear
     layer used to get logits.
    :param bypasses: list of tuples (from, to)
    :param input_seq: list for sequence of input images as tf Tensors
    :param T_tot: total number of time steps to run the model.
    :param features_layer: if None (equivalent to a value of len(layers) + 1) ,
     outputs logitsfrom last FC. Otherwise, accepts a number 0 through
     len(layers) + 1 and _model will output the features of that layer.
    :param initial_states: optional; dict of initial state {layer#: tf Tensor}
    :param num_labels: Size of logits to output [1000 for ImageNet]
    :return: Returns a dictionary logits (output of a linear FC layer after
    all layers). {time t: logits} for t >= shortest_path and t < T_total}
    """

    # create networkx graph with layer #s as nodes
    layers = model_base(weight_decay=weight_decay, memory_decay=memory_decay,
                        dropout=dropout, init_weights=init_weights, train=train)
    graph = _construct_graph(layers, bypasses, batch_size)

    nlayers = len(layers)  # number of layers including the final FC/logits
    ntimes = len(input_seq)

    # ensure that graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('graph not acyclic')

    # ensure that T_tot >= shortest_path through graph
    shortest_path = nx.shortest_path_length(graph, source='0',
                                            target=str(nlayers))
    if ntimes < shortest_path:
        raise ValueError('T_tot ({}) < shortest path length ({})'.format(T_tot,
                                                                 shortest_path))

    # get first and last time points where layers matter
    if trim_top:
        _first(graph)
    else:
        graph.node['0']['first'] = 0  # input matters at t = 0,
        # rest starting t = 1
        for node in graph:
            graph.node[node]['first'] = 1

    if trim_bottom:
        _last(graph, ntimes)
    else:
        for node in graph:
            graph.node[node]['last'] = ntimes

    # check inputs: Compares input sequence length with the input length
    # that is needed for output at T_tot. Zero pads or truncates as needed
    # TODO: can this scenario happen?
    if len(input_seq) > graph.node['0']['last'] + 1:  # more inputs than needed => truncate
        print('truncating input sequence to length', graph.node['0']['last'] + 1)
        del input_seq[graph.node['0']['last'] + 1:]
    elif len(input_seq) < graph.node['0']['last'] + 1:  # too short => pad with zero inputs
        print('zero-padding input sequence to length', graph.node['0']['last'] + 1)
        num_needed = (graph.node['0']['last'] + 1) - len(input_seq)  # inputs to add
        if not input_seq:  # need input length of at least one
            raise ValueError('input sequence should not be empty')
        padding = [tf.zeros_like(input_seq[0]) for i in range(0, num_needed)]
        input_seq.extend(padding)

    # add inputs to outputs dict for layer 0
    # outputs = {layer#: {t1:__, t2:__, ... }}
    graph.node['0']['inputs'] = None
    graph.node['0']['outputs'] = [i['data'] for i in input_seq]
    graph.node['0']['initial_states'] = None
    graph.node['0']['final_states'] = None

    # create zero initial states if none specified
    # if initial_states is None:
    # zero state returns zeros (tf.float32) based on state size.
    for layer in graph:
        if layer == '0':
            st = None
        else:
            st = graph.node[layer]['cell'].zero_state(None, None)
        graph.node[layer]['initial_states'] = st

    reuse = None if train else True
    # create graph layer by layer
    for n, node in enumerate(sorted(graph.nodes())):
        if node != '0':
            layer = graph.node[node]
            with tf.variable_scope(layer['name'], reuse=reuse):
                print('{:-^80}'.format(layer['name']))
                # create inputs list for layer j, each element is an input in time
                layer['inputs'] = []  # list of inputs to layer j in time
                parents = graph.predecessors(node)  # list of incoming nodes

                # for relevant time points: gather inputs, pool, and concatenate
                for t in range(layer['first'], layer['last'] + 1):
                    # concatenate inputs (pooled to right spatial size) at time t
                    # incoming_shape = layer_sizes[j - 1]['output']  # with no bypass
                    # if node == 5 and t == 2: import pdb; pdb.set_trace()
                    if len(layer['cell'].state_size) == 4:
                        if n == 1:
                            output_size = graph.node['0']['outputs'][0].get_shape().as_list()[1]
                        else:
                            output_size = graph.node[str(n-1)]['cell'].output_size[1]

                        inputs_t = []
                        for parent in sorted(parents):
                            input_tp = _maxpool(
                                input_=graph.node[parent]['outputs'][t - 1],
                                out_spatial=output_size,
                                kernel_size=bypass_pool_kernel_size,
                                name='bypass_pool')
                            inputs_t.append(input_tp)

                        # concat in channel dim
                        layer['inputs'].append(tf.concat(3, inputs_t))

                    else:  # if input is 2D (ex: after FC layer)
                        print('FC')
                        # no bypass to FC layers beyond first FC
                        if len(parents) != 1:
                            raise ValueError('No bypass to FC layers '
                                             'beyond first FC allowed')
                        inputs_t = graph.node[parents[0]]['outputs'][t - 1]
                        layer['inputs'].append(inputs_t)
                    print('inputs at t = {}: {}'.format(t, inputs_t))

                # run tf.nn.rnn and get list of outputs
                # Even if initial_states[j] is None, tf.nn.rnn will just set
                # zero initial state (given dtype)
                if len(layer['inputs']) > 0:
                    out, fstate = tf.nn.rnn(cell=layer['cell'],
                                            inputs=layer['inputs'],
                                            initial_state=layer['initial_states'],
                                            dtype=tf.float32)
                else:  # if empty, layer j doesn't contribute to output t<= T_tot
                    fstate = layer['initial_states']

                layer['outputs'] = out
                layer['final_states'] = fstate

    if features_layer is None:
        return graph.node[str(len(layers))]['outputs']
    else:
        return graph[features_layer]['outputs']


def _construct_graph(layers, bypasses, batch_size):
    """
    Constructs networkx DiGraph based on bypass connections
    :param bypasses: list of tuples (from, to)
    :param N_cells: number of layers not including last FC for logits
    :return: Returns a networkx DiGraph where nodes are layer #s, starting
    from 0 (input) to N_cells + 1 (last FC layer for logits)
    """
    graph = nx.DiGraph()
    nlayers = len(layers)
    graph.add_node('0', cell=None)
    prev_node = '0'
    names = []
    for node, layer in enumerate(layers):
        node = str(node + 1)
        cell = layer(batch_size) ## only need to specify batch size
        graph.add_node(node, cell=cell, name=cell.scope)
        graph.add_edge(str(int(node)-1), node)

    #  adjacent layers
    # graph.add_edges_from([(names[i], names[j]) for i,j in bypasses])  # add bypass connections
    graph.add_edges_from([(str(i), str(j)) for i,j in bypasses])
    print(graph.nodes())

    # check that bypasses don't add extraneous nodes
    if len(graph) != nlayers:
        import pdb; pdb.set_trace()
        raise ValueError('bypasses list created extraneous nodes')

    return graph


def _first(graph):
    """
    Returns dictionary with times of when each layer first matters, that is,
    receives information from input layer.
    :param graph: networkx graph representing model
    :return: dictionary first[j] = time t where layer j matters. j ranges
    from 0 to N_cells + 1
    """
    curr_layers = ['0']
    t = 0
    while len(curr_layers) > 0:
        next_layers = []
        for layer in curr_layers:
            if 'first' not in graph.node[layer]:
                graph.node[layer]['first'] = t
                next_layers.extend(graph.successors(layer))
        curr_layers = next_layers
        t += 1


def _last(graph, ntimes):
    """
    Returns dictionary with times of when each layer last matters, that is,
    last time t where layer j information reaches output layer at, before T_tot
    Note last[0] >= 0, for input layer
    :param graph: networkx graph representing model
    :param T_tot: total number of time steps to run the model.
    :return: dictionary {layer j: last time t}
    """
    curr_layers = [str(len(graph) - 1)]  # start with output layer
    t = ntimes
    while len(curr_layers) > 0:
        next_layers = []  # layers at prev time point
        for layer in curr_layers:
            if 'last' not in graph.node[layer]:
                graph.node[layer]['last'] = t
                # then add adjacency list onto next_layer
                next_layers.extend(graph.predecessors(layer))
        curr_layers = next_layers
        t -= 1


def _maxpool(input_, out_spatial, kernel_size=None, name='pool'):
    """
    Returns a tf operation for maxpool of input
    Stride determined by the spatial size ratio of output and input
    kernel_size = None will set kernel_size same as stride.
    """
    in_spatial = input_.get_shape().as_list()[1]
    stride = in_spatial // out_spatial  # how much to pool by
    if stride < 1:
        raise ValueError('spatial dimension of output should not be greater '
                         'than that of input')
    if kernel_size is None:
        kernel_size = stride
    pool = tf.nn.max_pool(input_,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME',
                          name=name)
    return pool


def get_loss(inputs, outputs, loss_fun=None, time_penalty=1.2):
    """Calculate total loss (with time penalty)"""

    if loss_fun is None:
        loss_fun = tf.nn.sparse_softmax_cross_entropy_with_logits

    losses = []
    for t, (inp, output) in enumerate(zip(inputs, outputs)):
        loss_t = tf.reduce_mean(loss_fun(output, inp['labels']),
                                name='xentropy_loss_t{}'.format(t))
        loss_t = loss_t * time_penalty**t
        tf.add_to_collection('losses', loss_t)
        losses.append(loss_t)

    # use 'losses' collection to also add weight decay loss
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss
