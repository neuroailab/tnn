"""
The "model" function is used to create a TF graph given layers and bypasses
"""

from __future__ import absolute_import, division, print_function

import networkx as nx
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

from tfutils.model import ConvNet


class ConvRNNCell(ConvNet, RNNCell):

    def __init__(self, output_size, state_size, seed=None, scope=None):
        super(ConvRNNCell, self).__init__(seed=seed)
        self.scope = type(self).__name__ if scope is None else scope
        self._output_size = output_size
        self._state_size = state_size
        self.state = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state=None):
        raise NotImplementedError

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor

        Args:
            batch_size: int, float, or unit Tensor representing the batch size.
            dtype: the data type to use for the state.
        Returns:
            A tensor of shape `state_size` filled with zeros.
        """
        zeros = tf.zeros(self.state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def memory(self, state, memory_decay=0, trainable=False, in_layer=None):
        if in_layer is None: in_layer = self.output
        initializer = tf.constant_initializer(value=memory_decay)
        mem = tf.get_variable(initializer=initializer,
                              shape=1,
                              trainable=trainable,
                              name='decay_param')
        decay_factor = tf.sigmoid(mem)
        self.output = tf.mul(state, decay_factor) + in_layer
        return self.output

    def conv(self, inputs, state,
             activation='relu',
             memory_decay=0,
             memory_trainable=False,
             *args, **kwargs):
        super(ConvRNNCell, self).conv(in_layer=inputs,
                                      out_shape=self.state_size[-1],
                                      activation=None,
                                      *args, **kwargs)
        self.state = self.memory(state,
                                 memory_decay=memory_decay,
                                 trainable=memory_trainable)
        if activation is not None:
            self.activation(kind=activation)
        name = tf.get_variable_scope().name
        self.params[name]['conv']['activation'] = activation
        self.params[name]['conv']['memory_decay'] = memory_decay
        self.params[name]['conv']['memory_trainable'] = memory_trainable
        return self.output

    def fc(self, inputs, state,
           activation='relu',
           dropout=None,
           memory_decay=0,
           memory_trainable=False,
           *args, **kwargs):
        super(ConvRNNCell, self).fc(in_layer=inputs,
                                    out_shape=self.state_size[-1],
                                    activation=None,
                                    dropout=None,
                                    *args, **kwargs)
        self.state = self.memory(state,
                                 memory_decay=memory_decay,
                                 trainable=memory_trainable)
        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            self.activation(dropout=dropout)
        name = tf.get_variable_scope().name
        self.params[name]['fc']['activation'] = activation
        self.params[name]['fc']['dropout'] = dropout
        self.params[name]['fc']['memory_decay'] = memory_decay
        self.params[name]['fc']['memory_trainable'] = memory_trainable
        return self.output


def alexnet(input_spatial_size=224,
            batch_size=256,
            init_weights='xavier',
            weight_decay=.0005,
            memory_decay=None,
            memory_trainable=False,
            dropout=.5,
            train=True,
            seed=None
            ):
    dropout = dropout if train else None

    class AlexNetCell(ConvRNNCell):

        def __init__(self, seed=None, *args, **kwargs):
            super(AlexNetCell, self).__init__(seed=seed, *args, **kwargs)

        def conv(self,
                 activation='relu',
                 init=init_weights,
                 weight_decay=weight_decay,
                 memory_decay=memory_decay,
                 memory_trainable=memory_trainable,
                 *args, **kwargs):
            super(AlexNetCell, self).conv(activation=activation,
                                          init=init_weights,
                                          weight_decay=weight_decay,
                                          memory_decay=memory_decay,
                                          memory_trainable=memory_trainable,
                                          *args, **kwargs)

        def fc(self,
               activation='relu',
               dropout=dropout,
               init=init_weights,
               weight_decay=weight_decay,
               memory_decay=memory_decay,
               memory_trainable=memory_trainable,
               *args, **kwargs):
            super(AlexNetCell, self).conv(activation=activation,
                                          dropout=dropout,
                                          init=init,
                                          weight_decay=weight_decay,
                                          memory_decay=memory_decay,
                                          memory_trainable=memory_trainable,
                                          *args, **kwargs)


    class Conv1(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 8 - 1,
                           input_spatial_size // 8 - 1, 64]
            state_size = [batch_size, input_spatial_size // 4,
                          input_spatial_size // 4, 64]
            super(Conv1, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv1'):
                self.conv(inputs, state, ksize=11, stride=4, stddev=.01, bias=0)
                self.norm(depth_radius=4, bias=1, alpha=.001 / 9.0, beta=.75)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class Conv2(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 192]
            state_size = [batch_size, input_spatial_size // 8 - 1,
                          input_spatial_size // 8 - 1, 192]
            super(Conv2, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv2'):
                self.conv(inputs, state, ksize=5, stride=1, stddev=.01, bias=1)
                self.norm(depth_radius=4, bias=1, alpha=.001 / 9.0, beta=.75)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class Conv3(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 384]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 384]
            super(Conv3, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv3'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=0)
                return self.output, self.state

    class Conv4(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 16 - 1,
                           input_spatial_size // 16 - 1, 384]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 384]
            super(Conv4, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv4'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=1)
                return self.output, self.state

    class Conv5(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, input_spatial_size // 32,
                           input_spatial_size // 32, 256]
            state_size = [batch_size, input_spatial_size // 16 - 1,
                          input_spatial_size // 16 - 1, 256]
            super(Conv5, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('Conv5'):
                self.conv(inputs, state, ksize=3, stride=1, stddev=.01, bias=1)
                self.pool(3, 2, padding='VALID')
                return self.output, self.state

    class FC6(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 4096]
            state_size = [batch_size, 4096]
            super(FC6, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC6'):
                self.fc(inputs, state, stddev=.01, bias=1)
                return self.output, self.state

    class FC7(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 4096]
            state_size = [batch_size, 4096]
            super(FC7, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC7'):
                self.fc(inputs, state, stddev=.01, bias=1)
                return self.output, self.state

    class FC8(ConvRNNCell):
        def __init__(self):
            output_size = [batch_size, 1000]
            state_size = [batch_size, 1000]
            super(FC8, self).__init__(output_size, state_size)

        def __call__(self, inputs, state):
            with tf.variable_scope('FC8'):
                self.fc(inputs, state, activation=None, dropout=None, stddev=.01, bias=0)
                return self.output, self.state

    layers = [Conv1, Conv2, Conv3, Conv4, Conv5, FC6, FC7, FC8]

    return layers


def get_model(inputs,
              train=False,
              cfg_initial=None,
              seed=None,
              model_base=None,
              bypasses=[],
              init_weights='xavier',
              weight_decay=None,
              dropout=None,
              memory_decay=None,
              memory_trainable=False,
              trim_top=True,
              trim_bottom=True,
              features_layer=None,
              bypass_pool_kernel_size=None,
              input_spatial_size=None,
              input_seq_len=1,
              target='data'):
    """
    Creates model graph and returns logits.
    model_base: string name of model base. (Ex: 'alexnet')
    :param layers: Dictionary to construct cells for each layer of the form
     {layer #: ['cell type', {arguments}] Does not include the final linear
     layer used to get logits.
    :param bypasses: list of tuples (from, to)
    :param inputs: list for sequence of input images as tf Tensors
    :param features_layer: if None (equivalent to a value of len(layers) + 1) ,
     outputs logitsfrom last FC. Otherwise, accepts a number 0 through
     len(layers) + 1 and _model will output the features of that layer.
    :param initial_states: optional; dict of initial state {layer#: tf Tensor}
    :return: Returns a dictionary logits (output of a linear FC layer after
    all layers). {time t: logits} for t >= shortest_path and t < T_total}
    """

    # create networkx graph with layer #s as nodes
    if model_base is None: model_base = alexnet
    layers = model_base(input_spatial_size=input_spatial_size,
                        batch_size=inputs[target].get_shape().as_list()[0],
                        init_weights=init_weights,
                        weight_decay=weight_decay,
                        memory_decay=memory_decay,
                        memory_trainable=memory_trainable,
                        dropout=dropout,
                        train=train,
                        seed=seed)
    graph = _construct_graph(layers, bypasses)

    nlayers = len(layers)  # number of layers including the final FC/logits
    shortest_path = nx.shortest_path_length(graph, source='0',
                                            target=str(nlayers))
    ntimes = input_seq_len + shortest_path - 1  # total num. of time steps

    # ensure that graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('graph not acyclic')

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

    # # check inputs: Compares input sequence length with the input length
    # # that is needed for output at T_tot. Zero pads or truncates as needed
    # # TODO: can this scenario happen?
    # if len(input_seq) > graph.node['0']['last'] + 1:  # more inputs than needed => truncate
    #     print('truncating input sequence to length', graph.node['0']['last'] + 1)
    #     del input_seq[graph.node['0']['last'] + 1:]
    # elif len(input_seq) < graph.node['0']['last'] + 1:  # too short => pad with zero inputs
    #     print('zero-padding input sequence to length', graph.node['0']['last'] + 1)
    #     num_needed = (graph.node['0']['last'] + 1) - len(input_seq)  # inputs to add
    #     if not input_seq:  # need input length of at least one
    #         raise ValueError('input sequence should not be empty')
    #     padding = [{'data': tf.zeros_like(input_seq[0]['data'])}
    #                for i in range(0, num_needed)]
    #     input_seq.extend(padding)

    # add inputs to outputs dict for layer 0
    # outputs = {layer#: {t1:__, t2:__, ... }}
    graph.node['0']['inputs'] = None
    graph.node['0']['outputs'] = [inputs[target] for _ in range(input_seq_len)]
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
                import pdb; pdb.set_trace()
                # print('{:-^80}'.format(layer['name']))
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
                        # import pdb; pdb.set_trace()
                        layer['inputs'].append(tf.concat(3, inputs_t))

                    else:  # if input is 2D (ex: after FC layer)
                        # print('FC')
                        # no bypass to FC layers beyond first FC
                        if len(parents) != 1:
                            raise ValueError('No bypass to FC layers '
                                             'beyond first FC allowed')
                        inputs_t = graph.node[parents[0]]['outputs'][t - 1]
                        layer['inputs'].append(inputs_t)
                    # print('inputs at t = {}: {}'.format(t, inputs_t))

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

                # fill in empty outputs with zeros since we index by t
                out_first = []
                for t in range(0, layer['first']):
                    out_first.append(
                        tf.zeros(shape=layer['cell'].output_size,
                                 dtype=tf.float32))
                out = out_first + out

                layer['outputs'] = out
                layer['final_states'] = fstate

    for node in graph:
        if node != '0':
            layer = graph.node[node]
            layer['outputs'] = layer['outputs'][layer['first']: layer['last'] + 1]

    if features_layer is None:
        return graph.node[str(len(layers))]['outputs'], cfg_initial
    else:
        for node in graph:
            if graph.node[node]['name'] == features_layer:
                return graph.node[node]['outputs'], cfg_initial
                break
        else:
            raise ValueError('Layer {} not found'.format(features_layer))


def _construct_graph(layers, bypasses):
    """
    Constructs networkx DiGraph based on bypass connections
    :param bypasses: list of tuples (from, to)
    :param N_cells: number of layers not including last FC for logits
    :return: Returns a networkx DiGraph where nodes are layer #s, starting
    from 0 (input) to N_cells + 1 (last FC layer for logits)
    """
    graph = nx.DiGraph()
    nlayers = len(layers)
    graph.add_node('0', cell=None, name='input')
    prev_node = '0'
    names = []
    for node, layer in enumerate(layers):
        node = str(node + 1)
        cell = layer()  # initialize cell
        graph.add_node(node, cell=cell, name=cell.scope)
        graph.add_edge(str(int(node)-1), node)

    #  adjacent layers
    # graph.add_edges_from([(names[i], names[j]) for i,j in bypasses])  # add bypass connections
    graph.add_edges_from([(str(i), str(j)) for i,j in bypasses])
    # print(graph.nodes())

    # check that bypasses don't add extraneous nodes
    if len(graph) != nlayers + 1:
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
    :param ntimes: total number of time steps to run the model.
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
                          padding='VALID',
                          name=name)
    return pool


def get_loss(inputs,
             outputs,
             target,
             loss_per_case_func,
             agg_func,
             loss_func_kwargs=None,
             agg_func_kwargs=None,
             time_penalty=1.2):
    if loss_func_kwargs is None:
        loss_func_kwargs = {}
    if agg_func_kwargs is None:
        agg_func_kwargs = {}

    losses = []
    for t, out in enumerate(outputs):
        loss_t = loss_per_case_func(out, inputs[target],
                                    name='xentropy_loss_t{}'.format(t),
                                    **loss_func_kwargs)
        loss_t_mean = agg_func(loss_t, **agg_func_kwargs)
        loss_t_mean *= time_penalty**t
        losses.append(loss_t_mean)
    # use 'losses' collection to also add weight decay loss
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = sum(losses) + sum(reg_losses)

    return total_loss
