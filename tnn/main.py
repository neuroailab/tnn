from __future__ import absolute_import, division, print_function

import json
import itertools
import copy
import math

import networkx as nx
import tensorflow as tf
import numpy as np

import tfutils.model
import tnn.cell


def _get_func_from_kwargs(function, **kwargs):
    """
    Guess the function from its name
    """
    if callable(function):
        f = function
    else:
        try:
            f = getattr(tnn.cell, function)
        except:
            try:
                f = getattr(tfutils.model, function)
            except:
                try:
                    f = getattr(tf.nn, function)
                except:
                    try:
                        f = getattr(tf, function)
                    except:
                        f = getattr(tf.contrib.layers, function)
    return f, kwargs


def import_json(json_file_name):
    with open(json_file_name) as f:
        json_data = json.load(f)

    assert 'nodes' in json_data, 'nodes field not in the json file'
    assert len(json_data['nodes']) > 0, 'no nodes in the json file'
    assert 'edges' in json_data, 'edges field not in the json file'

    edges = [(str(i['from']), str(i['to'])) for i in json_data['edges']]
    node_names = []
    for node in json_data['nodes']:
        assert 'name' in node
        node_names.append(node['name'])
    if len(edges) != 0:
        assert set(itertools.chain(*edges)) == set(node_names), 'nodes and edges do not match'

    return json_data['nodes'], edges


def graph_from_json(json_file_name):
    json_nodes, edges = import_json(json_file_name)

    if len(edges) == 0: # only one node in the graph
       edges = {}
       for json_node in json_nodes:
          edges[json_node['name']] = {}

    G = nx.DiGraph(data=edges)
    for json_node in json_nodes:
        attr = G.node[json_node['name']]

        if 'shape' in json_node:
            attr['shape'] = json_node['shape']
        elif 'shape_from' in json_node:
            attr['shape_from'] = json_node['shape_from']
        if 'dtype' in json_node:
            attr['dtype'] = json_node['dtype']

        attr['cell'] = tnn.cell.GenFuncCell
        attr['kwargs'] = {}
        attr['kwargs']['harbor'] = _get_func_from_kwargs(**json_node['harbor'])
        attr['kwargs']['pre_memory'] = []
        for kwargs in json_node['pre_memory']:
            attr['kwargs']['pre_memory'].append(_get_func_from_kwargs(**kwargs))
        attr['kwargs']['memory'] = _get_func_from_kwargs(**json_node['memory'])
        attr['kwargs']['post_memory'] = []
        for kwargs in json_node['post_memory']:
            attr['kwargs']['post_memory'].append(_get_func_from_kwargs(**kwargs))
        attr['kwargs']['input_init'] = _get_func_from_kwargs(**json_node['input_init'])
        attr['kwargs']['state_init'] = _get_func_from_kwargs(**json_node['state_init'])
        attr['kwargs']['dtype'] = json_node['dtype']
        attr['kwargs']['name'] = json_node['name']

    return G


def check_inputs(G, input_nodes):
    '''Given a networkx graph G and a set of input_nodes,
    checks whether the inputs are valid'''

    for n in input_nodes:
        if n not in G.nodes():
            raise ValueError('The input nodes provided must all be in the graph.')

    input_cover = set([])
    for n in input_nodes:
        input_cover |= (set([n]) | set(nx.descendants(G, n)))
    if input_cover != set(G.nodes()):
        missed_nodes = ', '.join(list(set(G.nodes()) - input_cover))
        raise ValueError('Not all valid input nodes have been provided, as the following nodes will not receive any data: {}'.format(missed_nodes))


def init_nodes(G, input_nodes, batch_size=256, channel_op='concat', to_exclude=None):
    """
    Note: Modifies G in place
    """
    check_inputs(G, input_nodes)

    with tf.Graph().as_default():  # separate graph that we'll destroy right away
        # find output and harbor sizes for input nodes
        for node in input_nodes:
            attr = G.node[node]
            if 'shape' not in attr:
                raise ValueError('input node {} must have "shape" defined'.format(node))

            kwargs = attr['kwargs']
            shape = [batch_size] + attr['shape']
            kwargs['harbor_shape'] = shape
            output, state = attr['cell'](**kwargs)()
            attr['output_shape'] = output.shape.as_list()

        # find output and initial harbor sizes for input nodes
        init_nodes = copy.copy(input_nodes)
        while len(init_nodes) < len(G):
            nodes = []
            for node in init_nodes:
                nodes += [n for n in G.successors(node) if n not in init_nodes]

            for node in set(nodes):
                shape_from = G.node[node]['shape_from']
                if shape_from in init_nodes:
                    nodes.pop(nodes.index(node))
                    init_nodes.append(node)
                    kwargs = G.node[node]['kwargs']
                    kwargs['harbor_shape'] = G.node[shape_from]['output_shape'][:]
                    output, state = G.node[node]['cell'](**kwargs)()
                    G.node[node]['output_shape'] = output.shape.as_list()

    # now correct harbor sizes to the final sizes and initialize cells
    for node, attr in G.nodes(data=True):
        if node not in input_nodes:
            exclude_preds = []
            if to_exclude is not None and node in to_exclude.keys():
                exclude_preds = to_exclude[node]
                if not isinstance(exclude_preds, list):
                    exclude_preds = [exclude_preds]
            pred_shapes = [G.node[pred]['output_shape'] for pred in G.predecessors(node) if pred not in exclude_preds]
            attr['kwargs']['harbor_shape'] = harbor_policy(pred_shapes,
                                                           attr['kwargs']['harbor_shape'], channel_op=channel_op)

        attr['cell'] = attr['cell'](**attr['kwargs'])

def harbor_policy(in_shapes, shape, channel_op='concat'):
    nchnls = []
    if len(shape) == 4:
        for shp in in_shapes:
            if len(shp) == 4:
                c = shp[-1]
            elif len(shp) == 2:
                c = shape[3]
            nchnls.append(c)
    elif len(shape) == 2:
        for shp in in_shapes:
            c = np.prod(shp[1:])
            nchnls.append(c)
    if channel_op != 'concat':
        return shape
    else:
        return shape[:-1] + [sum(nchnls)]

def unroll(G, input_seq, ntimes=None):
    """
    Unrolls a TensorFlow graph in time

    Given a NetworkX DiGraph, connects states and outputs over time for `ntimes`
    steps, producing a TensorFlow graph unrolled in time.

    :Args:
        - G
            NetworkX DiGraph that stores initialized GenFuncCell in 'cell' nodes
        - input_seq (dict)
            A dict of inputs that specifies the input for each input node as its keys
    :Kwargs:
        - ntimes (int or None, default: None)
            The number of time steps
    """
    # find the longest path from the inputs to the outputs:
    input_nodes = input_seq.keys()
    check_inputs(G, input_nodes)
    output_nodes = [n for n in G if len(G.successors(n))==0]
    inp_out = itertools.product(input_nodes, output_nodes)
    paths = []
    for inp, out in inp_out:
        paths.extend([p for p in nx.all_simple_paths(G, inp, out)])

    path_lengths = map(len, paths)
    longest_path_len = max(path_lengths) if path_lengths else 0

    if ntimes is None:
        ntimes = longest_path_len + 1
        print('Using a default ntimes of: ', ntimes) # useful for logging

    for k in input_seq.keys():
        input_val = input_seq[k]
        if not isinstance(input_val, (tuple, list)):
            input_seq[k] = [input_val] * ntimes

    for node, attr in G.nodes(data=True):
        attr['outputs'] = []
        attr['states'] = []

    for t in range(ntimes):  # Loop over time
        for node, attr in G.nodes(data=True):  # Loop over nodes
            if t == 0:
                inputs = []
                if node in input_nodes:
                    inputs.append(input_seq[node][t])
                for pred in sorted(G.predecessors(node)):
                    cell = G.node[pred]['cell']
                    output_shape = G.node[pred]['output_shape']
                    _inp = cell.input_init[0](shape=output_shape,
                                                name=pred + '/standin',
                                                **cell.input_init[1])
                    inputs.append(_inp)

                if all([i is None for i in inputs]):
                    inputs = None
                state = None
            else:
                inputs = []
                if node in input_nodes:
                    inputs.append(input_seq[node][t])
                for pred in sorted(G.predecessors(node)):
                    inputs.append(G.node[pred]['outputs'][t-1])
                state = attr['states'][t-1]

            output, state = attr['cell'](inputs=inputs, state=state)
            attr['outputs'].append(output)
            attr['states'].append(state)


def unroll_tf(G, input_seq, ntimes=None, ff_order=None):
    """
    Unrolls a TensorFlow graph in time, but differs from the unroll() in that
    a full feedforward pass occurs at each timestep (as in the default 
    Tensorflow RNN unroller)

    Given a NetworkX DiGraph, connects states and outputs over time for `ntimes`
    steps, producing a TensorFlow graph unrolled in time.

    :Args:
        - G
            NetworkX DiGraph that stores initialized GenFuncCell in 'cell' nodes
        - input_seq (dict)
            A dict of inputs that specifies the input for each input node as its keys
    :Kwargs:
        - ntimes (int or None, default: None)
            The number of time steps
        - ff_order (list or None, default: None)
            The default feedforward order of the nodes
            If set to None, the unroller will first try to topologically sort the nodes.
            However, if there are feedbacks, this will fail, so it will pick the union of
            simple paths from input to output, and print it. Thus, you can only set ff_order
            if you have feedbacks and do not want the unroller to pick a path for you.
    """
    # find the longest path from the inputs to the outputs:
    input_nodes = input_seq.keys()
    check_inputs(G, input_nodes)
    output_nodes = [n for n in G if len(G.successors(n))==0]
    inp_out = itertools.product(input_nodes, output_nodes)
    paths = []
    for inp, out in inp_out:
        paths.extend([p for p in nx.all_simple_paths(G, inp, out)])

    path_lengths = map(len, paths)
    longest_path_len = max(path_lengths) if path_lengths else 0

    if ntimes is None:
        ntimes = longest_path_len + 1
        print('Using a default ntimes of: ', ntimes) # useful for logging

    for k in input_seq.keys():
        input_val = input_seq[k]
        if not isinstance(input_val, (tuple, list)):
            input_seq[k] = [input_val] * ntimes

    node_attr = {}
    for node, attr in G.nodes(data=True):
        attr['outputs'] = []
        attr['states'] = []
        node_attr[node] = attr
    
    s = topological_sort(G, ff_order=ff_order, longest_path_len=longest_path_len, node_attr=node_attr)

    for t in range(ntimes):  # Loop over time
        for node in s:  # Loop over nodes in topological order
            attr = node_attr[node]
            if t == 0:
                inputs = []
                if node in input_nodes:
                    inputs.append(input_seq[node][t])
                for pred in sorted(G.predecessors(node)):
                    pred_idx = s.index(pred)
                    curr_idx = s.index(node)
                    if curr_idx > pred_idx: # pred is feedforward or skip input
                        _inp = G.node[pred]['outputs'][t]
                    else: # pred is feedback, so we initialize it to 0
                        cell = G.node[pred]['cell']
                        output_shape = G.node[pred]['output_shape']
                        _inp = cell.input_init[0](shape=output_shape,
                                                name=pred + '/standin',
                                                **cell.input_init[1])

                    inputs.append(_inp)

                if all([i is None for i in inputs]):
                    inputs = None
                state = None
            else:
                inputs = []
                if node in input_nodes:
                    inputs.append(input_seq[node][t])
                for pred in sorted(G.predecessors(node)):
                    pred_idx = s.index(pred)
                    curr_idx = s.index(node)
                    if curr_idx > pred_idx: # pred is feedforward or skip input
                        inputs.append(G.node[pred]['outputs'][t])
                    else: # pred is feedback, so we get its output at t-1
                        inputs.append(G.node[pred]['outputs'][t-1])

                state = attr['states'][t-1]

            output, state = attr['cell'](inputs=inputs, state=state)
            attr['outputs'].append(output)
            attr['states'].append(state)
