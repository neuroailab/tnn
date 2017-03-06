from __future__ import absolute_import, division, print_function

import sys, json, itertools, copy

import networkx as nx
import tensorflow as tf

sys.path.insert(0, '../../tfutils/base_class')
import tfutils.model

import tconvnet.cell


def _get_func_from_kwargs(function, **kwargs):
    """
    Guess the function from its name
    """
    try:
        f = getattr(tconvnet.cell, function)
    except:
        try:
            f = getattr(tfutils.model, function)
        except:
            try:
                f = getattr(tf.nn, function)
            except:
                f = getattr(tf, function)
    return f, kwargs


def import_json(json_file_name):
    with open(json_file_name) as f:
        json_data = json.load(f)

    assert 'nodes' in json_data, 'nodes field not in the json file'
    assert len(json_data['nodes']) > 0, 'no nodes in the json file'
    assert 'edges' in json_data, 'edges field not in the json file'
    assert len(json_data['edges']) > 0, 'no edges in the json file'

    edges = [(str(i['from']), str(i['to'])) for i in json_data['edges']]
    node_names = []
    for node in json_data['nodes']:
        assert 'name' in node
        node_names.append(node['name'])
    assert set(itertools.chain(*edges)) == set(node_names), 'nodes and edges do not match'

    return json_data['nodes'], edges, json_data['seed']


def graph_from_json(json_file_name):
    json_nodes, edges, seed = import_json(json_file_name)
    tf.set_random_seed(seed)

    G = nx.DiGraph(data=edges)
    for json_node in json_nodes:
        attr = G.node[json_node['name']]

        if 'shape' in json_node:
            attr['shape'] = json_node['shape']
        elif 'shape_from' in json_node:
            attr['shape_from'] = json_node['shape_from']
        if 'dtype' in json_node:
            attr['dtype'] = json_node['dtype']

        attr['cell'] = tconvnet.cell.GenFuncCell
        attr['kwargs'] = {}
        attr['kwargs']['harbor'] = _get_func_from_kwargs(**json_node['harbor'])
        attr['kwargs']['pre_memory'] = []
        for kwargs in json_node['pre_memory']:
            attr['kwargs']['pre_memory'].append(_get_func_from_kwargs(**kwargs))
        attr['kwargs']['memory'] = _get_func_from_kwargs(**json_node['memory'])
        attr['kwargs']['post_memory'] = []
        for kwargs in json_node['post_memory']:
            attr['kwargs']['post_memory'].append(_get_func_from_kwargs(**kwargs))
        attr['kwargs']['state_init'] = _get_func_from_kwargs(**json_node['state_init'])
        attr['kwargs']['output_init'] = _get_func_from_kwargs(**json_node['output_init'])
        attr['kwargs']['name'] = json_node['name']

    return G


def init_nodes(G, batch_size=256):
    input_nodes = [n for n in G if len(G.predecessors(n)) == 0]

    with tf.Graph().as_default():  # separate graph that we'll destroy right away
        # initialize input nodes
        for node in input_nodes:
            attr = G.node[node]
            if 'shape' not in attr:
                raise ValueError('input node {} must have "shape" defined'.format(node))

            kwargs = G.node[node]['kwargs']
            shape = [batch_size] + attr['shape']
            kwargs['input_shapes'] = [shape]
            kwargs['input_dtypes'] = [attr['dtype']]
            kwargs['harbor_shape'] = shape

            attr['cell'] = attr['cell'](**kwargs)  # iniitialize cell
            output, state = attr['cell']()
            attr['output_size'] = output.shape.as_list()
            attr['state_size'] = state.shape.as_list()
            attr['dtype'] = output.dtype

        # initialize the remaining nodes
        init_nodes = input_nodes
        while len(init_nodes) < len(G):
            nodes = []
            for node in init_nodes:
                nodes += [n for n in G.successors(node) if n not in init_nodes]

            for node in nodes:
                init_preds = [n in init_nodes for n in G.predecessors(node)]
                if all(init_preds):
                    nodes.pop(nodes.index(node))
                    init_nodes.append(node)
                    attr = G.node[node]

                    kwargs = G.node[node]['kwargs']
                    kwargs['input_shapes'] = [G.node[p]['output_size'] for p in sorted(G.predecessors(node))]
                    kwargs['input_dtypes'] = [G.node[p]['dtype'] for p in sorted(G.predecessors(node))]
                    kwargs['harbor_shape'] = G.node[attr['shape_from']]['output_size']

                    attr['cell'] = attr['cell'](**kwargs)  # iniitialize cell
                    output, state = attr['cell']()
                    attr['output_size'] = output.shape.as_list()
                    attr['state_size'] = state.shape.as_list()
                    attr['dtype'] = output.dtype


def unroll(G, input_seq, ntimes=None):
    """
    Unrolls a TensorFlow graph in time

    Given a NetworkX DiGraph, connects states and outputs over time for `ntimes`
    steps, producing a TensorFlow graph unrolled in time.

    :Args:
        - G
            NetworkX DiGraph that stores initialized GenFuncCell in 'cell' nodes
        - input_dict (dict)
            A dict of inputs
    :Kwargs:
        - ntimes (int or None, default: None)
            The number of time steps
    """

    # find the longest path from the inputs to the outputs:
    input_nodes = [n for n in G if len(G.predecessors(n)) == 0]
    output_nodes = [n for n in G if len(G.successors(n)) == 0]
    inp_out = itertools.product(input_nodes, output_nodes)
    paths = [nx.all_simple_paths(G, inp, out) for inp, out in inp_out]
    longest_path_len = max(len(list(p)) for p in itertools.chain(paths)) - 1

    if ntimes is None:
        ntimes = longest_path_len

    if not isinstance(input_seq, (tuple, list)):
        input_seq = [input_seq] * ntimes

    for node, attr in G.nodes(data=True):
        attr['outputs'] = []
        attr['states'] = []

    for t in range(ntimes):  # Loop over time
        for node, attr in G.nodes(data=True):  # Loop over nodes
            # if node not in input_nodes and node not in output_nodes:
                if t == 0:
                    inputs = []
                    if node in input_nodes:
                        inputs.append(input_seq[t])
                    else:
                        for pred in sorted(G.predecessors(node)):
                            inputs.append(None)
                    if all([i is None for i in inputs]):
                        inputs = None

                    state = None

                else:
                    inputs = []
                    if node in input_nodes:
                        inputs.append(input_seq[t])
                    else:
                        for pred in sorted(G.predecessors(node)):
                            inputs.append(G.node[pred]['outputs'][t-1])
                    state = attr['states'][t-1]

                output, state = attr['cell'](inputs=inputs, state=state)
                attr['outputs'].append(output)
                attr['states'].append(state)

        tf.get_variable_scope().reuse_variables()