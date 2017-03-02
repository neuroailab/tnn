from __future__ import absolute_import, division, print_function

import sys, json, itertools, copy

import networkx as nx
import tensorflow as tf

sys.path.insert(0, '../../tfutils/base_class')
import tfutils.model

import tconvnet.cell
from tconvnet.cell import GenFuncCell


def _get_func_from_kwargs(function, **kwargs):
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
    assert 'edges' in json_data, 'edges field not in the JSON file'
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

    G = nx.DiGraph(data=edges)
    for json_node in json_nodes:
        attr = G.node[json_node['name']]
        attr['name'] = json_node['name']
        attr['type'] = json_node['type']

        if json_node['type'] == 'cell':
            attr['shape_from'] = json_node['shape_from']

            attr['cell'] = GenFuncCell
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
            attr['kwargs']['seed'] = seed
            attr['kwargs']['name'] = json_node['name']

        elif json_node['type'] == 'output':
            attr['function'] = getattr(tf.nn, json_node['function'])

        elif json_node['type'] == 'input':
            attr['shape'] = json_node['shape']
            attr['dtype'] = getattr(tf, json_node['dtype'])

    return G


def init_sizes(G):
    input_nodes = [n for n in G if len(G.predecessors(n)) == 0]
    output_nodes = [n for n in G if len(G.successors(n)) == 0]

    # Calculate the sizes of all of the nodes
    for node in input_nodes:
        attr = G.node[node]
        attr['output_size'] = attr['shape']
        attr['state_size'] = attr['shape']
        attr['dtype'] = attr['dtype']

    init_nodes = input_nodes + output_nodes
    g = tf.Graph()  # separate graph that we'll destroy right away
    with g.as_default():
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
                    output, state = attr['cell'](**kwargs)()
                    attr['output_size'] = output.shape.as_list()
                    attr['state_size'] = state.shape.as_list()
                    attr['dtype'] = output.dtype


def unroll(G, input_dict, ntimes):

    # find the longest path from the inputs to the outputs:
    # inp_out = itertools.product(input_nodes, output_nodes)
    # paths = [nx.all_simple_paths(G, inp, out) for inp, out in inp_out]
    # longest_path_len = max(len(p) for p in itertools.chain(paths)) - 2

    # if isinstance(input_seq['images'], list):
    #     if ntimes is None:
    #         raise Exception('Cannot supply list for input_sequence AND '
    #                         'specify ntimes. Pick one of them.')
    #     ntimes = len(input_seq['images'])

    #     if len(input_seq['images']) < self.longest_path:
    #         raise Exception('The input sequence is not long enough! '
    #                         '%s is shorter than required %s' %
    #                         (len(input_seq['images']), self.longest_path_len))
    # else:
    #     if ntimes is None:
    #         ntimes = self.longest_path_len
    #     else:
    #         if ntimes < self.longest_path_len:
    #             raise Exception('Specified unroll length is not long enough! '
    #                             '%s is shorter than required %s' %
    #                             (ntimes, self.longest_path))

    input_nodes = [n for n in G if len(G.predecessors(n)) == 0]
    output_nodes = [n for n in G if len(G.successors(n)) == 0]

    for node, attr in G.nodes(data=True):
        attr['outputs'] = []
        attr['states'] = []

    for t in range(ntimes):  # Loop over time
        for node, attr in G.nodes(data=True):  # Loop over nodes
            if node not in input_nodes and node not in output_nodes:
                if t == 0:
                    inputs = []
                    for pred in sorted(G.predecessors(node)):
                        if pred in input_nodes:
                            inputs.append(input_dict[pred])
                        else:
                            inputs.append(None)
                    if all([i is None for i in inputs]):
                        inputs = None

                    state = None

                else:
                    inputs = []
                    for pred in sorted(G.predecessors(node)):
                        if pred in input_nodes:
                            inputs.append(input_dict[pred])
                        else:
                            inputs.append(G.node[pred]['outputs'][t-1])
                    state = attr['states'][t-1]

                output, state = attr['cell'](**attr['kwargs'])(inputs=inputs, state=state)
                attr['outputs'].append(output)
                attr['states'].append(state)

        for node in output_nodes:
            attr = G.node[node]
            assert len(G.predecessors(node)) == 2
            for pred in sorted(G.predecessors(node)):
                if pred == 'labels':
                    labels = input_dict['labels']
                else:  # must be logits then
                    logits = G.node[pred]['outputs'][t]

            with tf.variable_scope(attr['name']):
                loss = attr['function'](logits=logits, labels=labels)

            attr['outputs'].append(loss)

        tf.get_variable_scope().reuse_variables()


if __name__ == '__main__':
    G = graph_from_json('json/alexnet.json')
    init_sizes(G)
    input_dict = {'images': tf.zeros([256,224,224,3]),
                  'labels': tf.zeros([256], dtype=tf.int64)}
    unroll(G, input_dict=input_dict, ntimes=9)