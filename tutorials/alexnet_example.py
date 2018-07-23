from __future__ import absolute_import, division, print_function

import tensorflow as tf

from tnn import main


def loss(G):
    output_nodes = [n for n in G if len(G.successors(n)) == 0]

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


G = main.graph_from_json('../json/alexnet.json')
main.init_nodes(G, input_nodes=['conv1'], batch_size=256)
input_images = tf.zeros([256,224,224,3])
# input_dict = {'images': tf.zeros([256,224,224,3]),
#                 'labels': tf.zeros([256], dtype=tf.int64)}
main.unroll(G, input_seq={'conv1': input_images}, ntimes=9)
# loss(G)
