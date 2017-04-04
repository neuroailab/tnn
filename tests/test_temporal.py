from __future__ import absolute_import, division, print_function

import os
import numpy as np
import tensorflow as tf
import math

from tnn import main

BATCH_SIZE = 256
MEM = .5
SEED = 0

this_dir = os.path.dirname(os.path.realpath(__file__))
json_dir = os.path.join(os.path.split(this_dir)[0], 'json')


def test_memory():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 28, 28, 1]).astype(np.float32))

    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        for node, attr in G.nodes(data=True):
            if node in ['conv1', 'conv2']:
                attr['kwargs']['memory'][1]['memory_decay'] = MEM
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images}, ntimes=6)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    graph = tf.get_default_graph()

    conv1_state = np.zeros(G.node['conv1']['states'][0].get_shape().as_list())
    conv2_state = np.zeros(G.node['conv2']['states'][0].get_shape().as_list())
    state1, state2 = sess.run([G.node['conv1']['states'], G.node['conv2']['states']])
    for i, (s1, s2) in enumerate(zip(state1, state2)):
        if i == 0:
            state1_inp = graph.get_tensor_by_name('tconvnet/conv1/conv:0')
            state2_inp = graph.get_tensor_by_name('tconvnet/conv2/conv:0')
        else:
            state1_inp = graph.get_tensor_by_name('tconvnet/conv1_{}/conv:0'.format(i))
            state2_inp = graph.get_tensor_by_name('tconvnet/conv2_{}/conv:0'.format(i))
        state1_inp, state2_inp = sess.run([state1_inp, state2_inp])

        conv1_state = conv1_state * MEM + state1_inp
        assert np.allclose(s1, conv1_state)
        conv2_state = conv2_state * MEM + state2_inp
        assert np.allclose(s2, conv2_state)

    sess.close()


def test_state_and_output_sizes(G):
    assert G.node['conv1']['cell'].state.shape.as_list() == [BATCH_SIZE, 54, 54, 96]
    assert G.node['conv2']['cell'].state.shape.as_list() == [BATCH_SIZE, 27, 27, 256]
    assert G.node['conv3']['cell'].state.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv4']['cell'].state.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv5']['cell'].state.shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['fc6']['cell'].state.shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc7']['cell'].state.shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc8']['cell'].state.shape.as_list() == [BATCH_SIZE, 1000]

    assert G.node['conv1']['cell'].output.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    assert G.node['conv2']['cell'].output.shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    assert G.node['conv3']['cell'].output.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv4']['cell'].output.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    assert G.node['conv5']['cell'].output.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    assert G.node['fc6']['cell'].output.shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc7']['cell'].output.shape.as_list() == [BATCH_SIZE, 4096]
    assert G.node['fc8']['cell'].output.shape.as_list() == [BATCH_SIZE, 1000]


def test_bypass():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 224, 224, 3]).astype(np.float32))
    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        G.add_edges_from([('conv1', 'conv3'), ('conv1', 'conv5'), ('conv3', 'conv5')])
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images})

    test_state_and_output_sizes(G)

    graph = tf.get_default_graph()

    # harbor output sizes
    harbor = graph.get_tensor_by_name('tconvnet/conv1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('tconvnet/conv2/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('tconvnet/conv3/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96+256]
    harbor = graph.get_tensor_by_name('tconvnet/conv4/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/conv5/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96+384+384]
    harbor = graph.get_tensor_by_name('tconvnet/fc6/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc7/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('tconvnet/fc8/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]

    # check if harbor outputs at t are equal to the concat of outputs
    # from incoming nodes at t-1

    # layer 3 gets inputs from 1 and 2
    conv3h = graph.get_tensor_by_name('tconvnet/conv3_5/harbor:0')
    conv1o = G.node['conv1']['outputs'][4]
    conv1om = tf.image.resize_images(conv1o, size=conv3h.shape[1:3])
    conv2o = G.node['conv2']['outputs'][4]
    concat = tf.concat([conv1om, conv2o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv3hr, concatr = sess.run([conv3h, concat])
        assert np.array_equal(conv3hr, concatr)

    # layer 5 gets inputs from 1, 3, 4
    conv5h = graph.get_tensor_by_name('tconvnet/conv5_7/harbor:0')
    conv1o = G.node['conv1']['outputs'][6]
    conv1om = tf.image.resize_images(conv1o, size=conv5h.shape[1:3])
    conv3o = G.node['conv3']['outputs'][6]
    conv4o = G.node['conv4']['outputs'][6]
    # import pdb; pdb.set_trace()
    concat = tf.concat([conv1om, conv3o, conv4o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv5hr, concatr = sess.run([conv5h, concat])
        assert np.array_equal(conv5hr, concatr)


def test_bypass2():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 224, 224, 3]).astype(np.float32))
    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        G.add_edges_from([('conv1', 'conv3'), ('conv2', 'conv4')])
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images})

    test_state_and_output_sizes(G)

    graph = tf.get_default_graph()

    harbor = graph.get_tensor_by_name('tconvnet/conv1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('tconvnet/conv2/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('tconvnet/conv3/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 96 + 256]
    harbor = graph.get_tensor_by_name('tconvnet/conv4/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384 + 256]
    harbor = graph.get_tensor_by_name('tconvnet/conv5/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/fc6/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc7/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('tconvnet/fc8/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]

    # layer 3 gets inputs from 1 and 2
    conv3h = graph.get_tensor_by_name('tconvnet/conv3_5/harbor:0')
    conv1o = G.node['conv1']['outputs'][4]
    conv1om = tf.image.resize_images(conv1o, size=conv3h.shape[1:3])
    conv2o = G.node['conv2']['outputs'][4]
    concat = tf.concat([conv1om, conv2o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv3hr, concatr = sess.run([conv3h, concat])
        assert np.array_equal(conv3hr, concatr)


    # layer 4 gets inputs from 2 and 3
    conv4h = graph.get_tensor_by_name('tconvnet/conv4_6/harbor:0')
    conv2o = G.node['conv2']['outputs'][5]
    conv2om = tf.image.resize_images(conv2o, size=conv4h.shape[1:3])
    conv3o = G.node['conv3']['outputs'][5]
    concat = tf.concat([conv2om, conv3o], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv4hr, concatr = sess.run([conv4h, concat])
        assert np.array_equal(conv4hr, concatr)


def test_bypass3():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 224, 224, 3]).astype(np.float32))
    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        G.add_edges_from([('conv5', 'fc7')])
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images})

    test_state_and_output_sizes(G)

    graph = tf.get_default_graph()

    harbor = graph.get_tensor_by_name('tconvnet/conv1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('tconvnet/conv2/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('tconvnet/conv3/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    harbor = graph.get_tensor_by_name('tconvnet/conv4/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/conv5/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/fc6/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc7/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096 + 7 * 7 * 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc8/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]


def test_feedback2():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 224, 224, 3]).astype(np.float32))
    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        G.add_edges_from([('fc7', 'conv5')])
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images})

    test_state_and_output_sizes(G)

    graph = tf.get_default_graph()

    harbor = graph.get_tensor_by_name('tconvnet/conv1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('tconvnet/conv2/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('tconvnet/conv3/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 256]
    harbor = graph.get_tensor_by_name('tconvnet/conv4/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/conv5_1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384 + int(math.ceil(4096 / (14 * 14)))]
    harbor = graph.get_tensor_by_name('tconvnet/fc6/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc7/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('tconvnet/fc8/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]



def test_feedback():
    images = tf.constant(np.random.standard_normal([BATCH_SIZE, 224, 224, 3]).astype(np.float32))
    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        json_path = os.path.join(json_dir, 'alexnet.json')
        G = main.graph_from_json(json_path)
        G.add_edges_from([('conv5', 'conv3'), ('conv5', 'conv4'), ('conv4', 'conv3')])
        main.init_nodes(G, input_nodes=['conv1'], batch_size=BATCH_SIZE)
        main.unroll(G, input_seq={'conv1': images})

    test_state_and_output_sizes(G)

    graph = tf.get_default_graph()

    # harbor output sizes
    harbor = graph.get_tensor_by_name('tconvnet/conv1/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 224, 224, 3]
    harbor = graph.get_tensor_by_name('tconvnet/conv2/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 27, 27, 96]
    harbor = graph.get_tensor_by_name('tconvnet/conv3/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 256+384+256]
    harbor = graph.get_tensor_by_name('tconvnet/conv4/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384+256]
    harbor = graph.get_tensor_by_name('tconvnet/conv5/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 14, 14, 384]
    harbor = graph.get_tensor_by_name('tconvnet/fc6/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 7, 7, 256]
    harbor = graph.get_tensor_by_name('tconvnet/fc7/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]
    harbor = graph.get_tensor_by_name('tconvnet/fc8/harbor:0')
    assert harbor.shape.as_list() == [BATCH_SIZE, 4096]

    # check if harbor outputs at t are equal to the concat of outputs
    # from incoming nodes at t-1

    # layer 4 gets inputs from 5 and 3
    conv4h = graph.get_tensor_by_name('tconvnet/conv4_5/harbor:0')
    conv3o = G.node['conv3']['outputs'][4]
    conv5o = G.node['conv5']['outputs'][4]
    conv5om = tf.image.resize_images(conv5o, conv4h.shape.as_list()[1:3])

    concat = tf.concat([conv3o, conv5om], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv4hr, concatr = sess.run([conv4h, concat])
        assert np.array_equal(conv4hr, concatr)

    # layer 3 gets inputs from 2, 4, 5
    conv3h = graph.get_tensor_by_name('tconvnet/conv3_7/harbor:0')
    conv2o = G.node['conv2']['outputs'][6]
    conv5o = G.node['conv5']['outputs'][6]
    conv5om = tf.image.resize_images(conv5o, conv3h.shape.as_list()[1:3])
    conv4o = G.node['conv4']['outputs'][6]
    conv4om = tf.image.resize_images(conv4o, conv3h.shape.as_list()[1:3])

    concat = tf.concat([conv2o, conv4om, conv5om], axis=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv3hr, concatr = sess.run([conv3h, concat])
        assert np.array_equal(conv3hr, concatr)


if __name__ == '__main__':
#    test_memory()

#    tf.reset_default_graph()
#    test_bypass()

#    tf.reset_default_graph()
#    test_feedback()

#    tf.reset_default_graph()
    test_feedback2()

