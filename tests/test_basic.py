from __future__ import absolute_import, division, print_function
import time

import tqdm
import numpy as np
import tensorflow as tf

from tnn import main
from tests import setup

BATCH_SIZE = 256


def test_mnist_fc():
    test_mnist(kind='fc')


def test_mnist_conv():
    test_mnist(kind='conv')


def test_mnist(kind='conv'):
    data = {'images': np.random.standard_normal([BATCH_SIZE, 28*28]).astype(np.float32),
            'labels': np.random.randint(10, size=BATCH_SIZE).astype(np.int32)}
    if kind == 'conv':
        data['images'] = np.reshape(data['images'], [-1, 28, 28, 1])

    # initialize the benchmark model
    with tf.variable_scope('benchmark'):
        if kind == 'conv':
            bench_targets = setup.mnist_conv(**data)
        elif kind == 'fc':
            bench_targets = setup.mnist_fc(**data)
        else:
            raise ValueError

    bench_vars = {v.name[len('benchmark')+1:]:v for v in tf.global_variables()
                  if v.name.startswith('benchmark')}
    bench_targets.update(bench_vars)
    for name, var in bench_vars.items():
        bench_targets['grad_' + name] = tf.gradients(bench_targets['loss'], var)

    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        G = main.graph_from_json('json/mnist_{}.json'.format(kind))
        main.init_nodes(G, batch_size=BATCH_SIZE)
        input_seq = tf.constant(data['images'])
        main.unroll(G, input_seq=input_seq)
        tnn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=G.node['fc2']['outputs'][-1],
                                                                  labels=tf.constant(data['labels']))

    tnn_targets = {n: G.node[n]['outputs'][-1] for n in G}
    tnn_targets['loss'] = tf.reduce_mean(tnn_loss)
    tnn_vars = {v.name[len('tconvnet')+1:]:v for v in tf.global_variables()
                if v.name.startswith('tconvnet') and 'memory_decay' not in v.name}
    tnn_targets.update(tnn_vars)
    for name, var in tnn_vars.items():
        tnn_targets['grad_' + name] = tf.gradients(tnn_targets['loss'], var)

    run(bench_targets, tnn_targets, nsteps=100)


def test_alexnet():
    ims = np.random.standard_normal([BATCH_SIZE, 224, 224, 3])
    labels = np.random.randint(1000, size=[BATCH_SIZE])
    data = {'images': tf.constant(ims.astype(np.float32)),
            'labels': tf.constant(labels.astype(np.int32))}
    # initialize the benchmark model
    with tf.variable_scope('benchmark'):
        bench_targets = setup.alexnet(data['images'], data['labels'], 'benchmark', train=False)
    bench_targets = {'loss': bench_targets['loss']}

    # initialize the tconvnet model
    with tf.variable_scope('tconvnet'):
        G = main.graph_from_json('json/alexnet.json')
        main.init_nodes(G, batch_size=BATCH_SIZE)
        main.unroll(G, input_seq=data['images'])
        tnn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=G.node['fc8']['outputs'][-1], labels=data['labels'])
        tnn_targets = {'loss': tf.reduce_mean(tnn_loss)}

    run(bench_targets, tnn_targets, nsteps=10, n_initial=10)


def run(bench_targets, tnn_targets, nsteps=100, n_initial=2, n_stable=50, check_close=True):
    assert np.array_equal(sorted(tnn_targets.keys()), sorted(bench_targets.keys()))

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    bench_targets['optimizer'] = opt.minimize(bench_targets['loss'])

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    tnn_targets['optimizer'] = opt.minimize(tnn_targets['loss'])

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(init)

    for step in tqdm.trange(nsteps):
        # check if the outputs are identical
        if step < n_initial:
            bench_res = sess.run(bench_targets)
            tnn_res = sess.run(tnn_targets)
            for name in bench_res:
                if name != 'optimizer':
                    if check_close:
                        assert np.allclose(bench_res[name], tnn_res[name], atol=1e-2)
                    else:
                        assert np.array_equal(bench_res[name], tnn_res[name])
        elif step > n_stable:  # after that, the loss should be stable
            _, bench_loss = sess.run([bench_targets['optimizer'], bench_targets['loss']])
            _, tnn_loss = sess.run([tnn_targets['optimizer'], tnn_targets['loss']])
            assert np.allclose(bench_loss, tnn_loss, atol=.1, rtol=.1)
        else:
            bench_loss = sess.run(bench_targets['loss'])
            tnn_loss = sess.run(tnn_targets['loss'])
            sess.run([bench_targets['optimizer'], tnn_targets['optimizer']])

    sess.close()


def train_tnn_alexnet():
    imagenet = setup.get_imagenet()
    images_plc = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    labels_plc = tf.placeholder(tf.int64, shape=[BATCH_SIZE])

    with tf.variable_scope('tconvnet'):
        G = main.graph_from_json('json/alexnet.json')
        main.init_nodes(G, batch_size=BATCH_SIZE)
        main.unroll(G, input_seq=images_plc)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=G.node['fc8']['outputs'][-1], labels=labels_plc)
        loss = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    losses = []
    for step in range(1000):
        start = time.time()
        images_batch, labels_batch = imagenet.next()
        lo, _ = sess.run([loss, optimizer],
                         feed_dict={images_plc: images_batch, labels_plc: labels_batch})
        end = time.time()
        losses.append(lo)
        print(step, '{:.4f}'.format(lo), '{:.3f}'.format(end - start))
    assert np.mean(losses[-20:]) < 6.8


def memory_usage():
    ims = np.random.standard_normal([BATCH_SIZE, 224, 224, 3])
    labels = np.random.randint(1000, size=[BATCH_SIZE])
    data = {'images': tf.constant(ims.astype(np.float32)),
            'labels': tf.constant(labels.astype(np.int32))}
    # initialize the benchmark model
    # with tf.variable_scope('benchmark'):
    #     bench_targets = setup.alexnet(data['images'], data['labels'], 'benchmark', train=False)
    #     loss = bench_targets['loss']

    with tf.variable_scope('tconvnet'):
        G = main.graph_from_json('json/alexnet.json')
        main.init_nodes(G, batch_size=BATCH_SIZE)
        main.unroll(G, input_seq=data['images'])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=G.node['fc8']['outputs'][-1], labels=data['labels'])

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(init)
    sess.run(loss)
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    test_mnist_fc()

    tf.reset_default_graph()
    test_mnist_conv()

    tf.reset_default_graph()
    test_alexnet()