import sys, os, time

import numpy as np
import tensorflow as tf

import setup

sys.path.append('model')
from tconvnet import main

BATCH_SIZE = 256

codedir = os.path.dirname(os.path.abspath(__file__))
projdir = os.path.split(codedir)[0]


def test_mnist(mnist, kind='conv'):
    data = {'images': mnist.train.images[:BATCH_SIZE],
            'labels': mnist.train.labels[:BATCH_SIZE].astype(np.int32)}
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
        tnn_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=G.node['fc2']['outputs'][-1], labels=data['labels'])

    tnn_targets = {n: G.node[n]['outputs'][-1] for n in G}
    tnn_targets['loss'] = tf.reduce_mean(tnn_loss)
    tnn_vars = {v.name[len('tconvnet')+1:]:v for v in tf.global_variables()
                if v.name.startswith('tconvnet') and 'memory_decay' not in v.name}
    tnn_targets.update(tnn_vars)
    for name, var in tnn_vars.items():
        tnn_targets['grad_' + name] = tf.gradients(tnn_targets['loss'], var)

    run(bench_targets, tnn_targets, nsteps=100)


def test_alexnet(imagenet):
    # initialize the benchmark model
    images, labels = imagenet.next()
    images = tf.constant(images)
    labels = tf.constant(labels)
    with tf.variable_scope('benchmark'):
        bench_targets = setup.alexnet(images, labels, 'benchmark', train=False)

    bench_vars = {'/'.join(v.name.split('/')[1:]):v for v in tf.global_variables()
                  if v.name.startswith('benchmark')}

    bench_targets.update(bench_vars)

    for name, var in bench_vars.items():
        bench_targets['grad_' + name] = tf.gradients(bench_targets['loss'], var)

    # initialize the unicycle model
    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name=os.path.join(projdir, 'json', 'sample_alexnet.json'))
        out = unicycle_model({'images': images}, G)
        outp = out.get_output()
        s = outp.get_shape().as_list()
        print(s)
        logits = tf.reshape(outp, (s[0], -1))
        uni_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=labels)

    uni_targets = {
                #    'conv1': G.node['conv_1']['tf_cell'].get_output(),
                #    'conv2': G.node['conv_2']['tf_cell'].get_output(),
                #    'conv3': G.node['conv_3']['tf_cell'].get_output(),
                #    'conv4': G.node['conv_4']['tf_cell'].get_output(),
                #    'conv5': G.node['conv_5']['tf_cell'].get_output(),
                #    'fc6': G.node['fc_6']['tf_cell'].get_output(),
                #    'fc7': G.node['fc_7']['tf_cell'].get_output(),
                #    'fc8': G.node['fc_8']['tf_cell'].get_output(),
                   'loss': tf.reduce_mean(uni_loss)
                   }

    uni_vars = {}
    for v in tf.global_variables():
        if v.name.startswith('unicycle') and 'decay_param' not in v.name:
            name, varno = v.name.split('/')[1].split(':')
            name, layer, layerno = name.split('_')
            if name == 'biases':
                name = 'bias'
            uni_vars[layer + layerno + '/' + name + ':' + varno] = v

    # uni_targets.update(uni_vars)
    grads = []
    for name, var in uni_vars.items():
        _g = tf.gradients(uni_targets['loss'], var)
    #     uni_targets['grad_' + name] = _g
        grads.append([name, _g[0] == None])

    #print(G.node['conv_1']['tf_cell'].get_output(1).op.name)
    print(grads)
    return G, uni_targets, bench_targets
    #run(bench_targets, uni_targets, nsteps=100, check_close=True)


def train_uni_alexnet(imagenet):
    images_plc = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    labels_plc = tf.placeholder(tf.int64, shape=[BATCH_SIZE])

    with tf.variable_scope('unicycle'):
        unicycle_model = Unicycle()
        G = unicycle_model.build(json_file_name='sample_alexnet.json')
        out = unicycle_model({'images': images_plc}, G)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out.get_output(),
                                                              labels=labels_plc)
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


def run(bench_targets, tnn_targets, nsteps=100, check_close=True):
    assert np.array_equal(sorted(tnn_targets.keys()), sorted(bench_targets.keys()))

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    bench_targets['optimizer'] = opt.minimize(bench_targets['loss'])

    opt = tf.train.MomentumOptimizer(learning_rate=.01, momentum=.9)
    tnn_targets['optimizer'] = opt.minimize(tnn_targets['loss'])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # tf.summary.FileWriter('tensorboard', sess.graph)
    # sys.exit()

    for step in range(nsteps):
        print(step)
        # check if the outputs are identical
        if step < 2:
            bench_res = sess.run(bench_targets)
            tnn_res = sess.run(tnn_targets)

            for name in bench_res:
                if name != 'optimizer':
                    if check_close:
                        try:
                            assert np.allclose(bench_res[name], tnn_res[name], atol=1e-2, rtol=1e-2)
                        except:
                            print(step, name, np.sum((np.abs(bench_res[name]) - np.abs(tnn_res[name]))**2))
                    else:
                        assert np.array_equal(bench_res[name], tnn_res[name])
        elif step > 50:  # after that, the loss should be stable
            _, bench_loss = sess.run([bench_targets['optimizer'], bench_targets['loss']])
            _, tnn_loss = sess.run([tnn_targets['optimizer'], tnn_targets['loss']])
            assert np.allclose(bench_loss, tnn_loss, atol=1e-3, rtol=1e-3)
        else:
            sess.run([bench_targets['optimizer'], tnn_targets['optimizer']])

    sess.close()


if __name__ == '__main__':
    mnist = setup.get_mnist_data()

    # test_mnist(mnist, kind='fc')

    # tf.reset_default_graph()
    test_mnist(mnist, kind='conv')

    #tf.reset_default_graph()
    #imagenet = setup.get_imagenet()
    #test_alexnet(imagenet)
    # train_uni_alexnet(imagenet)
