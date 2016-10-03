from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class ConvRNNCell(RNNCell):

    def __init__(self, output_size, state_size, scope=None):
        self.scope = type(self).__name__ if scope is None else scope
        # self.seed = seed
        # outputs, new_state = self.__call__(inputs, state=None)
        # self._output_size = outputs.get_shape.as_list()
        # self._state_size = new_state.get_shape.as_list()
        self._output_size = output_size
        self._state_size = state_size

    @property
    def state_size(self):
        # actual state size.
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
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def initializer(self, kind='xavier', stddev=.1):
        if kind == 'xavier':
            initializer = tf.contrib.layers.initializers.xavier_initializer()
        elif kind == 'trunc_norm':
            initializer = tf.truncated_normal_initializer(mean=0, stddev=stddev)
        else:
            raise ValueError('Please provide an appropriate initialization '
                             'method: xavier or trunc_norm')
        return initializer

    def conv(self, in_layer, out_shape, ksize=3, stride=1,
             padding='SAME', stddev=.01, bias=1, init='xavier', lrn=False,
             weight_decay=None):
        in_shape = in_layer.get_shape().as_list()[-1]
        # self.conv_counter += 1
        # self.global_counter += 1
        # name = 'conv' + str(self.conv_counter)
        # with tf.variable_scope(name) as scope:
        # import pdb; pdb.set_trace()
        kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                 shape=[ksize, ksize, in_shape, out_shape],
                                 dtype=tf.float32,
                                 name='weights')
        self.weight_decay(weight_decay, kernel)
        conv = tf.nn.conv2d(in_layer, kernel,
                            strides=[1, stride, stride, 1],
                            padding=padding)
        biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                 shape=[out_shape],
                                 dtype=tf.float32,
                                 name='bias')
        conv_bias = tf.nn.bias_add(conv, biases, name='conv')
        # self.parameters += [kernel, biases]
        self._output = conv_bias
        return conv_bias

    def lrn(self, in_layer, depth_radius=2, bias=1., alpha=2e-5, beta=.75):
        # name = 'conv' + str(self.conv_counter)
        # with tf.variable_scope(name) as scope:
        lrn = tf.nn.lrn(in_layer,
                        depth_radius=depth_radius,
                        bias=bias,
                        alpha=alpha,
                        beta=beta,
                        name='norm')
        self._output = lrn
        return lrn

    def memory(self, in_layer, state, memory_decay=0, trainable=False):
        initializer = tf.constant_initializer(value=memory_decay)
        mem = tf.get_variable(initializer=initializer,
                              shape=1,
                              trainable=trainable,
                              name='decay_param')
        decay_factor = tf.sigmoid(mem)
        new_state = tf.mul(state, decay_factor) + in_layer
        self._state = new_state
        return new_state

    def relu(self, in_layer):
        # name = 'conv' + str(self.conv_counter)
        # with tf.variable_scope(name) as scope:
        relu_out = tf.nn.relu(in_layer, name='relu')
        self._output = relu_out
        return relu_out

    def fc(self, in_layer, out_shape, dropout=None, stddev=.01,
           bias=1, init='xavier'):
        in_shape = in_layer.get_shape().as_list()[-1]
        # self.fc_counter += 1
        # self.global_counter += 1
        # name = 'fc' + str(self.fc_counter)
        # stdevs = [.01,.01,.01] #[.0005, .005, .1]
        # with tf.name_scope(name) as scope:
        kernel = tf.get_variable(initializer=self.initializer(init, stddev=stddev),
                                    shape=[in_shape, out_shape],
                                    dtype=tf.float32,
                                    name='weights')
        biases = tf.get_variable(initializer=tf.constant_initializer(bias),
                                    shape=[out_shape],
                                    name='bias')
        fcm = tf.matmul(in_layer, kernel)
        fc_out = tf.nn.bias_add(fcm, biases, name='fc')
        self._output = fc_out
        return fc_out

    def dropout(self, in_layer, dropout=None):
        # with tf.name_scope(name) as scope:
        drop = tf.nn.dropout(in_layer, dropout, name='dropout')
        self._output = drop
        return drop

    def pool(self, in_layer, ksize=3, stride=2, padding='SAME'):
        pool_out = tf.nn.max_pool(in_layer,
                                  ksize=[1, ksize, ksize, 1],
                                  strides=[1, stride, stride, 1],
                                  padding=padding,
                                  name='pool')
        self._output = pool_out
        return pool_out

    def weight_decay(self, weight_decay, weights):
        if weight_decay is not None:
            loss = tf.mul(tf.nn.l2_loss(weights), weight_decay, name='weight_loss')
            tf.add_to_collection('losses', loss)