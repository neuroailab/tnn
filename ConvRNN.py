
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
""" RNNCell Types included in this bundle (only 3 payments of 29.99!!):
 @@ConvRNNCell
 @@ConvPoolRNNCell
 (but wait, there's more!)
 @@FcRNNCell
 """

# graph parameters
WEIGHT_STDDEV = 0.1  # for truncated normal distribution
BIAS_INIT = 0.1  # initialization for bias variables
KERNEL_SIZE = 3
DECAY_PARAM_INITIAL = 0 #(actual decay factor is initialized to ~ sigmoid(p_j) = 0.5)




# other graph creation helpers
def _weights(shape):  # weights for convolution
    # shape = [spatial, spatial, num_input_channels, num_output_channels]
    # initialized with truncated normal distribution
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=WEIGHT_STDDEV)
    return tf.get_variable(name='weights', shape=shape, dtype=tf.float32, initializer=initializer)


def _bias(shape):  # bias variable for convolution
    initializer = tf.constant_initializer(value=BIAS_INIT)
    return tf.get_variable(name='bias', shape=shape, dtype=tf.float32, initializer=initializer)


def _activation_summary(x):  # helper function to create tf summaries for conv, fc, etc.
    # x = Tensor [aka output after convs, FCs, Softmax], returns nothing
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))  # cool! measure sparsity too!

def _conv(input, weights, bias, name):
    # 1x1 stride conv + bias. No ReLU
    conv2d = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv2d, bias)
    return conv

def _relu(input):
    return tf.nn.relu(input, name='relu')

def _decay_param():
    initializer = tf.constant_initializer(value=DECAY_PARAM_INITIAL)
    decay_param = tf.get_variable(name='decay_param', shape=1, initializer=initializer)
    return decay_param


def to_decay_factor(decay_parameter):
    # decay_parameter is tf.Tensor
    decay_factor = tf.sigmoid(decay_parameter)
    return decay_factor



def maxpool(input, in_spatial, out_spatial, name='pool'):
    stride = in_spatial / out_spatial  # how much to pool by
    pool = tf.nn.max_pool(input, ksize=[1, stride, stride, 1],  # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool


class ConvPoolRNNCell(RNNCell):
    def __init__(self, state_size, output_size, kernel_size=3):
        """ state_size = same size as conv of input"""
        # output_size,state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state. size of conv(input)
        self._output_size = output_size

    def zero_state(self, batch_size, dtype):  # unfortunately we need to have the same method signature
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # output_shape = [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS] determine pooling stride and # conv filters
            # get shape as a list [#batches, width, height, depth/channels]
            input_shape = inputs.get_shape().as_list()
            output_shape = self._output_size
            # weights_shape = [spatial, spatial, num_input_channels, num_output_channels]
            weights_shape = [KERNEL_SIZE, KERNEL_SIZE, input_shape[3], output_shape[3]]
            # conv
            weights = _weights(weights_shape)
            bias = _bias([output_shape[3]])
            conv = _conv(inputs, weights, bias, name='conv')  # conv of inputs
            decay_factor = to_decay_factor(_decay_param())
            new_state = tf.mul(state, decay_factor) + conv  # new state = decay(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
            _activation_summary(relu)
            # pool
            pool = maxpool(input=relu, in_spatial=input_shape[1],
                           out_spatial=output_shape[1], name='pool')
        return pool, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size  # [spatial, num_channels]


class ConvRNNCell(RNNCell):
    """ Since input preprocessing (concatenation and adding and all that jazz) happens
        outside of the cell, state_size = size(Conv(input)). """

    def __init__(self, state_size, kernel_size=3):
        """ Since input preprocessing (concatenation and adding and all that jazz) happens
        outside of the cell, conv(input_size) = state_size = output_size. """
        # output_size, state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state. = shape of OUTPUT of conv
        self._output_size = state_size  # the spatial and the number of channels

    def zero_state(self, batch_size, dtype):  # unfortunately we need to have the same method signature
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # output_shape = [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS] determine pooling stride and # conv filters
            # get shape as a list [#batches, width, height, depth/channels]
            input_shape = inputs.get_shape().as_list()
            output_shape = self._output_size
            # weights_shape = [spatial, spatial, num_input_channels, num_output_channels]
            weights_shape = [KERNEL_SIZE, KERNEL_SIZE, input_shape[3], output_shape[3]]
            # conv
            weights = _weights(weights_shape)
            bias = _bias([output_shape[3]])
            conv = _conv(inputs, weights, bias, name='conv')  # conv of inputs
            decay_factor = to_decay_factor(_decay_param())
            new_state = tf.mul(state, decay_factor) + conv  # new state = decay(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
            _activation_summary(relu)
        return relu, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size  # [spatial, num_channels]

#### FcRNN fun #########
# state = decay(old_state) + W(input) + b
# out = ReLU(state)
def _flatten(input):
    # flatten to: [batch_size, input_size]
    input_shape = input.get_shape().as_list()
    input_size = input_shape[1] * input_shape[2] * input_shape[3]
    input_flat = tf.reshape(input, [-1, input_size])
    return input_flat

def linear(input, output_size, scope=None):
    # input is flattened already. Check this:
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        raise ValueError('your input shape, ', input_shape, 'is not 2-D. FLATTEN IT!!')
    # output_size = integer
    input_size = input.get_shape().as_list()[1]
    weights = _weights([input_size, output_size])
    bias = _bias([output_size])
    return tf.add(tf.matmul(input, weights), bias)

class FcRNNCell(RNNCell):
    """ output = ReLU(state) where state = decay(old_state) + (W*input + b)"""

    def __init__(self, state_size):
        """state_size = same size as output. Used for zero_state creation"""
        # state_size, output_size = [batch_size, fc_output_size]
        self._state_size = state_size
        self._output_size = state_size

    def zero_state(self, batch_size, dtype):  # unfortunately we need to have the same method signature
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # flatten input if shape is not 2d (batchsize|size)
            if len(inputs.get_shape().as_list()) > 2: # needs flattening. assumed to be 4d
                inputs = _flatten(inputs)
            # W*input + b
            linear_ = linear(input=inputs, output_size=self._output_size[1])

            decay_factor = to_decay_factor(_decay_param())
            new_state = tf.mul(state, decay_factor) + linear_  # new state = decay(old state) + linear(inputs)
            relu = _relu(new_state)  # activation function
            _activation_summary(relu)
        return relu, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size  # 1d, size of output (excluding batch size).
