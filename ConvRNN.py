from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import array_ops
""" RNNCell Types included:
 @@ConvRNNCell
 @@ConvPoolRNNCell
 @@FcRNNCell
 """
# Default graph parameters. Can specify different values when creating RNNCells.
DEFAULT_INITIALIZER = 'xavier' # or 'trunc_norm'
WEIGHT_STDDEV = 0.1  # for truncated normal distribution (if selected)
BIAS_INIT = 0.1  # initialization for bias variables
CONV_SIZE = 3
CONV_STRIDE = 1
POOL_SIZE = None # defaults to match the pool_stride, which is determined by the input and output sizes to the maxpool function.
# POOL_STRIDE --> PRE-DETERMINED by input and output sizes.
DECAY_PARAM_INITIAL = 0 #(actual decay factor is initialized to ~ sigmoid(p_j) = 0.5)


# other graph creation helpers
def _weights(shape, init='xavier', stddev=WEIGHT_STDDEV):  # weights for convolution
    """ shape = [spatial, spatial, num_input_channels, num_output_channels]
    init: 'xavier' initialization or 'trunc_norm' truncated normal distribution
    stddev: stddev of truncated normal distribution, if used.
    """
    if init == 'xavier':
        initializer = tf.contrib.layers.initializers.xavier_initializer()
    elif init == 'trunc_norm':
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=stddev)
    else:
        raise ValueError('Please provide an appropriate initialization method: xavier or trunc_norm')
    return tf.get_variable(name='weights', shape=shape, dtype=tf.float32, initializer=initializer)


def _bias(shape, bias_init=BIAS_INIT):  # bias variable for convolution
    initializer = tf.constant_initializer(value=bias_init)
    return tf.get_variable(name='bias', shape=shape, dtype=tf.float32, initializer=initializer)

def _conv(input, weights, bias, conv_stride=CONV_STRIDE, name='conv'):
    # conv with stride as specified (kernel size determined by weights) + bias. (No ReLU)
    # If conv_stride > 1 we use 'VALID' padding instead of 'SAME'
    if conv_stride > 1:
        padding = 'VALID'
    else:
        padding = 'SAME'
    conv2d = tf.nn.conv2d(input, weights, strides=[1, conv_stride, conv_stride, 1], padding=padding)
    conv = tf.nn.bias_add(conv2d, bias, name=name)
    return conv

def _relu(input):
    return tf.nn.relu(input, name='relu')

def _decay_param(initial=DECAY_PARAM_INITIAL, trainable=True):
    # Note! This is the parameter for the actual decay FACTOR in [0,1] (take the sigmoid of this value)
    # trainable=True for trainable decay param
    initializer = tf.constant_initializer(value=initial)
    decay_param = tf.get_variable(name='decay_param', shape=1, initializer=initializer, trainable=trainable)
    return decay_param

def to_decay_factor(decay_parameter):
    # decay_parameter is tf.Tensor
    decay_factor = tf.sigmoid(decay_parameter)
    return decay_factor

def maxpool(input, in_spatial, out_spatial, kernel_size=None, name='pool'):
    """ We determine pooling stride by the in and out spatial. This is an important feature
    for the pooling between bypass layers.
    kernel_size=None will set kernel_size same as stride.
    """
    stride = in_spatial / out_spatial  # how much to pool by
    if kernel_size == None:
        kernel_size = stride
    pool = tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1],  # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool


class ConvPoolRNNCell(RNNCell):
    def __init__(self, state_size, output_size,
                 conv_size=CONV_SIZE, conv_stride=CONV_STRIDE,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 bias_init=BIAS_INIT,
                 pool_size=POOL_SIZE,
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """ state_size = same size as conv of input
        memory=True to use decay factor in state transition. If False, state is not remembered"""
        # output_size,state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state. size of conv(input)
        self._output_size = output_size
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init
        self.pool_size = pool_size
        self.decay_param_init = decay_param_init
        self.memory = memory

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
            weights_shape = [self.conv_size, self.conv_size, input_shape[3], output_shape[3]]
            # conv
            weights = _weights(weights_shape, init=self.weight_init, stddev=self.weight_stddev)
            bias = _bias([output_shape[3]], bias_init=self.bias_init)
            conv = _conv(inputs, weights, bias, conv_stride=self.conv_stride)  # conv of inputs
            if self.memory:
                decay_factor = to_decay_factor(_decay_param(initial=self.decay_param_init))
            else: # no 'self-loop' with state
                decay_factor = 0
            new_state = tf.mul(state, decay_factor) + conv  # new state = decay(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
            # pool
            pool = maxpool(input=relu, in_spatial=input_shape[1],
                           out_spatial=output_shape[1], kernel_size=self.pool_size)
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

    def __init__(self, state_size,
                 conv_size=CONV_SIZE, conv_stride=CONV_STRIDE,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 bias_init=BIAS_INIT,
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """ state_size = same size as conv of input
        memory=True to use decay factor in state transition. If False, state is not remembered"""
        # output_size,state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state. = shape of OUTPUT of conv
        self._output_size = state_size  # the spatial and the number of channels
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init
        self.decay_param_init = decay_param_init
        self.memory = memory


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
            weights_shape = [self.conv_size, self.conv_size, input_shape[3], output_shape[3]]
            # conv
            weights = _weights(weights_shape, init=self.weight_init, stddev=self.weight_stddev)
            bias = _bias([output_shape[3]], bias_init=self.bias_init)
            conv = _conv(inputs, weights, bias, conv_stride=self.conv_stride)  # conv of inputs
            if self.memory:
                decay_factor = to_decay_factor(_decay_param(initial=self.decay_param_init))
            else:  # no 'self-loop' with state
                decay_factor = 0
            new_state = tf.mul(state, decay_factor) + conv  # new state = decay(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
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

def linear(input, output_size,
           weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
           bias_init=BIAS_INIT, scope=None):
    # input is flattened already. Check this:
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        raise ValueError('your input shape, ', input_shape, 'is not 2-D. FLATTEN IT!!')
    # output_size = integer
    input_size = input.get_shape().as_list()[1]
    weights = _weights([input_size, output_size], init=weight_init, stddev=weight_stddev)
    bias = _bias([output_size], bias_init=bias_init)
    return tf.add(tf.matmul(input, weights), bias)

class FcRNNCell(RNNCell):
    """ output = ReLU(state) where state = decay(old_state) + (W*input + b)"""

    def __init__(self, state_size,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 bias_init=BIAS_INIT,
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """state_size = same size as output. Used for zero_state creation"""
        # state_size, output_size = [batch_size, fc_output_size]
        self._state_size = state_size  # used for zero_state. size of conv(input)
        self._output_size = state_size
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init
        self.decay_param_init = decay_param_init
        self.memory = memory

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
            linear_ = linear(input=inputs, output_size=self._output_size[1],
                   weight_init=self.weight_init, weight_stddev=self.weight_stddev,
                   bias_init=self.bias_init, scope=None)
            if self.memory:
                decay_factor = to_decay_factor(_decay_param(initial=self.decay_param_init))
            else:  # no 'self-loop' with state
                decay_factor = 0
            new_state = tf.mul(state, decay_factor) + linear_  # new state = decay(old state) + linear(inputs)
            relu = _relu(new_state)  # activation function

        return relu, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size  # 1d, size of output (excluding batch size).
