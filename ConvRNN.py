"""
This module contains the basic building blocks of the model.
It implements all the standard ConvNet layers, but with RNN self loops.
They inherit from the RNNCell base class.

RNNCell Types included:
 @@ConvRNNCell
 @@ConvPoolRNNCell
 @@MultiConvPoolRNNCell
 @@FcRNNCell
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell

# Default graph parameters. Can specify different values when creating RNNCells
DEFAULT_INITIALIZER = 'xavier'  # or 'trunc_norm'
WEIGHT_STDDEV = 0.1  # for truncated normal distribution (if selected)
BIAS_INIT = 0.1  # initialization for bias variables
CONV_SIZE = 3
CONV_STRIDE = 1
POOL_SIZE = None  # defaults to match the pool_stride, which is determined by
# the input and output sizes to the maxpool function.
# POOL_STRIDE --> PRE-DETERMINED by input and output sizes.
DECAY_PARAM_INITIAL = 0  # (actual decay factor is sigmoid(p_j) = 0.5)

# Local response norm
LRN_RADIUS = 2
LRN_ALPHA = 2e-05
LRN_BETA = 0.75
LRN_BIAS = 1.0


def _weights(shape, init=DEFAULT_INITIALIZER, stddev=WEIGHT_STDDEV):
    """
    return TF variable that generates weights for convolution

    shape = [spatial, spatial, num_input_channels, num_output_channels]
    init: 'xavier' initialization or 'trunc_norm' truncated normal distribution
    stddev: stddev of truncated normal distribution, if used.
    """
    if init == 'xavier':
        initializer = tf.contrib.layers.initializers.xavier_initializer()
    elif init == 'trunc_norm':
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=stddev)
    else:
        raise ValueError('Please provide an appropriate initialization '
                         'method: xavier or trunc_norm')
    return tf.get_variable(name='weights', shape=shape, dtype=tf.float32,
                           initializer=initializer)


def _bias(shape, bias_init=BIAS_INIT):  # bias variable for convolution
    initializer = tf.constant_initializer(value=bias_init)
    return tf.get_variable(name='bias', shape=shape, dtype=tf.float32,
                           initializer=initializer)


def _conv(input, weights, bias, conv_stride=CONV_STRIDE, name='conv'):
    # conv with stride as specified (kernel size determined by weights) + bias.
    conv2d = tf.nn.conv2d(input, weights,
                          strides=[1, conv_stride, conv_stride, 1],
                          padding='SAME')
    conv = tf.nn.bias_add(conv2d, bias, name=name)
    return conv


def _relu(input):
    return tf.nn.relu(input, name='relu')


def _decay_param(initial=DECAY_PARAM_INITIAL, trainable=True):
    """
    Returns the parameter for the actual decay FACTOR in [0,1]
    (take the sigmoid of this value)
    trainable=True for trainable decay param
    """
    initializer = tf.constant_initializer(value=initial)
    decay_param = tf.get_variable(name='decay_param', shape=1,
                                  initializer=initializer, trainable=trainable)
    return decay_param


def to_decay_factor(decay_parameter):
    # decay_parameter is tf.Tensor
    decay_factor = tf.sigmoid(decay_parameter)
    return decay_factor


def maxpool(input_, out_spatial, kernel_size=None, name='pool'):
    """ Returns a tf operation for maxpool of input, with stride determined
    by the spatial size ratio of output and input
    kernel_size = None will set kernel_size same as stride.
    """
    in_spatial = input_.get_shape().as_list()[1]
    stride = in_spatial / out_spatial  # how much to pool by
    if stride < 1:
        raise ValueError('spatial dimension of output should not be greater '
                         'than that of input')
    if kernel_size is None:
        kernel_size = stride
    pool = tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1],
                          # kernel (filter) size
                          strides=[1, stride, stride, 1], padding='SAME',
                          name=name)
    return pool


class ConvPoolRNNCell(RNNCell):
    def __init__(self, state_size, output_size,
                 conv_size=CONV_SIZE, conv_stride=CONV_STRIDE,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 weight_decay=None,
                 bias_init=BIAS_INIT,
                 lrn=False,
                 pool_size=POOL_SIZE,
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """ state_size = same size as conv of input
        memory=True to use decay factor in state transition.
        If False, state is not remembered"""
        # output_size,state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state. conv(input) size
        self._output_size = output_size
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.weight_decay = weight_decay
        self.bias_init = bias_init
        self.lrn = lrn  # using local response normalization
        self.pool_size = pool_size
        self.decay_param_init = decay_param_init
        self.memory = memory

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # output_shape = [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
            # determines pooling stride and # conv filters
            # Get shape as a list [#batches, width, height, depth/channels]
            input_shape = inputs.get_shape().as_list()
            output_shape = self._output_size
            # weights_shape = [spatial, spatial,
            # num_input_channels, num_output_channels]
            weights_shape = [self.conv_size, self.conv_size, input_shape[3],
                             output_shape[3]]
            # conv
            weights = _weights(weights_shape, init=self.weight_init,
                               stddev=self.weight_stddev)
            if self.weight_decay is not None:
                weight_decay = tf.mul(tf.nn.l2_loss(weights),
                                      self.weight_decay, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            bias = _bias([output_shape[3]], bias_init=self.bias_init)
            conv = _conv(inputs, weights, bias,
                         conv_stride=self.conv_stride)  # conv of inputs

            if self.lrn:
                conv = tf.nn.local_response_normalization(conv,
                                                      depth_radius=LRN_RADIUS,
                                                      alpha=LRN_ALPHA,
                                                      beta=LRN_BETA,
                                                      bias=LRN_BIAS)
            if self.memory:
                decay_factor = to_decay_factor(
                    _decay_param(initial=self.decay_param_init))
                # new state = decay(old state) + conv(inputs)
                new_state = tf.mul(state, decay_factor) + conv
            else:  # no 'self-loop' with state
                new_state = conv  # new state = 0*(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
            # pool. Stride depends on input spatial (of conv's output)
            # and desired out_spatial
            pool = maxpool(input_=relu,
                           out_spatial=output_shape[1],
                           kernel_size=self.pool_size)
        return pool, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size


class ConvRNNCell(RNNCell):
    """ Since input preprocessing (concatenation and adding) happens
        outside of the cell, state_size = size(Conv(input)). """

    def __init__(self, state_size,
                 conv_size=CONV_SIZE, conv_stride=CONV_STRIDE,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 weight_decay=None,
                 bias_init=BIAS_INIT,
                 lrn=False,
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """ state_size = same size as conv of input
        memory=True to use decay factor in state transition.
        If False, state is not remembered"""
        # output_size,state_size - [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
        self._state_size = state_size  # used for zero_state= conv shape
        self._output_size = state_size
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.weight_decay = weight_decay
        self.bias_init = bias_init
        self.lrn = lrn
        self.decay_param_init = decay_param_init
        self.memory = memory

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # output_shape = [BATCHSIZE, SPATIAL, SPATIAL, NUMCHANNELS]
            #  determine pooling stride and # conv filters
            # Get shape as a list [#batches, width, height, depth/channels]
            input_shape = inputs.get_shape().as_list()
            output_shape = self._output_size
            # weights_shape = [spatial, spatial,
            # num_input_channels, num_output_channels]
            weights_shape = [self.conv_size, self.conv_size, input_shape[3],
                             output_shape[3]]
            # conv
            weights = _weights(weights_shape, init=self.weight_init,
                               stddev=self.weight_stddev)
            if self.weight_decay is not None:
                weight_decay = tf.mul(tf.nn.l2_loss(weights),
                                      self.weight_decay, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)
            bias = _bias([output_shape[3]], bias_init=self.bias_init)
            conv = _conv(inputs, weights, bias,
                         conv_stride=self.conv_stride)  # conv of inputs

            if self.lrn:  # local response norm
                conv = tf.nn.local_response_normalization(conv,
                                                      depth_radius=LRN_RADIUS,
                                                      alpha=LRN_ALPHA,
                                                      beta=LRN_BETA,
                                                      bias=LRN_BIAS)

            if self.memory:
                decay_factor = to_decay_factor(
                    _decay_param(initial=self.decay_param_init))
                # new state = decay(old state) + conv(inputs)
                new_state = tf.mul(state, decay_factor) + conv
            else:  # no 'self-loop' with state
                new_state = conv  # new state = 0*(old state) + conv(inputs)
            relu = _relu(new_state)  # activation function
        return relu, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size


def _flatten(input):
    # flatten to: [batch_size, input_size]
    input_shape = input.get_shape().as_list()
    input_size = input_shape[1] * input_shape[2] * input_shape[3]
    input_flat = tf.reshape(input, [-1, input_size])
    return input_flat


def linear(input, output_size,
           weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
           bias_init=BIAS_INIT):
    # input is flattened already. Check this:
    input_shape = input.get_shape().as_list()
    if len(input_shape) > 2:
        raise ValueError('your input shape, ', input_shape,
                         'is not 2-D. FLATTEN IT!!')
    # output_size = integer
    input_size = input.get_shape().as_list()[1]
    weights = _weights([input_size, output_size], init=weight_init,
                       stddev=weight_stddev)
    bias = _bias([output_size], bias_init=bias_init)
    return tf.add(tf.matmul(input, weights), bias)


class FcRNNCell(RNNCell):
    """ output = ReLU(state) where state = decay(old_state) + (W*input + b)
    # out = ReLU(state)
    """

    def __init__(self, state_size,
                 weight_init=DEFAULT_INITIALIZER, weight_stddev=WEIGHT_STDDEV,
                 bias_init=BIAS_INIT,
                 keep_prob=None,  # for dropout
                 decay_param_init=DECAY_PARAM_INITIAL,
                 memory=True):
        """state_size = same size as output. Used for zero_state creation"""
        # state_size, output_size = [batch_size, fc_output_size]
        self._state_size = state_size  # used for zero_state. conv(input) size
        self._output_size = state_size
        self.weight_init = weight_init
        self.weight_stddev = weight_stddev
        self.keep_prob = keep_prob  # if None, no dropout used
        self.bias_init = bias_init
        self.decay_param_init = decay_param_init
        self.memory = memory

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor"""
        zeros = tf.zeros(self._state_size, dtype=tf.float32, name='zero_state')
        return zeros

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # flatten input if shape is not 2d (batchsize|size)
            if len(inputs.get_shape().as_list()) > 2:
                # needs flattening. assumed to be 4d
                inputs = _flatten(inputs)
            # W*input + b
            linear_ = linear(input=inputs, output_size=self._output_size[1],
                             weight_init=self.weight_init,
                             weight_stddev=self.weight_stddev,
                             bias_init=self.bias_init)
            if self.memory:
                decay_factor = to_decay_factor(
                    _decay_param(initial=self.decay_param_init))
                # new state = decay(old state) + linear(inputs)
                new_state = tf.mul(state, decay_factor) + linear_
            else:  # no 'self-loop' with state
                new_state = linear_  # new st = 0*(old state) + linear(inputs)
            relu = _relu(new_state)  # activation function
            # dropout
            if self.keep_prob is None:
                print('no dropout used')
            else:
                relu = tf.nn.dropout(relu, self.keep_prob)  # apply dropout
                print('using dropout! keep_prob:', self.keep_prob)
        return relu, new_state

    @property
    def state_size(self):  # actual state size.
        return self._state_size

    @property
    def output_size(self):
        return self._output_size  # 1d, size of output (excluding batch size).
