"""
General Functional Cell
"""

from __future__ import absolute_import, division, print_function

import re
import math

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

import tfutils.model


def harbor(inputs, shape, reuse=None):
    """
    Default harbor function that resizes inputs to desired shape and concats them.

    :Args:
        - inputs
        - shape
    """
    outputs = []
    for inp in inputs:
        if len(shape) == 2:
            if len(inp.shape) == 2:
                outputs.append(inp)
            elif len(inp.shape) == 4:
                out = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                outputs.append(out)
            else:
                raise ValueError

        elif len(shape) == 4:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                xs, ys = shape[1: 3]
                s = inp.shape.as_list()[1]
                nchnls = int(math.ceil(s / float(xs * ys)))
                if s % (xs * ys) != 0:
                    out_depth = xs * ys * nchnls
                    nm = pat.sub('__', inp.name.split('/')[1].split('_')[0])
                    nm = 'harbor_imsizefc_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, out_depth)
                out = tf.reshape(inp, (inp.shape.as_list()[0], xs, ys, nchnls))
            elif len(inp.shape) == 4:
                out = tf.image.resize_images(inp, shape[1:3])
            else:
                raise ValueError
            outputs.append(out)

        else:
            raise ValueError('harbor cannot process layer of dim {}'.format(len(shape)))

    output = tf.concat(outputs, axis=-1, name='harbor')

    return output

def custom_harbor(inputs, shape, op1='resize', op2='concat', kernel_init='xavier', reuse=None):
    """
    Custom harbor function that resizes inputs to desired shape and concats them, or can tile, add or multiply
    inputs based on user specification.

    :Args:
        - inputs
        - shape
    """
    outputs = []
    for inp in inputs:
        if len(shape) == 2:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                if op2 != 'concat' and inp.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_fc_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, shape[1], kernel_init=kernel_init)

                outputs.append(inp)

            elif len(inp.shape) == 4:
                out = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                if op2 != 'concat' and out.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_conv_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = tfutils.model.fc(out, shape[1], kernel_init=kernel_init)    

                outputs.append(out)
            else:
                raise ValueError

        elif len(shape) == 4:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                nchannels = shape[3]
                if nchannels != inp.shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_conv_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, nchannels, kernel_init=kernel_init)
                 
                xs, ys = shape[1: 3]
                inp = tf.tile(inp, [1, xs*ys])
                out = tf.reshape(inp, (inp.shape.as_list()[0], xs, ys, nchannels))

            elif len(inp.shape) == 4:
                if op1 == 'tile':
                    inp_height = inp.get_shape().as_list()[1]
                    inp_width = inp.get_shape().as_list()[2]
                    height_multiple = 1 + (shape[1] // inp_height)
                    width_multiple = 1 + (shape[2] // inp_width)
                    tiled_out = tf.tile(inp, [1, height_multiple, width_multiple, 1])
                    out = tf.map_fn(lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]), tiled_out, dtype=tf.float32) 
                else:
                    out = tf.image.resize_images(inp, shape[1:3])

                if op2 != 'concat' and out.shape[3] != shape[3]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'conv_to_conv_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = tfutils.model.conv(out, out_depth=shape[3], ksize=[1, 1], kernel_init=kernel_init)
            else:
                raise ValueError
            outputs.append(out)

        else:
            raise ValueError('harbor cannot process layer of dim {}'.format(len(shape)))

    if op2 == 'add':
        output = tf.add_n(outputs, name='harbor')
    elif op2 == 'multiply':
        if len(outputs) == 1:
            output = outputs[0]
        else:
            output = tf.multiply(outputs[0], outputs[1])
            if len(outputs) > 2:
                for output_elem in outputs[2:]:
                    output = tf.multiply(output, output_elem)
    else:
        output = tf.concat(outputs, axis=-1, name='harbor')

    return output

def memory(inp, state, memory_decay=0, trainable=False, name='memory'):
    """
    Memory that decays over time
    """
    initializer = tfutils.model.initializer(kind='constant', value=memory_decay)
    mem = tf.get_variable(initializer=initializer,
                          shape=1,
                          dtype=tf.float32,
                          trainable=trainable,
                          name='memory_decay')
    state = tf.add(state * mem, inp, name=name)
    return state


class GenFuncCell(RNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None
                 ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = input_init if input_init[1] is not None else (input_init[0], {})
        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})

        self.dtype = dtype
        self.name = name

        self._reuse = None

    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs

        If inputs or state are None, they are initialized from scratch.

        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state

        :Returns:
            (output, state)
        """
        # if hasattr(self, 'output') and inputs is None:
        #     raise ValueError('must provide inputs')

        # if inputs is None:
        #     inputs = [None] * len(self.input_shapes)
        # import pdb; pdb.set_trace()

        with tf.variable_scope(self.name, reuse=self._reuse):
            # inputs_full = []
            # for inp, shape, dtype in zip(inputs, self.input_shapes, self.input_dtypes):
            #     if inp is None:
            #         inp = self.output_init[0](shape=shape, dtype=dtype, **self.output_init[1])
            #     inputs_full.append(inp)

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, reuse=self._reuse, **self.harbor[1])
       
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                pre_name_counter += 1
            if state is None:
                state = self.state_init[0](shape=output.shape,
                                           dtype=self.dtype,
                                           **self.state_init[1])
            state = self.memory[0](output, state, **self.memory[1])
            self.state = tf.identity(state, name='state')

            output = self.state
            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                post_name_counter += 1
            self.output = tf.identity(tf.cast(output, self.dtype), name='output')
            # scope.reuse_variables()
            self._reuse = True
        self.state_shape = self.state.shape
        self.output_shape = self.output.shape
        return self.output, self.state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output is not None:
        return self.output_shape
        # else:
        #     raise ValueError('Output not initialized yet')
