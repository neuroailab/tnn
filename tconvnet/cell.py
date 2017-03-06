"""
General Functional Cell
"""

from __future__ import absolute_import, division, print_function
import sys, copy

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

sys.path.insert(0, '../../tfutils/base_class')
import tfutils.model


def harbor(inputs, shape):
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
            if len(inp.shape) == 2:
                raise ValueError('layers of dim 2 cannot project to layers of dim 4')
            elif len(inp.shape) == 4:
                out = tf.image.resize_images(inp, shape[1:3])#tf.constant(shape))
                outputs.append(out)
                # h = inp.shape.as_list()[1] // shape[1]
                # w = inp.shape.as_list()[2] // shape[2]
                # if [h, w] == list(shape):
                #     outputs.append(inp)
                # else:
                #     out = tf.nn.max_pool(inp,
                #                          ksize=[1, 3 * h // 2, 3 * w // 2, 1],
                #                          strides=[1, h, w, 1],
                #                          padding='SAME')
                #     outputs.append(out)
            else:
                raise ValueError

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
                 input_shapes,
                 input_dtypes,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 state_init=(tf.zeros, None),
                 output_init=(tf.zeros, None),
                 name=None
                 ):

        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})
        self.output_init = output_init if output_init[1] is not None else (output_init[0], {})

        self.name = name

        self.state = None
        self.output = None

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
        if inputs is None:
            inputs = [None] * len(self.input_shapes)

        with tf.variable_scope(self.name):
            inputs_full = []
            for inp, shape, dtype in zip(inputs, self.input_shapes, self.input_dtypes):
                if inp is None:
                    inp = self.output_init[0](shape=shape, dtype=dtype, **self.output_init[1])
                inputs_full.append(inp)

            output = self.harbor[0](inputs_full, self.harbor_shape, **self.harbor[1])

            for function, kwargs in self.pre_memory:
                output = function(output, **kwargs)

            if state is None:
                state = self.state_init[0](shape=output.shape,
                                           dtype=output.dtype,
                                           **self.state_init[1])

            state = self.memory[0](output, state, **self.memory[1])
            self.state = tf.identity(state, name='state')

            output = self.state
            for function, kwargs in self.post_memory:
                output = function(output, **kwargs)

            self.output = tf.identity(output, name='output')  # for naming consistency
        return self.output, self.state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        if self.state is not None:
            return self.state.shape
        else:
            raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        if self.output is not None:
            return self.output.shape
        else:
            raise ValueError('Output not initialized yet')