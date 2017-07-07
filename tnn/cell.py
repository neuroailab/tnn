"""
General Functional Cell
"""

from __future__ import absolute_import, division, print_function

import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

import tfutils.model

def gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms):
    '''Helper function that returns the skip, feedforward, and feedback inputs'''
    assert(ff_inpnm is not None)
    assert(node_nms is not None)

    if l1_inpnm not in node_nms:
        node_nms = [l1_inpnm] + node_nms # easy to forget to add this
    
    # determine the skip and possible feedback inputs to this layer
    ff_idx = 0
    for idx, elem in enumerate(node_nms):
        if elem == ff_inpnm:
            ff_idx = idx
            break

    skips = node_nms[:ff_idx] # exclude ff input
    feedbacks = node_nms[ff_idx+2:] # exclude ff input, note no layer has ff_idx to be the last element

    skip_ins = []
    feedback_ins = []
    ff_in = None
    for inp in inputs:
        pat = re.compile(':|/')
        if l1_inpnm not in inp.name:
            nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
        else:
            nm = l1_inpnm
      
        if ff_inpnm == nm:
            ff_in = inp
        elif nm in feedbacks: # a feedback input
            if len(inp.shape) == 4: # flatten conv inputs to pass through mlp later
                reshaped_inp = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                feedback_ins.append(reshaped_inp)
            elif len(inp.shape) == 2:
                feedback_ins.append(inp)
            else:
                raise ValueError
        elif nm in skips:
            skip_ins.append(inp)
    
    return ff_in, skip_ins, feedback_ins

def crop_func(inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse):
    # note: e.g. node_nms = ['split', 'V1', 'V2', 'V4', 'pIT', 'aIT']

    ff_in, skip_ins, feedback_ins = gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms)
    if len(feedback_ins) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4: # we do nothing in this case, and proceed as usual (appeases initialization too)
        return inputs
    feedback_ins = tf.concat(feedback_ins, axis=-1, name='comb')
    mlp_nm = 'crop_mlp_for_%s' % ff_inpnm
    with tf.variable_scope(mlp_nm, reuse=reuse):
        mlp_out = tfutils.model.fc(feedback_ins, 5, kernel_init=kernel_init, activation=None) # best way to initialize this?

    alpha = tf.slice(mlp_out, [0, 0], [-1, 1])
    alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
    alpha = tf.nn.tanh(alpha) # we want to potentially have negatives to downweight
    boxes = tf.slice(mlp_out, [0, 1], [-1, 4])
    boxes = tf.nn.sigmoid(boxes) # keep values in [0, 1] range
    offset_height = tf.slice(boxes, [0, 0], [-1, 1])
    offset_height = tf.squeeze(tf.to_int32(tf.multiply(offset_height, ff_in.get_shape().as_list()[1])), axis=-1)
    offset_width = tf.slice(boxes, [0, 1], [-1, 1])
    offset_width = tf.squeeze(tf.to_int32(tf.multiply(offset_width, ff_in.get_shape().as_list()[2])), axis=-1)
    target_height = tf.slice(boxes, [0, 2], [-1, 1])
    target_height = tf.squeeze(tf.to_int32(tf.multiply(target_height, ff_in.get_shape().as_list()[1])), axis=-1)
    target_width = tf.slice(boxes, [0, 3], [-1, 1])
    target_width = tf.squeeze(tf.to_int32(tf.multiply(target_width, ff_in.get_shape().as_list()[2])), axis=-1)
    # clip height and width of bounding box
    target_height = tf.to_int32(tf.minimum(offset_height + target_height, ff_in.get_shape().as_list()[1]) - offset_height)
    target_width = tf.to_int32(tf.minimum(offset_width + target_width, ff_in.get_shape().as_list()[2]) - offset_width)
    # construct mask
    elems = (offset_height, offset_width, target_height, target_width)
    mask = tf.map_fn(lambda x: tf.pad(tf.ones([x[2], x[3], ff_in.get_shape().as_list()[3]]), \
        [[x[0], ff_in.get_shape().as_list()[1] - (x[0]+x[2])], [x[1], ff_in.get_shape().as_list()[2] - (x[1]+x[3])], [0, 0]], \
        "CONSTANT"), elems, dtype=tf.float32)

    padded_img = tf.multiply(ff_in, mask)
    padded_img = tf.multiply(alpha, padded_img)
    pat = re.compile(':|/')
    ff_nm = pat.sub('__', ff_in.name.split('/')[-2].split('_')[0])
    new_name = ff_nm + '_mod'
    new_in = tf.add(ff_in, padded_img, name=new_name)

    new_out = [new_in] + skip_ins # skips will be combined after
    return new_out

def tile_func(inp, shape):
    inp_height = inp.get_shape().as_list()[1]
    inp_width = inp.get_shape().as_list()[2]
    height_multiple = 1 + (shape[1] // inp_height)
    width_multiple = 1 + (shape[2] // inp_width)
    tiled_out = tf.tile(inp, [1, height_multiple, width_multiple, 1])
    return tf.map_fn(lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]), tiled_out, dtype=tf.float32) 

def harbor(inputs, shape, name, ff_inpnm=None, node_nms=None, l1_inpnm='split', preproc=None, spatial_op='resize', channel_op='concat', kernel_init='xavier', reuse=None):
    """
    Default harbor function which can crop the input (as a preproc), followed by a spatial_op which by default resizes inputs to a desired shape (or pad or tile), and finished with a channel_op which by default concatenates along the channel dimension (or add or multiply based on user specification).

    :Args:
        - inputs
        - shape
    """
    outputs = []
    if preproc == 'crop':
        inputs = crop_func(inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse)

    for inp in inputs:
        if len(shape) == 2:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                if channel_op != 'concat' and inp.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_fc_harbor_from_%s_to_%s' % (nm, name)
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, shape[1], kernel_init=kernel_init)

                outputs.append(inp)

            elif len(inp.shape) == 4:
                out = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                if channel_op != 'concat' and out.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'conv_to_fc_harbor_from_%s_to_%s' % (nm, name)
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
                    nm = 'fc_to_conv_harbor_from_%s_to_%s' % (nm, name)
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, nchannels, kernel_init=kernel_init)
                 
                xs, ys = shape[1: 3]
                inp = tf.tile(inp, [1, xs*ys])
                out = tf.reshape(inp, (inp.shape.as_list()[0], xs, ys, nchannels))

            elif len(inp.shape) == 4:
                if spatial_op == 'tile':
                    out = tile_func(inp, shape)
                elif spatial_op == 'pad':
                    out = tf.map_fn(lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]), inp, dtype=tf.float32)
                else:
                    out = tf.image.resize_images(inp, shape[1:3])

                if channel_op != 'concat' and out.shape[3] != shape[3]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'conv_to_conv_harbor_from_%s_to_%s' % (nm, name)
                    with tf.variable_scope(nm, reuse=reuse):
                        out = tfutils.model.conv(out, out_depth=shape[3], ksize=[1, 1], kernel_init=kernel_init)
            else:
                raise ValueError
            outputs.append(out)

        else:
            raise ValueError('harbor cannot process layer of dim {}'.format(len(shape)))

    if channel_op == 'add':
        output = tf.add_n(outputs, name='harbor')
    elif channel_op == 'multiply':
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
            output = self.harbor[0](inputs, self.harbor_shape, self.name, reuse=self._reuse, **self.harbor[1])
       
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
