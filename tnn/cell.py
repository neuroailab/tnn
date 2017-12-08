"""
General Functional Cell
"""

from __future__ import absolute_import, division, print_function

import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import tnn.spatial_transformer
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
    feedbacks = node_nms[ff_idx+2:] # exclude ff input and itself, note no layer has ff_idx to be the last element

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
#                reshaped_inp = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                feedback_ins.append(inp)
            elif len(inp.shape) == 2:
                feedback_ins.append(inp)
            else:
                raise ValueError
        elif nm in skips:
            skip_ins.append(inp)
    
    return ff_in, skip_ins, feedback_ins

def input_aggregator(inputs, shape, spatial_op, channel_op, kernel_init='xavier', weight_decay=None, reuse=None, ff_inpnm=None, ksize=3, activation=None, kernel_init_kwargs=None, padding='SAME'):
    '''Helper function that combines the inputs appropriately based on the spatial and channel_ops'''

    outputs = []
    for inp in inputs:
        if len(shape) == 2:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                if channel_op != 'concat' and inp.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_fc_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, shape[1], kernel_init=kernel_init, kernel_init_kwargs=kernel_init_kwargs, weight_decay=weight_decay, activation=activation, batch_norm=False)

                outputs.append(inp)

            elif len(inp.shape) == 4:
                out = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                if channel_op != 'concat' and out.shape[1] != shape[1]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'conv_to_fc_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = tfutils.model.fc(out, shape[1], kernel_init=kernel_init, kernel_init_kwargs=kernel_init_kwargs, weight_decay=weight_decay, activation=activation, batch_norm=False)    

                outputs.append(out)
            else:
                raise ValueError

        elif len(shape) == 4:
            pat = re.compile(':|/')
            if len(inp.shape) == 2:
                nchannels = shape[3]
                old_channels = inp.shape[1]
                if nchannels != old_channels:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'fc_to_conv_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        inp = tfutils.model.fc(inp, nchannels, kernel_init=kernel_init, kernel_init_kwargs=kernel_init_kwargs, weight_decay=weight_decay, activation=activation, batch_norm=False)

                if spatial_op == 'emphasis':
                    if activation == 'softmax' and nchannels != old_channels:
                        # softmax has already been applied to the fc
                        # so now we multiply by nchannels to keep mean value as 1
                        channel_normalizer = tf.cast(nchannels, dtype=tf.float32)
                        inp = tf.multiply(channel_normalizer, inp)

                # we may choose a different activation (like relu) and/or
                # we did not need to learn an fc above so we directly apply the fc
                # to the conv input, but in all cases we tile
                xs, ys = shape[1: 3]
                inp = tf.tile(inp, [1, xs*ys])
                out = tf.reshape(inp, (inp.shape.as_list()[0], xs, ys, nchannels))

            elif len(inp.shape) == 4:
                if spatial_op == 'tile':
                    out = tile_func(inp, shape)
                elif spatial_op == 'pad':
                    out = tf.map_fn(lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]), inp, dtype=tf.float32)
                elif spatial_op == 'sp_transform':
                    out = transform_func(inp, shape=shape, weight_decay=weight_decay, ff_inpnm=ff_inpnm, reuse=reuse)
                elif spatial_op == 'flatten':
                    out = tf.reshape(inp, [inp.get_shape().as_list()[0], -1])
                elif spatial_op == 'deconv':
                    out = deconv(inp, shape=shape, weight_decay=weight_decay, ksize=ksize, activation=activation, padding=padding, reuse=reuse)
                else:
                    out = tf.image.resize_images(inp, shape[1:3])

                if channel_op != 'concat' and out.shape[3] != shape[3]:
                    nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
                    nm = 'conv_to_conv_harbor_for_%s' % nm
                    with tf.variable_scope(nm, reuse=reuse):
                        out = tfutils.model.conv(out, out_depth=shape[3], ksize=[1, 1], kernel_init=kernel_init, kernel_init_kwargs=kernel_init_kwargs, weight_decay=weight_decay, activation=activation, batch_norm=False)
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

def crop_func(inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse):
    # note: e.g. node_nms = ['split', 'V1', 'V2', 'V4', 'pIT', 'aIT']

    ff_in, skip_ins, feedback_ins = gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms)
    not_ff = feedback_ins + skip_ins
    if len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4: # we do nothing in this case, and proceed as usual (appeases initialization too)
        return inputs
    # combine skips and feedbacks to learn the crop from 
    not_ff_ins = tf.concat(not_ff, axis=-1, name='comb')
    mlp_nm = 'crop_mlp_for_%s' % ff_inpnm
    with tf.variable_scope(mlp_nm, reuse=reuse):
        mlp_out = tfutils.model.fc(not_ff_ins, 5, kernel_init=kernel_init, activation=None, batch_norm=False)

    alpha = tf.slice(mlp_out, [0, 0], [-1, 1])
    alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)
    alpha = tf.nn.tanh(alpha) # we want to potentially have negatives to downweight
    boxes = tf.slice(mlp_out, [0, 1], [-1, 4])
    boxes = tf.nn.sigmoid(boxes) # keep values in [0, 1] range
    # dimensions of original ff
    total_height = tf.constant(ff_in.get_shape().as_list()[1], dtype=tf.float32)
    total_width = tf.constant(ff_in.get_shape().as_list()[2], dtype=tf.float32)
    total_depth = tf.constant(ff_in.get_shape().as_list()[3], dtype=tf.int32)
    # compute bbox coords
    offset_height_frac = tf.squeeze(tf.slice(boxes, [0, 0], [-1, 1]), axis=-1)
    offset_height = tf.floor(total_height * offset_height_frac)
    offset_width_frac = tf.squeeze(tf.slice(boxes, [0, 1], [-1, 1]), axis=-1)
    offset_width = tf.floor(total_width * offset_width_frac)
    target_height_frac = tf.squeeze(tf.slice(boxes, [0, 2], [-1, 1]))
    target_height = tf.floor(total_height * target_height_frac)
    target_width_frac = tf.squeeze(tf.slice(boxes, [0, 3], [-1, 1]))
    target_width = tf.floor(total_width * target_width_frac)
    # clip height and width of bounding box
    height_val = tf.minimum(offset_height + target_height, total_height)
    width_val = tf.minimum(offset_width + target_width, total_width)
    clipped_target_height = height_val - offset_height
    clipped_target_width = width_val - offset_width
    rem_height = total_height - height_val
    rem_width = total_width - width_val
    # construct mask
    offset_height = tf.cast(offset_height, tf.int32)
    offset_width = tf.cast(offset_width, tf.int32)
    clipped_target_height = tf.cast(clipped_target_height, tf.int32)
    clipped_target_width = tf.cast(clipped_target_width, tf.int32)
    rem_height = tf.cast(rem_height, tf.int32)
    rem_width = tf.cast(rem_width, tf.int32)
    elems = (offset_height, offset_width, clipped_target_height, clipped_target_width, rem_height, rem_width)
    mask = tf.map_fn(lambda x: tf.pad(tf.ones([x[2], x[3], total_depth]), \
        [[x[0], x[4]], [x[1], x[5]], [0, 0]], \
        "CONSTANT"), elems, dtype=tf.float32)

    padded_img = tf.multiply(ff_in, mask)
    padded_img = tf.multiply(alpha, padded_img)
    pat = re.compile(':|/')
    ff_nm = pat.sub('__', ff_in.name.split('/')[-2].split('_')[0])
    new_name = ff_nm + '_mod'
    new_in = tf.add(ff_in, padded_img, name=new_name)

    new_out = [new_in]
    return new_out

def tile_func(inp, shape):
    inp_height = inp.get_shape().as_list()[1]
    inp_width = inp.get_shape().as_list()[2]
    height_multiple = 1 + (shape[1] // inp_height)
    width_multiple = 1 + (shape[2] // inp_width)
    tiled_out = tf.tile(inp, [1, height_multiple, width_multiple, 1])
    return tf.map_fn(lambda im: tf.image.resize_image_with_crop_or_pad(im, shape[1], shape[2]), tiled_out, dtype=tf.float32) 

def transform_func(inp, shape, weight_decay, ff_inpnm, reuse):
    '''Learn an affine transformation on the input inp if it is a feedback or skip'''
    pat = re.compile(':|/')
    orig_nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])
    assert(ff_inpnm is not None)
    if ff_inpnm in orig_nm:
        return tf.image.resize_images(inp, shape[1:3]) # simply do nothing with feedforward input
    else:
        nm = 'spatial_transform_for_%s' % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            resh = tf.reshape(inp, [inp.get_shape().as_list()[0], -1], name='reshape')
            in_depth = resh.get_shape().as_list()[-1]
            if weight_decay is None:
                weight_decay = 0.

            # identity initialization (weights = zeros and biases = identity)
            kernel = tf.get_variable(initializer=tf.zeros_initializer(),
                                   shape=[in_depth, 6],
                                   dtype=tf.float32,
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   name='weights')

            initial_theta = np.array([[1., 0, 0], [0, 1., 0]])
            initial_theta = initial_theta.astype('float32')
            initial_theta = initial_theta.flatten()
            biases = tf.get_variable(initializer=tf.constant_initializer(value=initial_theta),
                                   shape=[6],
                                   dtype=tf.float32,
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   name='bias')

            fcm = tf.matmul(resh, kernel)
            loc_out = tf.nn.bias_add(fcm, biases, name='loc_out')
            out_size = (shape[1], shape[2])
            assert(len(inp.shape) == 4) # we can only spatial transform on a convolutional input
            h_trans = tnn.spatial_transformer.transformer(inp, loc_out, out_size)
            bs = inp.get_shape().as_list()[0]
            cs = inp.get_shape().as_list()[-1]
            h_trans.set_shape([bs, shape[1], shape[2], cs])
            return h_trans

def deconv(inp, shape, weight_decay, ksize, activation, padding, reuse):
    pat = re.compile(':|/')
    if len(inp.name.split('/')) == 1:
        orig_nm = pat.sub('__', inp.name.split('/')[-1].split('_')[0])
    else:
        orig_nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])

    if inp.shape[1] == shape[1] and inp.shape[2] == shape[2] and inp.shape[3] == shape[3]:
        return tf.image.resize_images(inp, shape[1:3]) # simply do nothing with feedforward input or inputs of the same shape
    elif inp.shape[1] > shape[1] or inp.shape[2] > shape[2]: # e.g. if connection is a skip
        nm = 'deconv_for_%s' % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            if weight_decay is None:
                weight_decay = 0.0
            if isinstance(ksize, int):
                ksize = [ksize, ksize]
            in_ch = inp.get_shape().as_list()[-1]
            out_ch = shape[3]

            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                     shape=[ksize[0], ksize[1], in_ch, out_ch],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='weights')  

            biases = tf.get_variable(initializer=tf.zeros_initializer(),
                                     shape=[out_ch],
                                     dtype=tf.float32,
                                     regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                     name='bias')

            # simply do a size-matching conv op
            stride_0 = inp.get_shape()[1] // shape[1]
            stride_1 = inp.get_shape()[2] // shape[2]
            if inp.get_shape().as_list()[1] > shape[1]  and inp.get_shape().as_list()[1] % shape[1] != 0: # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
               stride_0 += 1
            if inp.get_shape().as_list()[2] > shape[2] and inp.get_shape().as_list()[2] % shape[2] != 0: # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
               stride_1 += 1
            conv = tf.nn.conv2d(inp, kernel,
                                strides=[1, stride_0, stride_1, 1],
                                padding=padding)

            output = tf.nn.bias_add(conv, biases, name='deconv_out')
            if activation is not None:
                output = getattr(tf.nn, activation)(output, name=activation)
            return output
        
    else: # a feedback that requires transposed convolution
        nm = 'deconv_for_%s' % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
           if weight_decay is None:
               weight_decay = 0.
           if isinstance(ksize, int):
               ksize = [ksize, ksize]
           in_ch = inp.get_shape().as_list()[-1]
           out_ch = shape[3]
           kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(),
                                    shape=[ksize[0], ksize[1], out_ch, in_ch],
                                    dtype=tf.float32,
                                    regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                    name='weights')  

           biases = tf.get_variable(initializer=tf.zeros_initializer(),
                                   shape=[out_ch],
                                   dtype=tf.float32,
                                   regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                   name='bias')

           stride_0 = shape[1] // inp.get_shape().as_list()[1]
           stride_1 = shape[2] // inp.get_shape().as_list()[2]
           if shape[1] > inp.get_shape().as_list()[1] and shape[1] % inp.get_shape().as_list()[1] != 0: # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
               stride_0 += 1
           if shape[2] > inp.get_shape().as_list()[2] and shape[2] % inp.get_shape().as_list()[2] != 0: # e.g. 7 // 4 = 1, but should be 2, probably so we want to round up in non-divisible cases
               stride_1 += 1
           conv_t = tf.nn.conv2d_transpose(inp, kernel,
                                           output_shape=shape,
                                           strides=[1, stride_0, stride_1, 1],
                                           padding=padding)

           output = tf.nn.bias_add(conv_t, biases, name='deconv_out')
           if activation is not None:
               output = getattr(tf.nn, activation)(output, name=activation)
           return output

def sptransform_preproc(inputs, l1_inpnm, ff_inpnm, node_nms, shape, spatial_op, channel_op, kernel_init, weight_decay, dropout, reuse):
    '''Learn an affine transformation on the feedforward inputs (including skips) using the feedbacks
    into that layer'''
    ff_in, skip_ins, feedback_ins = gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms)
    new_inputs = [ff_in]
    not_ff = feedback_ins + skip_ins
    #print('New inputs: ', new_inputs)
    # Note you must pass to_exclude to the init_nodes method in main.py if using the concat channel op
    # since the total number of channels excludes the feedback channels as we are not combining them
    # otherwise it does not need to change since the number of input channels is unchanged
    if channel_op == 'concat':
        print('Make sure to exclude feedback nodes in the main.init_nodes() method!')

    if len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4: # we do nothing in this case, and proceed as usual (appeases initialization too)
        out_val = input_aggregator(inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm)
        return out_val

    # aggregate feedforward input
    ff_out = input_aggregator(new_inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm)
    #print('ff out: ', ff_out.shape, 'shape: ', shape)
    # combine skips and feedbacks to learn the affine transform from 
    not_ff_ins = tf.concat(not_ff, axis=-1, name='comb')
    if dropout is not None:
        not_ff_ins = tf.nn.dropout(not_ff_ins, keep_prob=dropout)
    #print('Feedback_in shape: ', not_ff_ins.shape)
    mlp_nm = 'spatial_transform_for_%s' % ff_inpnm
    #print('MLP NAME: ', mlp_nm)
    # we use the feedbacks to learn our localizer network
    with tf.variable_scope(mlp_nm, reuse=reuse):
        # might want to use tfutils fc to shorten this
        in_depth = not_ff_ins.get_shape().as_list()[-1]

        # identity initialization (weights = zeros and biases = identity)
        kernel = tf.get_variable(initializer=tf.zeros_initializer(),
                               shape=[in_depth, 6],
                               dtype=tf.float32,
                               name='weights')

        initial_theta = np.array([[1., 0, 0], [0, 1., 0]])
        initial_theta = initial_theta.astype('float32')
        initial_theta = initial_theta.flatten()
        biases = tf.get_variable(initializer=tf.constant_initializer(value=initial_theta),
                               shape=[6],
                               dtype=tf.float32,
                               name='bias')

        fcm = tf.matmul(not_ff_ins, kernel)
        loc_out = tf.nn.bias_add(fcm, biases, name='loc_out')
        out_size = (shape[1], shape[2])
        assert(len(ff_out.shape) == 4) # we can only spatial transform on a convolutional input
        h_trans = tnn.spatial_transformer.transformer(ff_out, loc_out, out_size)
        bs = ff_out.get_shape().as_list()[0]
        cs = ff_out.get_shape().as_list()[-1]
        h_trans.set_shape([bs, shape[1], shape[2], cs])
        return h_trans

def depth_preproc(inputs, l1_inpnm, ff_inpnm, node_nms, shape, spatial_op='resize', channel_op='concat', kernel_init='xavier', weight_decay=None, reuse=None, ksize=3, activation=None, kernel_init_kwargs=None):
    '''Separates feedback from feedforward inputs and then combines the non feedforward inputs together'''
    ff_in, skip_ins, feedback_ins = gather_inputs(inputs, shape, l1_inpnm, ff_inpnm, node_nms)
    not_ff = feedback_ins + skip_ins
    dict_out = {'ff': None, 'non_ff': None}

    if len(not_ff) == 0 or ff_in is None or len(shape) != 4 or len(ff_in.shape) != 4: # we do nothing in this case, and proceed as usual (appeases initialization too)
        out_val = input_aggregator(inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs)
        dict_out['ff'] = out_val
        return dict_out

    # aggregate non feedforward input (resize and concat by default)
    # non_ff_out = input_aggregator(not_ff, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs)
    dict_out['ff'] = ff_in
    dict_out['non_ff'] = not_ff
    return dict_out

def gate_preproc(inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs, padding):
    '''creates a gate for each feedback where there is a term for every feedback based on the pre memory output that the ff input is fed to
    then multiplies the gate with each feedback'''
    feedback_inps = []
    ff_inp = []
    for inp in inputs:
        if inp.shape[1] == shape[1] and inp.shape[2] == shape[2] and inp.shape[3] == shape[3]:
            ff_inp.append(inp) # simply do nothing with feedforward input or inputs of the same shape
        else:
            feedback_inps.append(inp)

    out_terms = ff_inp
    for inp in feedback_inps: # create gates for each feedback 
        pat = re.compile(':|/')
        if len(inp.name.split('/')) == 1:
            orig_nm = pat.sub('__', inp.name.split('/')[-1].split('_')[0])
        else:
            orig_nm = pat.sub('__', inp.name.split('/')[-2].split('_')[0])

        nm = 'gate_for_%s' % orig_nm
        with tf.variable_scope(nm, reuse=reuse):
            gate_vars = ff_inp
            for fb in feedback_inps: # create gate variables for each feedback
                d_out = deconv(fb, shape=shape, weight_decay=weight_decay, ksize=ksize, activation=activation, padding=padding, reuse=reuse)
                gate_vars.append(d_out)
            gate_name = 'gate_out_for_%s' % orig_nm
            linear_comp = tf.add_n(gate_vars, name=gate_name)
            gate_out = tf.nn.sigmoid(linear_comp)

        # transform each feedback to the right shape, then multiply by the gate
        transform_nm = 'transform_for_%s' % orig_nm
        with tf.variable_scope(transform_nm, reuse=reuse):
            lin_transform_out = deconv(inp, shape=shape, weight_decay=weight_decay, ksize=ksize, activation=activation, padding=padding, reuse=reuse)
            gate_comb = gate_out * lin_transform_out
            out_terms.append(gate_comb)

    output = tf.add_n(out_terms, name='transform_out')
    return output

def harbor(inputs, shape, name, ff_inpnm=None, node_nms=['split', 'V1', 'V2', 'V4', 'pIT', 'aIT'], l1_inpnm='split', preproc=None, spatial_op='resize', channel_op='concat', kernel_init='xavier', kernel_init_kwargs=None, weight_decay=None, dropout=None, ksize=3, activation=None, padding='SAME', reuse=None):
    """
    Default harbor function which can crop the input (as a preproc), followed by a spatial_op which by default resizes inputs to a desired shape (or pad or tile), and finished with a channel_op which by default concatenates along the channel dimension (or add or multiply based on user specification).

    :Args:
        - inputs
        - shape
    """
    if preproc == 'crop':
        inputs = crop_func(inputs, l1_inpnm, ff_inpnm, node_nms, shape, kernel_init, channel_op, reuse)
    elif preproc == 'depth':
        output = depth_preproc(inputs, l1_inpnm, ff_inpnm, node_nms, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ksize, activation, kernel_init_kwargs)
        return output
    elif preproc == 'sp_transform':
        # skips and feedforward inputs were combined already and then transformed by the feedback
        output = sptransform_preproc(inputs, l1_inpnm, ff_inpnm, node_nms, shape, spatial_op, channel_op, kernel_init, weight_decay, dropout, reuse)
        return output
    elif preproc == 'gate':
        output = gate_preproc(inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs, padding)
        return output

    output = input_aggregator(inputs, shape, spatial_op, channel_op, kernel_init, weight_decay, reuse, ff_inpnm, ksize, activation, kernel_init_kwargs, padding)

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

def component_conv(inp,
         inputs_list,
         out_depth,
         input_name=None,
         ksize=[3,3],
         strides=[1,1,1,1],
         padding='SAME',
         kernel_init='zeros',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=True,
         name='component_conv'
         ):

    """
    Function that breaks up the convolutional kernel to its basenet and non basenet components, when given
the name of its feedforward input. This is useful when loading basenet weights into tnn when using a 
harbor channel op of concat. Other channel ops should work with tfutils.model.conv just fine.
    """

    assert input_name is not None
    # assert out_shape is not None
    if weight_decay is None:
        weight_decay = 0.
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if kernel_init_kwargs is None:
        kernel_init_kwargs = {}
    in_depth = inp.get_shape().as_list()[-1]

    # weights
    init = tfutils.model.initializer(kernel_init, **kernel_init_kwargs)
    kernel_list = []
    w_idx = 0
    for input_elem in inputs_list:
       if input_name is not None and input_name in input_elem.name:
            kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], input_elem.get_shape().as_list()[-1], out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights_basenet')
       else:
            kernel = tf.get_variable(initializer=init,
                            shape=[ksize[0], ksize[1], input_elem.get_shape().as_list()[-1], out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='weights_' + str(w_idx))
            w_idx += 1

       kernel_list.append(kernel)


    new_kernel = tf.concat(kernel_list, axis=-2, name='weights')
    const_init = tfutils.model.initializer(kind='constant', value=bias)
    biases = tf.get_variable(initializer=const_init,
                            shape=[out_depth],
                            dtype=tf.float32,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                            name='bias')
    # ops
    conv = tf.nn.conv2d(inp, new_kernel,
                        strides=strides,
                        padding=padding)
    output = tf.nn.bias_add(conv, biases, name=name)

    if activation is not None:
        output = getattr(tf.nn, activation)(output, name=activation)
    if batch_norm:
        output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                            scale=None, variance_epsilon=1e-8, name='batch_norm')
    return output

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

        self.dtype_tmp = dtype
        self.name_tmp = name

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

        with tf.variable_scope(self.name_tmp, reuse=self._reuse):
            # inputs_full = []
            # for inp, shape, dtype in zip(inputs, self.input_shapes, self.input_dtypes):
            #     if inp is None:
            #         inp = self.output_init[0](shape=shape, dtype=dtype, **self.output_init[1])
            #     inputs_full.append(inp)

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs) # component_conv needs to know the inputs
                    else:
                       output = function(output, **kwargs)
                pre_name_counter += 1
            if state is None:
                state = self.state_init[0](shape=output.shape,
                                           dtype=self.dtype_tmp,
                                           **self.state_init[1])
            state = self.memory[0](output, state, **self.memory[1])
            self.state = tf.identity(state, name='state')

            output = self.state
            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')
            # scope.reuse_variables()
            self._reuse = True
        self.state_shape = self.state.shape
        self.output_shape_tmp = self.output_tmp.shape
        return self.output_tmp, self.state

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
        return self.output_shape_tmp
        # else:
        #     raise ValueError('Output not initialized yet')
