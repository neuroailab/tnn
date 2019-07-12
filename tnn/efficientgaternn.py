import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tfutils.model
from tnn.cell import *
from tnn.main import _get_func_from_kwargs
try:
    from tfutils.model import conv, depth_conv
except:
    from tfutils.model_tool_old import conv, depth_conv
import copy

def ksize(val):
    if isinstance(val, float):
        return [int(val), int(val)]
    elif isinstance(val, int):
        return [val, val]
    else:
        return val

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """
    def __init__(self, shape, out_depth, scope):
        self.shape = shape
        self._out_depth = out_depth
        self.scope = scope
    
    def __call__(self, inputs, state=None, fb_input=None, res_input=None, **training_kwargs):
        """Run this RNN cell on inputs, starting from the given state.
        """
        with tf.variable_scope(type(self).__name__ + '_' + self.scope): # "ConvRNNCell + self.scope#
            self.next_state = state
            output = tf.identity(inputs, name="convrnn_cell_passthrough")
            return output, self.next_state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
        batch_size: int, float, or unit Tensor representing the batch size.
        dtype: the data type to use for the state.
        Returns:
        tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
        filled with zeros
        """
        return None
        # shape = self.shape
        # out_depth = self._out_depth
        # zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype) 
        # return zeros
    
class EfficientGateCell(ConvRNNCell):
    """
    """

    def __init__(self,
                 shape,
                 out_depth,
                 tau_filter_size,
                 in_depth=None,
                 cell_depth=0,
                 bypass_state=False,
                 gate_filter_size=[3,3],
                 feedback_filter_size=[1,1],
                 activation="swish",
                 gate_nonlinearity=tf.nn.sigmoid,
                 kernel_initializer='normal',
                 kernel_initializer_kwargs={},
                 weight_decay=None,
                 batch_norm=False,
                 batch_norm_decay=0.9,
                 batch_norm_epsilon=1e-5,
                 batch_norm_gamma_init=1.0,
                 crossdevice_bn_kwargs={},
                 group_norm=False,
                 num_groups=32,
                 strides=1,
                 se_ratio=0,
                 residual_add=False,
                 res_cell=False
    ):
        """ 
        Initialize the memory function of the EfficientGateCell.

        """
        # shapes and filter sizes
        self.shape = shape # [H, W]
        self.tau_filter_size = ksize(tau_filter_size)
        self.gate_filter_size = ksize(gate_filter_size)
        self.feedback_filter_size = ksize(feedback_filter_size)
        self.out_depth = out_depth
        self.in_depth = self.out_depth if in_depth is None else in_depth
        self.res_cell = res_cell
        self.cell_depth = cell_depth
        self.bypass_state = bypass_state
        self.strides = strides
        self.residual_add = residual_add

        # functions
        self._relu = activation
        self._se_ratio = se_ratio
        self._se = tf.identity if not se_ratio \
                   else lambda x, rC: squeeze_and_excitation(
                           x, rC, activation=self._relu,
                           kernel_init=kernel_initializer,
                           kernel_init_kwargs=kernel_initializer_kwargs
                   )

        # function kwargs
        self.bn_kwargs = {
            'batch_norm': batch_norm and not group_norm,
            'group_norm': group_norm,
            'num_groups': num_groups,
            'batch_norm_decay': batch_norm_decay,
            'batch_norm_epsilon': batch_norm_epsilon,
            'batch_norm_gamma_init': batch_norm_gamma_init,
            'crossdevice_bn_kwargs': crossdevice_bn_kwargs
        }
        self.conv_kwargs = {
            'strides': self.strides,            
            'kernel_init': kernel_initializer,
            'kernel_init_kwargs': kernel_initializer_kwargs,
            'weight_decay': weight_decay,
            'data_format': 'channels_last',
            'padding': 'SAME',
            'use_bias': False
        }
        
    def zero_state(self, batch_size, dtype):
        """
        Return zero-filled state tensor(s)
        """
        return tf.zeros([batch_size, self.shape[0], self.shape[1], self.cell_depth + self.in_depth], dtype=dtype)


    def _conv_bn(self, inputs, ksize, scope, out_depth=None, depthwise=False, activation=True):
        if out_depth is None:
            out_depth = inputs.shape.as_list()[-1]
        kwargs = copy.deepcopy(self.conv_kwargs)
        kwargs.update(self.bn_kwargs)
        kwargs['ksize'] = ksize
        kwargs['activation'] = self._relu if activation else None

        # def _conv_op(x):
        #     return depth_conv(x, **kwargs) if depthwise \
        #         else conv(x, out_depth, **kwargs)
        
        with tf.variable_scope(scope):
            inputs = depth_conv(inputs, **kwargs) if depthwise else conv(inputs, out_depth, **kwargs)

        return inputs
    
    def __call__(self, inputs, state, fb_input, res_input, is_training=True, **training_kwargs):

        """
        """
        # update training-specific kwargs
        self.bn_kwargs['is_training'] = is_training
        self.bn_kwargs.update({'time_suffix': training_kwargs.get('time_suffix', None),
                               'time_sep': training_kwargs.get('time_sep', True)}) # time suffix
        self.res_depth = res_input.shape.as_list()[-1] if res_input is not None else self.out_depth        
        # print("bn kwargs", self.bn_kwargs['time_suffix'])
        
        # get previous state
        prev_cell, prev_state = tf.split(value=state, num_or_size_splits=[self.cell_depth, self.in_depth], axis=3, name="state_split")

        # updates
        with tf.variable_scope(type(self).__name__): # "EfficientGateCell"

            update = tf.zeros_like(inputs)
            # combine fb input with ff input
            if fb_input is not None:
                update += self._conv_bn(fb_input, self.feedback_filter_size, out_depth=self.in_depth, depthwise=False, activation=True, scope="feedback_to_state")
                print("added feedback: %s of shape %s" % (fb_input.name, fb_input.shape.as_list()))

            # update the state with a kxk depthwise conv/bn/relu
            update += self._conv_bn(inputs + prev_state, self.tau_filter_size, depthwise=True, activation=True, scope="state_to_state")
            next_state = prev_state + update

            # update the cell TODO
            if self.res_cell:
                assert res_input is not None
                next_cell = res_input
            else:
                next_cell = prev_cell
            
            # depthwise conv on expanded state, then squeeze-excitation, channel reduction, residual add
            inp = next_state if not self.bypass_state else (inputs + prev_state)
            print("bypassed state?", self.bypass_state)
            next_out = self._se(inp, self._se_ratio * self.res_depth)
            next_out = self._conv_bn(next_out, [1,1], out_depth=self.out_depth, depthwise=False, activation=False, scope="state_to_out")
            if (res_input is not None) and (res_input.shape.as_list() == next_out.shape.as_list()):
                next_out = drop_connect(next_out, self.bn_kwargs['is_training'], training_kwargs['drop_connect_rate'])
                print("drop connect/residual adding", training_kwargs['drop_connect_rate'], res_input.name, res_input.shape.as_list())
                next_out = tf.add(next_out, res_input)
            elif (res_input is not None) and self.residual_add: # add the matching channels with resize if necessary
                next_out = drop_connect(next_out, self.bn_kwargs['is_training'], training_kwargs['drop_connect_rate'])                
                next_out, remainder = tf.split(next_out, [res_input.shape.as_list()[-1], -1], axis=-1)
                if res_input.shape.as_list()[1:3] != self.shape:
                    res_input = tf.image.resize_images(res_input, size=self.shape)
                next_out = tf.add(next_out, res_input)
                next_out = tf.concat([next_out, remainder], axis=-1)
                print("added matching channels", next_out.shape.as_list())                

            # concat back on the cell
            next_state = tf.concat([next_cell, next_state], axis=3, name="cell_concat_next_state")
            
            return next_out, next_state
                
class tnn_EfficientGateCell(ConvRNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None):

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
        self.internal_time = 0
        self.max_internal_time = self.memory[1].get('max_internal_time', None)

        # Kwargs for trainining/validation"
        self.is_training = self.memory[1].get('is_training', True)
        self.training_kwargs = {
            'time_sep': self.memory[1].get('time_sep', True),
            'dropout_rate': self.memory[1].get('dropout_rate', 1),
            'drop_connect_rate': self.memory[1].get('drop_connect_rate', 1)
        }
        
        ### Memory includes both a typical ConvRNN cell (optional) and an IntegratedGraphCell (optional) ###
        self.convrnn_cell_kwargs = self.memory[1].get('convrnn_cell_kwargs', {})
        idx = [i for i in range(len(self.pre_memory)) if 'out_depth' in self.pre_memory[i][1]]
        idx = idx[-1] if len(idx) else None
        self.pre_memory_out_depth = self.pre_memory[idx][1]['out_depth'] if idx is not None else self.harbor_shape[-1]
        self._pre_conv_idx = idx

        if self.memory[1].get('convrnn_cell', None) == "EfficientGateCell":
            if 'in_depth' not in self.convrnn_cell_kwargs:
                self.convrnn_cell_kwargs['in_depth'] = self.pre_memory_out_depth # the expansion width
            if 'out_depth' not in self.convrnn_cell_kwargs:
                self.convrnn_cell_kwargs['out_depth'] = self.harbor_shape[-1] # channels coming out of harbor
            self.convrnn_cell = EfficientGateCell(**self.convrnn_cell_kwargs)
        else:
            self.convrnn_cell_kwargs['out_depth'] = self.pre_memory_out_depth
            self.convrnn_cell = ConvRNNCell(shape=self.convrnn_cell_kwargs.get('shape', None),
                                            out_depth=self.convrnn_cell_kwargs['out_depth'],
                                            scope="0")

        # not used in this cell 
        self.graph_cell = ConvRNNCell(shape=self.convrnn_cell_kwargs.get('shape', None),
                                      out_depth=self.convrnn_cell_kwargs['out_depth'],
                                      scope="1")
        
    def __call__(self, inputs=None, state=None):

        with tf.variable_scope(self.name_tmp, reuse=self._reuse):

            # state initializer
            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape, **self.input_init[1])]

            # Pass inputs through harbor
            fb_input = None            
            if self.memory[1].get('convrnn_cell', None) in ["EfficientGateCell"]:
                if len(inputs) == 1:
                    ff_idx = 0
                    output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, reuse=self._reuse, **self.harbor[1])

                elif len(inputs) > 1:
                    for j, inp in enumerate(inputs):
                        if self.pre_memory[self._pre_conv_idx][1]['input_name'] in inp.name:
                            ff_inpnm = inp.name
                            ff_idx = j
                            ff_depth = inputs[ff_idx].shape.as_list()[-1]
                    output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, ff_inpnm=ff_inpnm, reuse=self._reuse, **self.harbor[1])
                    fb_depth = output.shape.as_list()[-1] - ff_depth
                    if self.harbor[1]['channel_op'] == 'concat':
                        output, fb_input = tf.split(output, num_or_size_splits=[ff_depth, fb_depth], axis=3)
                        fb_size = fb_input.shape.as_list()[1:3]

            else:
                ff_idx = None
                output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, reuse=self._reuse, **self.harbor[1])


            # pre_memory usually contains feedforward convolutions, etc.
            res_input = None
            curr_time_suffix = 't' + str(self.internal_time)

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if kwargs.get('time_sep', False):
                        kwargs['time_suffix'] = curr_time_suffix # for scoping unshared BN
                    if function.__name__ == "component_conv":
                        ff_inputs = [inputs[ff_idx]] if ff_idx is not None else inputs
                        if kwargs.get('return_input', False):
                            output, res_input = function(output, ff_inputs, **kwargs) # component_conv needs to know the inputs
                        else:
                            output = function(output, ff_inputs, **kwargs) # component_conv needs to know the inputs

                    else:
                       output = function(output, **kwargs)
                    # output = tf.Print(output, [tf.shape(output), output.name.split('/')[-1], tf.reduce_max(output)], message="output of %s" % tf.get_variable_scope().name)                                                                                                            
                pre_name_counter += 1



            # memory for this TNN layer includes an optional convrnn_cell
            self.next_state = {}
            if state is None or state.get('convrnn_cell_state', None) is None:
                batch_size = output.shape.as_list()[0]
                convrnn_cell_state = self.convrnn_cell.zero_state(batch_size, dtype=self.dtype_tmp)
                state = {'convrnn_cell_state': convrnn_cell_state}

            # resize fb if there was a strided convolution, for instance
            ff_size = output.shape.as_list()[1:3]
            if fb_input is not None:
                if fb_size != ff_size: 
                    fb_input = tf.image.resize_images(fb_input, size=ff_size)

            self.training_kwargs['time_suffix'] = curr_time_suffix                    
            output, convrnn_cell_state = self.convrnn_cell(output, state['convrnn_cell_state'], fb_input=fb_input, res_input=res_input, is_training=self.is_training, **self.training_kwargs)
            self.next_state['convrnn_cell_state'] = convrnn_cell_state

            # graph cell is not used here currently
            if state is None or state.get('graph_cell_state', None) is None:
                batch_size = output.shape.as_list()[0]
                graph_cell_state = self.graph_cell.zero_state(batch_size, dtype=self.dtype_tmp)
                state = {'graph_cell_state': graph_cell_state}

            output, graph_cell_state = self.graph_cell(output, state['graph_cell_state'])
            self.next_state['graph_cell_state'] = graph_cell_state
            
            # post memory functions (e.g. more convs, pooling)
            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if kwargs.get('time_sep', False):
                        kwargs['time_suffix'] = curr_time_suffix # for scoping unshared BN
                    
                    if function.__name__ == "component_conv":
                        output = function(output, inputs, **kwargs)
                    elif function.__name__ == "residual_add":
                        output = function(output, res_input, is_training=self.is_training, drop_connect_rate=self.training_kwargs['drop_connect_rate'], **kwargs)
                    else:
                        output = function(output, **kwargs)
                post_name_counter += 1

            # layer output
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')
            self._reuse = True
            
        if (self.max_internal_time is None) or ((self.max_internal_time is not None) and (self.internal_time < self.max_internal_time)):
            self.internal_time += 1
        return self.output_tmp, self.next_state
    
