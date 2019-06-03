import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import tfutils.model
from tnn.cell import *
from tnn.main import _get_func_from_kwargs
from tfutils.model_tool_old import conv, depth_conv
import copy

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

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
        shape = self.shape
        out_depth = self._out_depth
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype) 
        return zeros


def ksize(val):
    if isinstance(val, float):
        return [int(val), int(val)]
    elif isinstance(val, int):
        return [val, val]
    else:
        return val
    
class EfficientGateCell(ConvRNNCell):
    """
    """

    def __init__(self,
                 shape,
                 out_depth,
                 tau_filter_size,
                 in_depth=None,
                 cell_depth=0,                 
                 gate_filter_size=[3,3],
                 feedback_filter_size=[1,1],
                 activation=tf.nn.swish,
                 gate_nonlinearity=tf.nn.sigmoid,
                 kernel_initializer='variance_scaling',
                 kernel_initializer_kwargs={'seed':0},
                 weight_decay=None,
                 batch_norm=False,
                 batch_norm_decay=0.9,
                 batch_norm_epsilon=1e-5,
                 batch_norm_gamma_init=0.1,
                 group_norm=False,
                 strides=1,
                 se_ratio=0
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
        self.cell_depth = cell_depth
        self.strides = strides
        
        # functions
        self._relu = activaton
        self._se = tf.identity if not se_ratio \
                   else lambda x: squeeze_and_excitation(
                           x, reduction_ratio=se_ratio, activation=self._relu,
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
            'batch_norm_gamma_init': batch_norm_gamma_init
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

    def _drop_connect(self, inputs, is_training, drop_connect_rate):
        raise NotImplementedError("Need to implement drop connect with variable rate")

    def _conv_bn(self, inputs, ksize, scope, out_depth=None, depthwise=False, activation=True):
        if out_depth is None:
            out_depth = inputs.shape.as_list()[-1]
        kwargs = copy.deepcopy(self.conv_kwargs)
        kwargs.update(self.bn_kwargs)
        kwargs['ksize'] = ksize
        kwargs['activation'] = self._relu if activation else None

        def _conv_op(x):
            return depth_conv(x, **kwargs) if depthwise \
                else conv(x, out_depth, **kwargs)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            out = _conv_op(inputs)

        return out
    
    def __call__(self, inputs, state, fb_input, res_input, is_training=True, **training_kwargs):

        """
        """
        # update training-specific kwargs
        self.bn_kwargs['is_training'] = is_training
        self.bn_kwargs.update(training_kwargs)

        # get previous state
        prev_cell, prev_state = tf.split(value=state, num_or_size_splits=[self.cell_depth, self.out_depth], axis=3, name="state_split")

        # updates
        with tf.variable_scope(type(self).__name__): # "EfficientGateCell"

            # depthwise conv on expanded state, then squeeze-excitation, channel reduction, residual add
            next_out = inputs + prev_state # add zeros at first timestep
            next_out = self._se(next_out)
            next_out = self._conv_bn(next_out, [1,1], out_depth=self.out_depth, depthwise=False, activation=False, scope="state_to_out")
            if (res_input is not None) and (res_input.shape.as_list() == next_out.shape.as_list()):
                if self.bn_kwargs['drop_connect_rate']:
                    next_out = self._drop_connect(next_out, self.bn_kwargs['is_training'], self.bn_kwargs['drop_connect_rate'])
                next_out = tf.add(next_out, res_input)

            # update the state with a kxk depthwise conv/bn/relu
            update = self._conv_bn(inputs + prev_state, self.tau_filter_size, depthwise=True, activation=True, scope="state_to_state")
            next_state = prev_state + update

            # update the cell TODO
            next_cell = prev_cell
            
            # concat back on the cell
            next_state = tf.concat([next_cell, next_state], axis=3, name="cell_concat_next_state")
            
            return next_out, next_state
                

class tnn_ReciprocalGateCell(ConvRNNCell):

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

        # signature: ReciprocalGateCell(shape, ff_filter_size, cell_filter_size, cell_depth, out_depth, **kwargs)
        self._strides = self.pre_memory[0][1].get('strides', [1,1,1,1])[1:3]
        self.memory[1]['shape'] = self.memory[1].get('shape', [self.harbor_shape[1] // self._strides[0], self.harbor_shape[2] // self._strides[1]])

        idx = [i for i in range(len(self.pre_memory)) if 'out_depth' in self.pre_memory[i][1]][0]
        self._pre_conv_idx = idx
        if 'out_depth' not in self.memory[1]:
            self.memory[1]['out_depth'] = self.pre_memory[idx][1]['out_depth']

        mem_kwargs = copy.deepcopy(self.memory[1])
        mem_kwargs.pop('time_sep', None)
        self.conv_cell = ReciprocalGateCell(**mem_kwargs)


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

        with tf.variable_scope(self.name_tmp, reuse=self._reuse):

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape, **self.input_init[1])]

            # separate feedback from feedforward input
            fb_input = None
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

            res_input = None
            curr_time_suffix = 't' + str(self.internal_time)
            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if kwargs.get('time_sep', False):
                        kwargs['time_suffix'] = curr_time_suffix # used for scoping in the op

                    if function.__name__ == "component_conv":
                        if kwargs.get('return_input', False):
                            output, res_input = function(output, [inputs[ff_idx]], **kwargs) # component_conv needs to know the inputs
                        else:
                            output = function(output, [inputs[ff_idx]], **kwargs) # component_conv needs to know the inputs
                            
                    else:
                        output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                batch_size = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(batch_size, dtype=self.dtype_tmp)

            if self.memory[1].get('time_sep', False):
                output, state = self.conv_cell(output, state, fb_input, res_input, time_sep=True, time_suffix=curr_time_suffix)
            else:
                output, state = self.conv_cell(output, state, fb_input, res_input, time_sep=False, time_suffix=None)

            self.state = tf.identity(state, name="state")

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if kwargs.get('time_sep', False):
                        kwargs['time_suffix'] = curr_time_suffix # used for scoping in the op

                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
                
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            # Now reuse variables across time
            self._reuse = True

        self.state_shape = self.conv_cell.state_size() # DELETE?
        self.output_tmp_shape = self.output_tmp.shape # DELETE?

        self.internal_time = self.internal_time + 1

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
        # if self.output_tmp is not None:
        return self.output_tmp_shape
        # else:
        #     raise ValueError('Output not initialized yet')
