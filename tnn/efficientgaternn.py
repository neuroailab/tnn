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
                 activation="swish",
                 gate_nonlinearity=tf.nn.sigmoid,
                 kernel_initializer='variance_scaling',
                 kernel_initializer_kwargs={'seed':0},
                 weight_decay=None,
                 batch_norm=False,
                 batch_norm_decay=0.9,
                 batch_norm_epsilon=1e-5,
                 batch_norm_gamma_init=1.0,
                 group_norm=False,
                 num_groups=32,
                 strides=1,
                 se_ratio=0,
                 residual_add=False
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
        self.residual_add = residual_add

        # functions
        self._relu = activation
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
        self.bn_kwargs.update({'time_suffix': training_kwargs.get('time_suffix', None),
                               'time_sep': training_kwargs.get('time_sep', True)}) # time suffix
        print("bn kwargs", self.bn_kwargs)
        
        # get previous state
        prev_cell, prev_state = tf.split(value=state, num_or_size_splits=[self.cell_depth, self.in_depth], axis=3, name="state_split")

        # updates
        with tf.variable_scope(type(self).__name__): # "EfficientGateCell"

            # depthwise conv on expanded state, then squeeze-excitation, channel reduction, residual add
            next_out = inputs + prev_state # add zeros at first timestep
            next_out = self._se(next_out)
            next_out = self._conv_bn(next_out, [1,1], out_depth=self.out_depth, depthwise=False, activation=False, scope="state_to_out")
            if (res_input is not None) and (res_input.shape.as_list() == next_out.shape.as_list()):
                if training_kwargs.get('drop_connect_rate', 0):
                    next_out = self._drop_connect(next_out, self.bn_kwargs['is_training'], training_kwargs['drop_connect_rate'])
                print("residual adding", res_input.name, res_input.shape.as_list())
                next_out = tf.add(next_out, res_input)
            elif (res_input is not None) and self.residual_add: # add the matching channels
                next_out, remainder = tf.split(next_out, [res_input.shape.as_list()[-1], -1], axis=-1)
                next_out = tf.add(next_out, res_input)
                next_out = tf.concat([next_out, remainder], axis=-1)
                print("added matching channels", next_out.shape.as_list())                

            # update the state with a kxk depthwise conv/bn/relu
            update = self._conv_bn(inputs + prev_state, self.tau_filter_size, depthwise=True, activation=True, scope="state_to_state")
            next_state = prev_state + update

            # update the cell TODO
            next_cell = prev_cell
            
            # concat back on the cell
            next_state = tf.concat([next_cell, next_state], axis=3, name="cell_concat_next_state")
            
            return next_out, next_state
                
