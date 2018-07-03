'''
Code adapted from 
https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow/blob/master/BasicConvLSTMCell.py
'''

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tnn.cell import *

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

class ConvBasicCell(ConvRNNCell):
  """Conv basic recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Basic cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state):
    """Basic RNN cell."""
    with tf.variable_scope(type(self).__name__):  # "ConvBasicCell"
      output = self._activation(
        _conv_linear([inputs, state], self.filter_size, self._out_depth, True,
            self._bias_initializer, self._kernel_initializer))

      return output, output


class ConvLinearBasicCell(ConvRNNCell):
  """Conv linear recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth, 
               activation=None,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Linear Basic cell. output, state = ff + W*state
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state):
    """Basic RNN cell."""
    with tf.variable_scope(type(self).__name__):  # "ConvLinearBasicCell"
      if self._activation is not None:
        new_state = _conv_linear([state], self.filter_size, self._out_depth, True,
            self._bias_initializer, self._kernel_initializer)
      else:
        new_state = self._activation(_conv_linear([state], self.filter_size, self._out_depth, True,
            self._bias_initializer, self._kernel_initializer))

      new_output = new_state + inputs #inputs is W_ffx_ff + b_ff
      return new_output, new_state


class ConvNormBasicCell(ConvRNNCell):
  """Conv norm recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               layer_norm=True,
               kernel_regularizer=5e-4,
               bias_regularizer=5e-4,
               activation=tf.nn.elu,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Norm Basic cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._layer_norm = layer_norm

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state):
    """Basic RNN cell."""
    with tf.variable_scope(type(self).__name__):  # "ConvNormBasicCell"
      if self._activation is not None:
        with tf.variable_scope("s"):
          s = _conv_linear([state], self.filter_size, self._out_depth, True,
            self._bias_initializer, self._kernel_initializer, self._bias_regularizer, self._kernel_regularizer)

        with tf.variable_scope("i"):
          i = _conv_linear([inputs], self.filter_size, self._out_depth, True,
            self._bias_initializer, self._kernel_initializer, self._bias_regularizer, self._kernel_regularizer)

        if self._layer_norm:
          new_state = tf.contrib.layers.layer_norm(i + s,
                                     activation_fn=self._activation,
                                     reuse=tf.AUTO_REUSE,
                                     scope='layer_norm'
                                    )
        else:
          new_state = self._activation(i + s)

      return new_state, new_state


class ConvLargeBasicCell(ConvRNNCell):
  """Conv basic recurrent network cell. Here we have a large hidden state to 
  mimic the number of parameters as the ConvLSTM as a control.
  """

  def __init__(self,
               shape,
               filter_size, 
               state_depth,
               out_depth, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Basic cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the hidden state
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth
    self._state_depth = state_depth 
    self._state_size = tf.TensorShape([self.shape[0], self.shape[1], self._state_depth])
    self._output_size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x state_depth]
      filled with zeros
    """
    shape = self.shape
    state_depth = self._state_depth
    zeros = tf.zeros([batch_size, shape[0], shape[1], state_depth], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Basic RNN cell with a large hidden state."""
    with tf.variable_scope(type(self).__name__):  # "ConvLargeBasicCell"
      hidden_output = self._activation(
        _conv_linear([inputs, state], self.filter_size, self._state_depth, True,
            self._bias_initializer, self._kernel_initializer))

      # perform a 1x1 conv on the hidden output to get the right channels into the next layer
      with tf.variable_scope('downsample'):
          output = _conv_linear([hidden_output], [1, 1], self._out_depth, True,
             self._bias_initializer, self._kernel_initializer)
      return output, hidden_output

class ConvGRUCell(ConvRNNCell):
  """Conv GRU recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv GRU cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      activation: Activation function of the inner states.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state):
    """Gated recurrent unit (GRU)."""
    with tf.variable_scope(type(self).__name__):  # "ConvGRUCell"
      with tf.variable_scope("gates"):
        # We start with bias of 1.0 to not reset and not update.
        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
          dtype = [a.dtype for a in [inputs, state]][0]
          bias_ones = tf.constant_initializer(1.0, dtype=dtype)
        value = tf.nn.sigmoid(
              _conv_linear([inputs, state], self.filter_size, 2*self._out_depth, True, bias_ones,
                self._kernel_initializer))
        r, u = tf.split(value=value, num_or_size_splits=2, axis=3)

      with tf.variable_scope("candidates"):
        c = self._activation(
          _conv_linear([inputs, r * state], self.filter_size, self._out_depth, True,
              self._bias_initializer, self._kernel_initializer))

      new_h = u * state + (1 - u) * c
      return new_h, new_h

class ConvLSTMCell(ConvRNNCell):
  """Conv LSTM recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=False, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None,
               weight_decay=0.0,
               layer_norm=False,
               norm_gain=1.0,
               norm_shift=0.0):
    """Initialize the Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      use_peepholes: bool, set True to enable peephole connections
      activation: Activation function of the inner states.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation == "elu":
        self._activation = tf.nn.elu
    else:
        self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._layer_norm = layer_norm
    self._weight_decay = weight_decay
    self._g = norm_gain
    self._b = norm_shift

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def _norm(self, inp, scope):
      shape = inp.get_shape()[-1:]
      gamma_init = tf.constant_initializer(self._g)
      beta_init = tf.constant_initializer(self._b)
      with tf.variable_scope(scope):
          gamma = tf.get_variable(shape=shape, initializer=gamma_init, name="gamma")
          beta = tf.get_variable(shape=shape, initializer=beta_init, name="beta")

      normalized = tf.contrib.layers.layer_norm(inp, reuse=True, scope=scope)
      return normalized

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

      concat = _conv_linear([inputs, h], \
                            self.filter_size, self._out_depth * 4, True, self._bias_initializer, self._kernel_initializer, kernel_regularizer=self._weight_decay)


      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

      if self._layer_norm:
          #print("using layer norm")
          i = self._norm(i, "input")
          j = self._norm(j, "transform")
          f = self._norm(f, "forget")
          o = self._norm(o, "output")

      if self._use_peepholes:
        with tf.variable_scope("peepholes", initializer=self._kernel_initializer):
          w_f_diag = tf.get_variable("w_f_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_i_diag = tf.get_variable("w_i_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_o_diag = tf.get_variable("w_o_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

      if self._use_peepholes:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c) 
                + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)) 
      else:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias) 
                + tf.nn.sigmoid(i) * self._activation(j))
        # new_c = (c * tf.nn.sigmoid(f) 
        #         + tf.nn.sigmoid(i) * self._activation(j))        

      if self._layer_norm:
          new_c = self._norm(new_c, "state")

      if self._use_peepholes:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * c) 
      else:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state

class ConvLSTMPPCell(ConvRNNCell):
  """Conv LSTM recurrent network cell, that uses an initialization to preserve performance.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               bias_init=10.0,
               rescale = 4.0,
               use_peepholes=False,
               state_is_tuple=False, 
               activation=tf.nn.tanh):
    """Initialize the Conv LSTM PP (Performance Preserving) cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      bias_init: magnitude of initial bias values (sign will be set internally for each gate)
      rescale: the range the input will be in by magnitude so it is close to small
      (should be at most 2.0) so that |input| <= 0.5 for initialization conditions
      to be effective
      use_peepholes: bool, set True to enable peephole connections
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._bias_init = bias_init
    self._rescale = rescale

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvLSTMPPCell"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

      r_val = tf.get_variable(name="rescale", shape=[1], dtype=inputs.dtype, initializer=tf.constant_initializer(self._rescale))
      r_denom = tf.nn.relu(r_val + 1e-8) * (tf.reduce_max(tf.abs(inputs)) + 1e-8) 
#      r_denom = self._rescale * (tf.reduce_max(tf.abs(inputs)) + 1e-8)
      r_inputs = tf.truediv(inputs, r_denom)

      with tf.variable_scope("oi"):
        concat_oi = _conv_linear([r_inputs, h], \
              self.filter_size, self._out_depth * 2, True, tf.constant_initializer(self._bias_init), tf.zeros_initializer())
        o, i = tf.split(axis=3, num_or_size_splits=2, value=concat_oi)

      with tf.variable_scope("f"):
        forget_bias_init = -1.0*self._bias_init
        f = _conv_linear([r_inputs, h], \
              self.filter_size, self._out_depth, True, tf.constant_initializer(forget_bias_init), tf.zeros_initializer())

      with tf.variable_scope("j_u"):
        j_u = _conv_linear([h], \
              self.filter_size, self._out_depth, True, tf.zeros_initializer(), tf.zeros_initializer())

      with tf.variable_scope("j_w"):
        # identity initialization
        w_init = np.zeros((self.filter_size[0], self.filter_size[1], r_inputs.shape[3], self._out_depth))
        mid_0 = self.filter_size[0] // 2
        mid_1 = self.filter_size[1] // 2
        i_trans = np.eye(r_inputs.shape[3], self._out_depth).reshape(1, 1, r_inputs.shape[3], self._out_depth)
        w_init[mid_0, mid_1, :, :] = w_init[mid_0, mid_1, :, :] + i_trans
        j_w = _conv_linear([r_inputs], \
              self.filter_size, self._out_depth, False, tf.zeros_initializer(), tf.constant_initializer(w_init))

      j = tf.add_n([j_u, j_w], name = "j")
      
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      # we zero initialize the peepholes here
      if self._use_peepholes:
        with tf.variable_scope("peepholes", initializer=tf.zeros_initializer()):
          w_f_diag = tf.get_variable("w_f_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_i_diag = tf.get_variable("w_i_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_o_diag = tf.get_variable("w_o_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

      if self._use_peepholes:
        new_c = (c * tf.nn.sigmoid(f + w_f_diag * c) 
                + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)) 
      else:
        new_c = (c * tf.nn.sigmoid(f) 
                + tf.nn.sigmoid(i) * self._activation(j))

      if self._use_peepholes:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * c) 
      else:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      new_h = r_denom * new_h # rescale back
      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state

class ConvLSTMPPAddCell(ConvRNNCell):
  """Conv LSTM recurrent network cell, that is performance preserving where the
  output gate is initialized to 0 and the output is added to the original input.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               output_bias_init=10.0,
               interpolate=True,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=False, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv LSTM PP (Performance Preserving) Add cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      output_bias_init: magnitude of initial output gate bias value (sign will be set internally)
      interpolate: to interpolate between the input and lstm output via
      the output gate rather than direct addition
      use_peepholes: bool, set True to enable peephole connections
      activation: Activation function of the inner states.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._forget_bias = forget_bias
    self._output_bias_init = output_bias_init
    self._interpolate = interpolate
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvLSTMPPAddCell"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

      concat = _conv_linear([inputs, h], \
              self.filter_size, self._out_depth * 3, True, self._bias_initializer, self._kernel_initializer)
      
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f = tf.split(axis=3, num_or_size_splits=3, value=concat)

      # output gate initialized to 0
      with tf.variable_scope("o"):
        output_init = -1.0*self._output_bias_init
        o = _conv_linear([inputs, h], \
              self.filter_size, self._out_depth, True, tf.constant_initializer(output_init), tf.zeros_initializer())

      if self._use_peepholes:
        with tf.variable_scope("peepholes"):
          w_f_diag = tf.get_variable("w_f_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype, initializer=self._kernel_initializer)

          w_i_diag = tf.get_variable("w_i_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype, initializer=self._kernel_initializer)

          w_o_diag = tf.get_variable("w_o_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype, initializer=tf.zeros_initializer()) # we want output gate to be initially 0

      if self._use_peepholes:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c) 
                + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)) 
      else:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias) 
                + tf.nn.sigmoid(i) * self._activation(j))

      if self._use_peepholes:
        o_gate = tf.nn.sigmoid(o + w_o_diag * c)
        new_h = self._activation(new_c) * o_gate
      else:
        o_gate = tf.nn.sigmoid(o)
        new_h = self._activation(new_c) * o_gate

      if self._interpolate:
        g_inputs = (1 - o_gate)*inputs
        new_h = tf.add_n([g_inputs, new_h])
      else:
        new_h = tf.add_n([inputs, new_h]) # add original input to output

      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state


class ConvDepthLSTMCell(ConvRNNCell):
  """Conv LSTM recurrent network cell, with feedback gating.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=False, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Depth LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      use_peepholes: bool, set True to enable peephole connections
      activation: Activation function of the inner states.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvDepthLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

      ff_inp = inputs['ff']
      nonff_inp = inputs['non_ff']
      concat = _conv_linear([ff_inp, h], \
              self.filter_size, self._out_depth * 4, True, self._bias_initializer, self._kernel_initializer)
      
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

      if nonff_inp is not None:
        # feedforward component of depth gate
        with tf.variable_scope("depth_gate_ff", initializer=self._kernel_initializer):
          d_ff = _conv_linear([ff_inp, h], \
              self.filter_size, self._out_depth, True, self._bias_initializer, self._kernel_initializer)

        collect_inputs = [d_ff]
        # learn a tranpose convolution on the non ff inputs and add them together with 
        # the convolutions on the feedforward input and previous hidden state
        # for the depth gate 
        # (NOTE: if we have more than one feedback, we should adapt this
        # to have a depth gate per feedback, but same formula just different weights)
        for nonff_idx, nonff_elem in enumerate(nonff_inp):
          with tf.variable_scope("depth_gate_nonff_" + str(nonff_idx), initializer=self._kernel_initializer):
            nonff_out = _transpose_conv_linear([nonff_elem], \
                        [ff_inp.get_shape().as_list()[0], self.shape[0], self.shape[1], self._out_depth], \
                        self.filter_size, self._out_depth, False, self._bias_initializer, self._kernel_initializer)
            collect_inputs.append(nonff_out)

        d = tf.add_n(collect_inputs, name='depth_gate')

      if self._use_peepholes:
        with tf.variable_scope("peepholes", initializer=self._kernel_initializer):
          w_f_diag = tf.get_variable("w_f_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_i_diag = tf.get_variable("w_i_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_o_diag = tf.get_variable("w_o_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          if nonff_inp is not None:
            w_d_diag = tf.get_variable("w_d_diag", 
              [self.shape[0], self.shape[1], self._out_depth],
              dtype=c.dtype)

      if nonff_inp is not None:
        collect_comp = [j]
        # for each non ff input, learn a transpose convolution
        for nonff_idx, nonff_elem in enumerate(nonff_inp):
          with tf.variable_scope("depth_comp_" + str(nonff_idx), initializer=self._kernel_initializer):
            mem_d = _transpose_conv_linear([nonff_elem], \
                    [ff_inp.get_shape().as_list()[0], self.shape[0], self.shape[1], self._out_depth], \
                    self.filter_size, self._out_depth, False, self._bias_initializer, self._kernel_initializer)
        
          if self._use_peepholes:
            d_comp = tf.nn.sigmoid(d + w_d_diag * c) * mem_d
          else:
            d_comp = tf.nn.sigmoid(d) * mem_d

          collect_comp.append(d_comp)

        cell_pre = tf.add_n(collect_comp, name='cell_pre')
        candidate_cell = self._activation(cell_pre)
      else:
        candidate_cell = self._activation(j)

      if self._use_peepholes:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c) 
                + tf.nn.sigmoid(i + w_i_diag * c) * candidate_cell) 
      else:
        new_c = (c * tf.nn.sigmoid(f + self._forget_bias) 
                + tf.nn.sigmoid(i) * candidate_cell)

      if self._use_peepholes:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * c) 
      else:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state

class ConvDepthLSTMCellVar2(ConvRNNCell):
  """Conv LSTM recurrent network cell (variant 2), with feedback gating.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=False, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv Depth LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      use_peepholes: bool, set True to enable peephole connections
      activation: Activation function of the inner states.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvDepthLSTMCellVar2"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)

      ff_inp = inputs['ff']
      nonff_inp = inputs['non_ff']
      concat = _conv_linear([ff_inp, h], \
              self.filter_size, self._out_depth * 4, True, self._bias_initializer, self._kernel_initializer)
      
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

      if nonff_inp is not None:
        # feedforward component of depth gate
        with tf.variable_scope("depth_gate_ff", initializer=self._kernel_initializer):
          d_ff = _conv_linear([ff_inp, h], \
              self.filter_size, self._out_depth, True, self._bias_initializer, self._kernel_initializer)

        collect_inputs = [d_ff]
        # learn a tranpose convolution on the non ff inputs and add them together with 
        # the convolutions on the feedforward input and previous hidden state
        # for the depth gate 
        # (NOTE: if we have more than one feedback, we should adapt this
        # to have a depth gate per feedback, but same formula just different weights)
        for nonff_idx, nonff_elem in enumerate(nonff_inp):
          with tf.variable_scope("depth_gate_nonff_" + str(nonff_idx), initializer=self._kernel_initializer):
            nonff_out = _transpose_conv_linear([nonff_elem], \
                        [ff_inp.get_shape().as_list()[0], self.shape[0], self.shape[1], self._out_depth], \
                        self.filter_size, self._out_depth, False, self._bias_initializer, self._kernel_initializer)
            collect_inputs.append(nonff_out)

        d = tf.add_n(collect_inputs, name='depth_gate')

      if self._use_peepholes:
        with tf.variable_scope("peepholes", initializer=self._kernel_initializer):
          w_f_diag = tf.get_variable("w_f_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_i_diag = tf.get_variable("w_i_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          w_o_diag = tf.get_variable("w_o_diag", 
            [self.shape[0], self.shape[1], self._out_depth],
            dtype=c.dtype)

          if nonff_inp is not None:
            w_d_diag = tf.get_variable("w_d_diag", 
              [self.shape[0], self.shape[1], self._out_depth],
              dtype=c.dtype)

      if nonff_inp is not None:
        collect_comp = []
        # for each non ff input, learn a transpose convolution (with a bias)
        for nonff_idx, nonff_elem in enumerate(nonff_inp):
          with tf.variable_scope("depth_comp_" + str(nonff_idx), initializer=self._kernel_initializer):
            mem_d = _transpose_conv_linear([nonff_elem], \
                    [ff_inp.get_shape().as_list()[0], self.shape[0], self.shape[1], self._out_depth], \
                    self.filter_size, self._out_depth, True, self._bias_initializer, self._kernel_initializer)
        
          if self._use_peepholes:
            d_comp = tf.nn.sigmoid(d + w_d_diag * c) * mem_d
          else:
            d_comp = tf.nn.sigmoid(d) * mem_d

          collect_comp.append(d_comp)

        total_comp = tf.add_n(collect_comp, name='total_comp')

        if self._use_peepholes:
          new_c = (total_comp 
                  + c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c) 
                  + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)) 
        else:
          new_c = (total_comp
                  + c * tf.nn.sigmoid(f + self._forget_bias) 
                  + tf.nn.sigmoid(i) * self._activation(j))
      else:
        if self._use_peepholes:
          new_c = (c * tf.nn.sigmoid(f + self._forget_bias + w_f_diag * c) 
                  + tf.nn.sigmoid(i + w_i_diag * c) * self._activation(j)) 
        else:
          new_c = (c * tf.nn.sigmoid(f + self._forget_bias) 
                  + tf.nn.sigmoid(i) * self._activation(j))

      if self._use_peepholes:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o + w_o_diag * c) 
      else:
        new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state

class tnn_ConvBasicCell(ConvRNNCell):

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

        self.conv_cell = ConvBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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
        


class tnn_ConvNormBasicCell(ConvRNNCell):

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

        self.conv_cell = ConvNormBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'], memory[1]['layer_norm'], memory[1]['kernel_regularizer'], memory[1]['bias_regularizer'])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvLinearBasicCell(ConvRNNCell):

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

        self.conv_cell = ConvLinearBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvLargeBasicCell(ConvRNNCell):

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

        self.conv_cell = ConvLargeBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['state_depth'], memory[1]['out_depth'])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
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


class tnn_StackedConvBasicCell(ConvRNNCell):

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

        self.conv_cell1 = ConvBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])
        self.conv_cell2 = ConvBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])
        self.conv_cell3 = ConvBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])
        self.conv_cell4 = ConvBasicCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                bs = output.get_shape().as_list()[0]
                state = tf.concat([self.conv_cell1.zero_state(bs, dtype=self.dtype_tmp), self.conv_cell2.zero_state(bs, dtype=self.dtype_tmp), self.conv_cell3.zero_state(bs, dtype=self.dtype_tmp), self.conv_cell4.zero_state(bs, dtype=self.dtype_tmp)], axis=-1)

            state1, state2, state3, state4 = tf.split(value=state, axis=3, num_or_size_splits=4)                
            with tf.variable_scope("cell1", reuse=self._reuse):
              output1, state1 = self.conv_cell1(output, state1)
            with tf.variable_scope("cell2", reuse=self._reuse):
              output2, state2 = self.conv_cell2(output1, state2)
            with tf.variable_scope("cell3", reuse=self._reuse):
              output3, state3 = self.conv_cell3(output2, state3)
            with tf.variable_scope("cell4", reuse=self._reuse):
              output4, state4 = self.conv_cell4(output3, state4)
            state = tf.concat([state1, state2, state3, state4], axis=-1)
            self.state = tf.identity(state, name='state')
            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output4 = function(output4, inputs, **kwargs)
                    else:
                       output4 = function(output4, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output4, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
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

class tnn_ConvGRUCell(ConvRNNCell):

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

        self.conv_cell = ConvGRUCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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

class tnn_ConvLSTMCell(ConvRNNCell):

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

        self.conv_cell = ConvLSTMCell(**self.memory[1])

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
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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

class tnn_ConvDepthLSTMCell(ConvRNNCell):

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

        self.conv_cell = ConvDepthLSTMCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output['ff'] = function(output['ff'], inputs, **kwargs) # component_conv needs to know the inputs
                    else:
                       output['ff'] = function(output['ff'], **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output['ff'].get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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


class tnn_ConvDepthLSTMCellVar2(ConvRNNCell):

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

        self.conv_cell = ConvDepthLSTMCellVar2(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

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
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, self.name_tmp, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output['ff'] = function(output['ff'], inputs, **kwargs) # component_conv needs to know the inputs
                    else:
                       output['ff'] = function(output['ff'], **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output['ff'].get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype_tmp)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output_tmp = tf.identity(tf.cast(output, self.dtype_tmp), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_tmp_shape = self.output_tmp.shape
        return self.output_tmp, state

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

def _conv_linear(args, filter_size, out_depth, bias, bias_initializer=None, kernel_initializer=None, bias_regularizer=None, kernel_regularizer=None):
  """convolution:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    out_depth: int, number of features.
    bias: boolean as to whether to have a bias.
    bias_initializer: starting value to initialize the bias.
    kernel_initializer: starting value to initialize the kernel.
  Returns:
    A 4D Tensor with shape [batch h w out_depth]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]

  if kernel_regularizer is None:
    kernel_regularizer = 0.
  if bias_regularizer is None:
    bias_regularizer = 0.
  if kernel_initializer is None:
    kernel_initializer = tf.contrib.layers.xavier_initializer()
  if bias_initializer is None:
    bias_initializer = tf.contrib.layers.xavier_initializer()

  # Now the computation.
  kernel = tf.get_variable(
      "weights", [filter_size[0], filter_size[1], total_arg_size_depth, out_depth], dtype=dtype, initializer=kernel_initializer, regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer))
  if len(args) == 1:
    res = tf.nn.conv2d(args[0], kernel, strides=[1, 1, 1, 1], padding='SAME')
  else:
    res = tf.nn.conv2d(tf.concat(axis=3, values=args), kernel, strides=[1, 1, 1, 1], padding='SAME')
  if not bias:
    return res
  if bias_initializer is None:
    bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
  bias_term = tf.get_variable(
      "bias", [out_depth],
      dtype=dtype,
      initializer=bias_initializer,
      regularizer=tf.contrib.layers.l2_regularizer(bias_regularizer))
  return res + bias_term

def _transpose_conv_linear(args, out_shape, filter_size, out_depth, bias, bias_initializer=None, kernel_initializer=None):
  """transpose convolution for dealing with feedbacks:
  Args:
    args: a 4D Tensor or a list of 4D, batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    out_depth: int, number of features.
    bias: boolean as to whether to have a bias.
    bias_initializer: starting value to initialize the bias.
    kernel_initializer: starting value to initialize the kernel.
  Returns:
    A 4D Tensor with shape [batch h w out_depth]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 4:
      raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
    if not shape[3]:
      raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[3]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  kernel = tf.get_variable(
      "weights", [filter_size[0], filter_size[1], out_depth, total_arg_size_depth], dtype=dtype, initializer=kernel_initializer)
  if len(args) == 1:
    new_inp = args[0]
    stride_0 = out_shape[1] // new_inp.get_shape().as_list()[1]
    stride_1 = out_shape[2] // new_inp.get_shape().as_list()[2]
    res = tf.nn.conv2d_transpose(new_inp, kernel, output_shape=out_shape, strides=[1, stride_0, stride_1, 1], padding='VALID')
  else:
    new_inp = tf.concat(axis=3, values=args)
    stride_0 = out_shape[1] // new_inp.get_shape().as_list()[1]
    stride_1 = out_shape[2] // new_inp.get_shape().as_list()[2]
    res = tf.nn.conv2d_transpose(new_inp, kernel, output_shape=out_shape, strides=[1, stride_0, stride_1, 1], padding='VALID')
  if not bias:
    return res
  if bias_initializer is None:
    bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
  bias_term = tf.get_variable(
      "bias", [out_depth],
      dtype=dtype,
      initializer=bias_initializer)
  return res + bias_term
