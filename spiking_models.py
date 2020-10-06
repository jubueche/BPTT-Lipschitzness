import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell, DropoutRNNCellMixin
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.variables import Variable
from tensorflow.python.keras.utils import tf_utils
# from rewiring_tools import weight_sampler

Cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell


def einsum_bi_ijk_to_bjk(a,b):
    batch_size = tf.shape(a)[0]
    shp_a = a.get_shape()
    shp_b = b.get_shape()

    b_ = tf.reshape(b,(int(shp_b[0]), int(shp_b[1]) * int(shp_b[2])))
    ab_ = tf.matmul(a,b_)
    ab = tf.reshape(ab_,(batch_size,int(shp_b[1]),int(shp_b[2])))
    return ab


def tf_roll(buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
        shp = buffer.get_shape()
        l_shp = len(shp)

        # Permute the index to roll over the right index
        perm = np.concatenate([[axis],np.arange(axis),np.arange(start=axis+1,stop=l_shp)])
        buffer = tf.transpose(buffer, perm=perm)

        # Add an element at the end of the buffer if requested, otherwise, add zero
        if new_last_element is None:
            shp = tf.shape(buffer)
            new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
        new_last_element = tf.expand_dims(new_last_element, axis=0)
        new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

        # Revert the index permutation
        inv_perm = np.argsort(perm)
        new_buffer = tf.transpose(new_buffer,perm=inv_perm)

        new_buffer = tf.identity(new_buffer,name='Roll')
        #new_buffer.set_shape(shp)
    return new_buffer


def map_to_named_tuple(S, f):
    state_dict = S._asdict()
    new_state_dict = OrderedDict({})
    for k, v in state_dict.items():
        new_state_dict[k] = f(v)

    new_named_tuple = S.__class__(**new_state_dict)
    return new_named_tuple

def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            if np.shape(v) == ():
                v = [v]
            shape = np.concatenate([[batch_size], v])
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)

        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def placeholder_container_from_example(state_example, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = state_example._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            placeholder_dict[k] = tf.placeholder(shape=v.shape, dtype=v.dtype, name=k)

        placeholder_tuple = state_example.__class__(**placeholder_dict)
        return placeholder_tuple

def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v})
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update



@function.Defun()
def SpikeFunctionGrad(v_scaled, dampening_factor, grad):
    dE_dz = grad
    dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
    dz_dv_scaled *= dampening_factor

    dE_dv_scaled = dE_dz * dz_dv_scaled

    return [dE_dv_scaled,
            tf.zeros_like(dampening_factor)]

@function.Defun(grad_func=SpikeFunctionGrad)
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)
    return tf.identity(z_, name="SpikeFunction")

def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = tf.equal(d,kd)
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer
def exp_convolve(tensor, decay, init=None):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        if init is not None:
            assert str(init.get_shape()) == str(tensor_time_major[0].get_shape())  # must be batch x neurons
            initializer = init
        else:
            initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.

    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in [Variable, Tensor]:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.int64, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute of key {} and value {} has type {}, recoding it as string.'.format(k, v, type(v)))
            dict_to_save[k] = str(v)

    return dict_to_save

FastALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',
    'r',
))

class KerasALIF(DropoutRNNCellMixin, Layer):
  def __init__(self,
               n_in, units, tau=20, thr=0.01,
               dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
               tau_adaptation=200., beta=1.6,
               rewiring_connectivity=-1, dampening_factor=0.3,
               in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
               add_current=0., thr_min=0.005,
               input_initializer='glorot_normal',  # FIXME: try glorot_uniform
               recurrent_initializer='glorot_normal',
               input_regularizer=None,
               recurrent_regularizer=None,
               input_constraint=None,
               recurrent_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               eprop_sym=False,
               **kwargs):
    super(KerasALIF, self).__init__(**kwargs)
    self.units = units

    # if np.isscalar(tau): tau = tf.ones(units, dtype=dtype) * np.mean(tau)
    # if np.isscalar(thr): thr = tf.ones(units, dtype=dtype) * np.mean(thr)
    # tau = tf.cast(tau, dtype=dtype)
    dt = tf.cast(dt, dtype=dtype)

    # thr = tf.compat.v1.identity(thr, name="thr")
    # tau = tf.compat.v1.identity(tau, name="tau")

    if np.isscalar(tau): tau = np.ones(units) * np.mean(tau)
    if np.isscalar(thr): thr = np.ones(units) * np.mean(thr)

    # - Create variable from numpy array
    tau = tf.compat.v1.Variable(initial_value=tau, name="tau", dtype=dtype, trainable=False)
    thr = tf.compat.v1.Variable(initial_value=thr, name="thr", dtype=dtype, trainable=False)

    self.dampening_factor = dampening_factor
    self.eprop_sym = eprop_sym

    # Parameters
    self.n_delay = n_delay
    self.n_refractory = n_refractory

    self.dt = dt
    self.n_in = n_in
    self.data_type = dtype

    # self._num_units = self.n_rec

    self.tau = tau
    self._decay = tf.exp(-dt / tau)
    self.thr = thr

    self.injected_noise_current = injected_noise_current

    self.rewiring_connectivity = rewiring_connectivity
    self.in_neuron_sign = in_neuron_sign
    self.rec_neuron_sign = rec_neuron_sign

    if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
    if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

    self.tau_adaptation = tau_adaptation
    self.beta = beta
    self.min_beta = np.min(beta)
    self.elifs = beta < 0
    self.decay_b = tf.exp(-dt / tau_adaptation)
    self.add_current = add_current
    self.thr_min = thr_min

    self.input_initializer = initializers.get(input_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)

    self.input_regularizer = regularizers.get(input_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

    self.input_constraint = constraints.get(input_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # self.state_size = FastALIFStateTuple(v=self.units, z=self.units, b=self.units, r=self.units)
    self.state_size = data_structures.NoDependency([self.units, self.units, self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape[-1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                         str(input_shape))
    # _check_supported_dtypes(self.dtype)
    n_in = input_shape[-1]
    n_rec = self.units
    self.W_in = self.add_weight(
      name="InputWeight", shape=[n_in, n_rec],
      initializer=self.input_initializer,
      regularizer=self.input_regularizer,
      constraint=self.input_constraint,
    )
    self.W_rec = self.add_weight(
      name="RecurrentWeight", shape=[n_rec, n_rec],
      initializer=self.recurrent_initializer,
      regularizer=self.recurrent_regularizer,
      constraint=self.recurrent_constraint,
    )

    # Disconnect autotapse
    recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
    self.W_rec = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.W_rec), self.W_rec)

    self.built = True

  def compute_z(self, v, adaptive_thr):
    v_scaled = (v - adaptive_thr) / adaptive_thr
    z = SpikeFunction(v_scaled, self.dampening_factor)
    z = z * 1 / self.dt
    return z

  def call(self, inputs, states, training=None):

    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(states[1], training)
    if 0 < self.dropout < 1.:
        inputs = inputs * dp_mask
    if 0 < self.recurrent_dropout < 1.:
        state = FastALIFStateTuple(v=states[0], z=states[1] * rec_dp_mask, b=states[2], r=states[3])
    else:
        state = FastALIFStateTuple(v=states[0], z=states[1], b=states[2], r=states[3])

    new_b = self.decay_b * state.b + (np.ones(self.units) - self.decay_b) * state.z
    thr = self.thr + new_b * self.beta

    if self.eprop_sym:
      z = tf.stop_gradient(state.z)
    else:
      z = state.z

    i_in = tf.matmul(inputs, self.W_in)
    i_rec = tf.matmul(z, self.W_rec)
    i_t = i_in + i_rec + self.add_current

    I_reset = z * thr * self.dt

    new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset

    # Spike generation
    is_refractory = tf.greater(state.r, .1)
    zeros_like_spikes = tf.zeros_like(z)
    new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))

    return new_z, [new_v, new_z, new_b, new_r]

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    base_config = super(KerasALIF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


DelayALIFStateTuple = namedtuple('DelayALIFStateTuple', (
    'z',
    'v',
    'b',
    'r',
    'i_future_buffer',
))


class KerasDelayALIF(DropoutRNNCellMixin, Layer):
  def __init__(self,
               n_in, units, tau=20, thr=0.01,
               dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
               tau_adaptation=200., beta=1.6,
               rewiring_connectivity=-1, dampening_factor=0.3,
               in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.,
               add_current=0., thr_min=0.005,
               input_initializer='glorot_normal',  # FIXME: try glorot_uniform
               recurrent_initializer='glorot_normal',
               input_regularizer=None,
               recurrent_regularizer=None,
               input_constraint=None,
               recurrent_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               eprop_sym=False,
               **kwargs):
    super(KerasDelayALIF, self).__init__(**kwargs)
    self.units = units

    if np.isscalar(tau): tau = tf.ones(units, dtype=dtype) * np.mean(tau)
    if np.isscalar(thr): thr = tf.ones(units, dtype=dtype) * np.mean(thr)
    tau = tf.cast(tau, dtype=dtype)
    dt = tf.cast(dt, dtype=dtype)

    self.dampening_factor = dampening_factor
    self.eprop_sym = eprop_sym

    # Parameters
    self.n_delay = n_delay
    self.n_refractory = n_refractory

    self.dt = dt
    self.n_in = n_in
    self.data_type = dtype

    # self._num_units = self.n_rec

    self.tau = tau
    self._decay = tf.exp(-dt / tau)
    self.thr = thr

    self.injected_noise_current = injected_noise_current

    self.rewiring_connectivity = rewiring_connectivity
    self.in_neuron_sign = in_neuron_sign
    self.rec_neuron_sign = rec_neuron_sign

    if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
    if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

    self.tau_adaptation = tau_adaptation
    self.beta = beta
    self.min_beta = np.min(beta)
    self.elifs = beta < 0
    self.decay_b = tf.exp(-dt / tau_adaptation)
    self.add_current = add_current
    self.thr_min = thr_min

    self.input_initializer = initializers.get(input_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)

    self.input_regularizer = regularizers.get(input_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

    self.input_constraint = constraints.get(input_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # self.state_size = DelayALIFStateTuple(z=self.units, v=self.units, b=self.units, r=self.units,
    #                                       i_future_buffer=(self.units, self.n_delay))
    self.state_size = data_structures.NoDependency(
      [self.units, self.units, self.units, self.units, self.units * self.n_delay])
    self.output_size = self.units

  def weight_matrix_with_delay_dimension(self, w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
      w_d_list = []
      for kd in range(n_delay):
        mask = tf.equal(d, kd)
        w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
        w_d_list.append(w_d)

      delay_axis = len(d.shape)
      WD = tf.stack(w_d_list, axis=delay_axis)

    return WD

  def tf_roll(self, buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
      shp = buffer.get_shape()
      l_shp = len(shp)

      # Permute the index to roll over the right index
      perm = np.concatenate([[axis], np.arange(axis), np.arange(start=axis + 1, stop=l_shp)])
      buffer = tf.transpose(buffer, perm=perm)

      # Add an element at the end of the buffer if requested, otherwise, add zero
      if new_last_element is None:
        shp = tf.shape(buffer)
        new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
      new_last_element = tf.expand_dims(new_last_element, axis=0)
      new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

      # Revert the index permutation
      inv_perm = np.argsort(perm)
      new_buffer = tf.transpose(new_buffer, perm=inv_perm)

      new_buffer = tf.identity(new_buffer, name='Roll')
      # new_buffer.set_shape(shp)
    return new_buffer

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape[-1] is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                         str(input_shape))
    # _check_supported_dtypes(self.dtype)
    n_in = input_shape[-1]
    n_rec = self.units

    # self.w_in_init = rd.randn(n_in, n_rec) / np.sqrt(n_in)
    # self.w_in_var = tf.Variable(self.w_in_init, dtype=self.data_type, name="InputWeight")
    self.w_in_var = self.add_weight(
      name="InputWeight", shape=[n_in, n_rec],
      initializer=self.input_initializer,
      regularizer=self.input_regularizer,
      constraint=self.input_constraint,
    )
    self.w_in_val = self.w_in_var
    self.w_in_delay = tf.Variable(rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec),
                                  dtype=tf.int32, name="InDelays", trainable=False)
    self.W_in = self.weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

    # self.w_rec_var = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=self.data_type, name='RecurrentWeight')
    self.w_rec_var = self.add_weight(
      name="RecurrentWeight", shape=[n_rec, n_rec],
      initializer=self.recurrent_initializer,
      regularizer=self.recurrent_regularizer,
      constraint=self.recurrent_constraint,
    )
    self.w_rec_val = self.w_rec_var
    recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))
    self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val), self.w_rec_val)
    self.w_rec_delay = tf.Variable(rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec), dtype=tf.int32,
                                   name="RecDelays", trainable=False)
    self.W_rec = self.weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    self.built = True

  def compute_z(self, v, adaptive_thr):
    v_scaled = (v - adaptive_thr) / adaptive_thr
    z = SpikeFunction(v_scaled, self.dampening_factor)
    z = z * 1 / self.dt
    return z

  def call(self, inputs, states, training=None):

    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(states[1], training)
    if 0 < self.dropout < 1.:
        inputs = inputs * dp_mask
    if 0 < self.recurrent_dropout < 1.:
        state = DelayALIFStateTuple(v=states[0], z=states[1] * rec_dp_mask, b=states[2], r=states[3],
                                    i_future_buffer=states[4])
    else:
        state = DelayALIFStateTuple(v=states[0], z=states[1], b=states[2], r=states[3], i_future_buffer=states[4])

    new_b = self.decay_b * state.b + (np.ones(self.units) - self.decay_b) * state.z
    thr = self.thr + new_b * self.beta

    if self.eprop_sym:
      z = tf.stop_gradient(state.z)
    else:
      z = state.z

    i_in = tf.einsum('bi,ijk->bjk', inputs, self.W_in)
    i_rec = tf.einsum('bi,ijk->bjk', z, self.W_rec)
    old_i_future_buffer = tf.reshape(state.i_future_buffer, shape=[-1, self.units, self.n_delay])
    i_future_buffer = old_i_future_buffer + i_in + i_rec
    i_t = i_future_buffer[:, :, 0] + self.add_current

    I_reset = z * thr * self.dt

    new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset

    # Spike generation
    is_refractory = tf.greater(state.r, .1)
    zeros_like_spikes = tf.zeros_like(z)
    new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
    new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                             0., float(self.n_refractory))
    new_i_future_buffer = self.tf_roll(i_future_buffer, axis=2)

    return new_z, [new_v, new_z, new_b, new_r, tf.reshape(new_i_future_buffer, shape=[-1, self.units * self.n_delay])]

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    base_config = super(KerasALIF, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    n_rec = self.units

    v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
    r0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

    i_buff0 = tf.zeros(shape=(batch_size, n_rec * self.n_delay), dtype=dtype)

    # return DelayALIFStateTuple(
    #   z=z0,
    #   v=v0,
    #   b=b0,
    #   r=r0,
    #   i_future_buffer=i_buff0
    # )
    return [v0, z0, b0, r0, i_buff0]
