# - Implementation of custom RNN with one hidden layer
import tensorflow as tf
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import function
from collections import namedtuple
import numpy as np
import sys
import os
from datetime import datetime
import input_data_eager as input_data
import tensorflow_probability as tfp
from six.moves import xrange
import tensorflow.keras.backend as K

import json

print(f"Tensorflow version {tf.__version__}")

DEBUG = False

@tf.custom_gradient # z = f(V,DF) is computed here. Given dE/dz, we can compute dE/dV using the chain rule: dE/dz * dz/dV 
def SpikeFunction(v_scaled):
    z_ = tf.math.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)
    def grad(dE_dz):
        dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
        dz_dv_scaled *= 0.3
        dE_dv_scaled = dE_dz * dz_dv_scaled
        return dE_dv_scaled
    return z_, grad


FastALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',
    'r',
))

class RNN:

    def __init__(self,params):

        self.units = params["n_hidden"]
        self.dt = tf.cast(params["dt"], dtype=tf.float32)
        if np.isscalar(params["tau"]): tau = np.ones(self.units) * np.mean(params["tau"])
        if np.isscalar(params["thr"]): thr = np.ones(self.units) * np.mean(params["thr"])
        # - Create variable from numpy array
        self.tau = tf.Variable(initial_value=tau, name="tau", dtype=tf.float32, trainable=False)
        self.thr = tf.Variable(initial_value=thr, name="thr", dtype=tf.float32, trainable=False)
        self.dampening_factor = params["dampening_factor"]
        # Parameters
        self.n_refractory = params["refr"]
        self.n_in = params['fingerprint_width']
        self.data_type = tf.float32
        self._decay = tf.exp(-1*self.dt / self.tau)
        self.tau_adaptation = params['spectrogram_length'] * params['in_repeat']
        self.beta = tf.ones(shape=(self.units,)) * params["beta"]
        self.min_beta = np.min(params["beta"])
        self.decay_b = tf.exp(-1*self.dt / params["tau_adaptation"])
        self.thr_min = params["thr_min"]
        self.model_settings = params
        self.d_out = params["label_count"]

    @tf.function(autograph=not DEBUG)
    def compute_z(self, v, adaptive_thr):
        v_scaled = (v - adaptive_thr) / adaptive_thr
        z = SpikeFunction(v_scaled)
        z = tf.ensure_shape(z, v_scaled.shape)
        z = z * 1 / self.dt
        return z

    @tf.function(autograph=not DEBUG)
    def call(self, fingerprint_input, W_in, W_rec, W_out, b_out, batch_sized=True):
        input_frequency_size = self.model_settings['fingerprint_width']
        input_channels = max(1, 2*self.model_settings['n_thr_spikes'] - 1)
        input_time_size = self.model_settings['spectrogram_length'] * self.model_settings['in_repeat']
        fingerprint_3d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size * input_channels]) # - [BS,T,In]
        
        recurrent_disconnect_mask = tf.linalg.diag(tf.ones(shape=[self.units], dtype=tf.bool))
        W_rec = tf.where(recurrent_disconnect_mask, tf.zeros_like(W_rec), W_rec)

        # - Initial state
        state0 = [tf.zeros((1,self.units)), tf.zeros((1,self.units)), tf.zeros((1,self.units)), tf.zeros((1,self.units))]
        #state0 = [tf.zeros((fingerprint_3d.shape[0], 1,self.units)), tf.zeros((fingerprint_3d.shape[0], 1,self.units)), tf.zeros((fingerprint_3d.shape[0], 1,self.units)), tf.zeros((fingerprint_3d.shape[0], 1,self.units))]
        state0_b = [tf.zeros((fingerprint_3d.shape[0], self.units)), tf.zeros((fingerprint_3d.shape[0], self.units)), tf.zeros((fingerprint_3d.shape[0], self.units)), tf.zeros((fingerprint_3d.shape[0],self.units))]
        # - Define step function
        @tf.function(autograph=not DEBUG)
        def step(prev_state, input_b):
            input_b = tf.reshape(input_b, shape=[1,input_b.shape[0]])
            state = FastALIFStateTuple(v=prev_state[0], z=prev_state[1], b=prev_state[2], r=prev_state[3])
            new_b = self.decay_b * state.b + (tf.ones(shape=[self.units],dtype=tf.float32) - self.decay_b) * state.z
            thr = self.thr + new_b * self.beta
            z = state.z
            i_in = tf.matmul(input_b, W_in)
            i_rec = tf.matmul(z, W_rec)
            i_t = i_in + i_rec
            I_reset = z * thr * self.dt
            new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset
            # Spike generation
            is_refractory = tf.greater(state.r, .1)
            zeros_like_spikes = tf.zeros_like(z)
            new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
            new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                    0., float(self.n_refractory))
            return [new_v, new_z, new_b, new_r]


        @tf.function(autograph=not DEBUG)
        def keras_step(input_b, prev_state):
            state = FastALIFStateTuple(v=prev_state[0], z=prev_state[1], b=prev_state[2], r=prev_state[3])
            new_b = self.decay_b * state.b + (tf.ones(shape=[self.units],dtype=tf.float32) - self.decay_b) * state.z
            thr = self.thr + new_b * self.beta
            z = state.z
            i_in = tf.matmul(input_b, W_in)
            i_rec = tf.matmul(z, W_rec)
            i_t = i_in + i_rec
            I_reset = z * thr * self.dt
            new_v = self._decay * state.v + (1 - self._decay) * i_t - I_reset
            # Spike generation
            is_refractory = tf.greater(state.r, .1)
            zeros_like_spikes = tf.zeros_like(z, dtype=tf.float32)
            new_z = tf.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, thr))
            new_r = tf.clip_by_value(state.r + self.n_refractory * new_z - 1,
                                    0., float(self.n_refractory))
            return new_z, [new_v, new_z, new_b, new_r]

        @tf.function(autograph=not DEBUG)
        def keras_evolve_single(inputs):
            inputs = tf.reshape(inputs, shape=[1,inputs.shape[0],inputs.shape[1]])
            o = K.rnn(step_function=keras_step, inputs=inputs, initial_states=state0)
            spikes = o[1]
            if(len(spikes.shape) > 1):
                spikes = tf.squeeze(spikes)
            return spikes

        @tf.function(autograph=not DEBUG)
        def evolve_single(inputs):
            accumulated_state = tf.scan(step, inputs, initializer=state0)
            Z = tf.squeeze(accumulated_state[1]) # -> [T,units]
            if self.model_settings['avg_spikes']:
                Z = tf.reshape(tf.reduce_mean(Z, axis=0), shape=(1,-1))
            out = tf.matmul(Z, W_out) + b_out
            return out # - [BS,Num_labels]

        # - Use Keras .rnn()
        if not(batch_sized):
            spikes = tf.map_fn(keras_evolve_single, fingerprint_3d)
            if(len(spikes.shape) > 1):
                spikes = tf.squeeze(spikes)
            if(fingerprint_3d.shape[0]==1):
                spikes = tf.reshape(spikes, shape=[1,spikes.shape[0],spikes.shape[1]])
            out = tf.matmul(tf.reduce_mean(spikes, axis=1), W_out) + b_out

        # - Keras .rnn() ONLY 
        if batch_sized:
            spikes = K.rnn(step_function=keras_step, inputs=fingerprint_3d, initial_states=state0_b)[1]
            if(len(spikes.shape) > 1):
                spikes = tf.squeeze(spikes)
            if(fingerprint_3d.shape[0]==1):
                spikes = tf.reshape(spikes, shape=[1,spikes.shape[0],spikes.shape[1]])
            out = tf.matmul(tf.reduce_mean(spikes, axis=1), W_out) + b_out

        # - Use tf.scan()
        # final_out = tf.squeeze(tf.map_fn(evolve_single, fingerprint_3d))
        
        return out, spikes


    