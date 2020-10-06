# - Implementation of custom RNN with one hidden layer
import tensorflow as tf
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import function
from collections import namedtuple
import numpy as np
from utils import get_parser, prepare_model_settings
import sys
import ujson as json
import os
from datetime import datetime
import input_data_eager as input_data

DEBUG = False

# - Helper functions
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
    z_ = tf.math.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)
    return tf.identity(z_, name="SpikeFunction")

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
        self.tau_adaptation = params["tau_adaptation"]
        self.beta = params["beta"]
        self.min_beta = np.min(params["beta"])
        self.decay_b = tf.exp(-1*self.dt / params["tau_adaptation"])
        self.thr_min = params["thr_min"]
        self.model_settings = params

    def compute_z(self, v, adaptive_thr):
        v_scaled = (v - adaptive_thr) / adaptive_thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = tf.ensure_shape(z, v_scaled.shape)
        z = z * 1 / self.dt
        return z

    @tf.function(autograph=not DEBUG)
    def call(self, fingerprint_input, W_in, W_rec, W_out, b_out):
        input_frequency_size = self.model_settings['fingerprint_width']
        input_channels = max(1, 2*self.model_settings['n_thr_spikes'] - 1)
        input_time_size = self.model_settings['spectrogram_length'] * self.model_settings['in_repeat']
        fingerprint_3d = tf.reshape(fingerprint_input, [-1, input_time_size, input_frequency_size * input_channels])
        
        # - Initial state
        state0 = [tf.zeros((1,self.units)), tf.zeros((1,self.units)), tf.zeros((1,self.units)), tf.zeros((1,self.units))]

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
        def evolve_single(inputs):
            accumulated_state = tf.scan(step, inputs, initializer=state0)
            Z = tf.squeeze(accumulated_state[1]) # -> [T,units]
            if self.model_settings['avg_spikes']:
                Z = tf.reshape(tf.reduce_mean(Z, axis=0), shape=(1,-1))
            out = tf.matmul(Z, W_out) + b_out
            return out # - [BS,Num_labels]

        final_out = tf.squeeze(tf.map_fn(evolve_single, fingerprint_3d)) # -> [BS,T,self.units]
        return final_out


if __name__ == '__main__':

    print(f"Tensorflow version {tf.__version__} Using eager evalation {tf.executing_eagerly()} should be True")

    parser = get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...")
        print(unparsed)
        sys.exit(0)

    model_settings = prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir,
        FLAGS.silence_percentage, FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir,
        FLAGS.n_thr_spikes, FLAGS.in_repeat
    )
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                        len(learning_rates_list)))
    n_thr_spikes = max(1, FLAGS.n_thr_spikes)

    # - Define trainable variables
    d_In = model_settings['fingerprint_width']
    d_Out = model_settings["label_count"]
    W_in = tf.Variable(initial_value=tf.random.normal(shape=(d_In,FLAGS.n_hidden), mean=0.0, stddev= tf.sqrt(2/(d_In + FLAGS.n_hidden))), trainable=True)
    W_rec = tf.Variable(initial_value=tf.linalg.set_diag(tf.random.normal(shape=(FLAGS.n_hidden,FLAGS.n_hidden), mean=0., stddev= tf.sqrt(1/FLAGS.n_hidden)), tf.zeros([FLAGS.n_hidden])), trainable=True)
    W_out = tf.Variable(initial_value=tf.random.normal(shape=(FLAGS.n_hidden,d_Out), mean=0.0, stddev=0.01), trainable=True)
    b_out = tf.Variable(initial_value=tf.zeros(shape=(d_Out,)), trainable=True)

    # - Create the model
    rnn = RNN(model_settings)

    # - Define loss function
    def loss_normal(target_output, logits):
        cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
        # - TODO Need to adapt call method so that it also returns spikes
        # if FLAGS.model_architecture == 'lsnn':
        #     regularization_f0 = 10 / 1000  # 10Hz
        #     loss_reg = tf.reduce_sum(tf.square(average_fr - regularization_f0) * FLAGS.reg)
        #     cross_entropy_mean += loss_reg
        return cross_entropy_mean

    n_iterations = 500
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(n_iterations):
        # - Get some data
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training')

        with tf.GradientTape() as tape:
            logits = rnn.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            loss = loss_normal(train_ground_truth, logits)
        # - Get the gradients
        gradients = tape.gradient(loss, [W_in,W_rec,W_out,b_out])
        optimizer.apply_gradients(zip(gradients,[W_in,W_rec,W_out,b_out]))

        if(i % 10 == 0):
            logits = rnn.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            loss = loss_normal(train_ground_truth, logits)
            print(f"Loss is {loss.numpy()}")


    # - Call the RNN
    out = rnn.call(train_fingerprints, W_in, W_rec, W_out, b_out,)
    print(out.numpy())



    