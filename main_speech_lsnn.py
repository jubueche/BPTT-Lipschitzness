from jax import config
config.FLAGS.jax_log_compiles=True

import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from TensorCommands import input_data
from six.moves import xrange
from RNN_Jax import RNN
from jax import random
import jax.numpy as jnp
from loss_jax import loss_normal, compute_gradient_and_update, attack_network
from jax.experimental import optimizers
import ujson as json
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from random import randint
import time
import string
import math
import sqlite3
from architectures import speech_lsnn as arch
from architectures import log

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

def get_lr_schedule(iteration, lrs):

    ts = onp.arange(1,sum(iteration),1)
    lr_sched = onp.zeros((len(ts),))
    for i in range(1,len(iteration)):
        iteration[i] += iteration[i-1]
    def get_lr(t):
        if(t < iteration[0]):
            return lrs[0]
        for i in range(1,len(iteration)):
            if(t < iteration[i] and t >= iteration[i-1]):
                return lrs[i]

    for idx,t in enumerate(ts):
        lr_sched[idx] = get_lr(t)
    lr_sched = jnp.array(lr_sched)
    def lr_schedule(t):
        return lr_sched[t]
    
    return lr_schedule

if __name__ == '__main__':

    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.json")

    def _next_power_of_two(x):
        return 1 if x == 0 else 2**(int(x) - 1).bit_length()

    FLAGS.desired_samples = int(FLAGS.sample_rate * FLAGS.clip_duration_ms / 1000)
    FLAGS.window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
    FLAGS.window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
    FLAGS.length_minus_window = (FLAGS.desired_samples - FLAGS.window_size_samples)
    if FLAGS.length_minus_window < 0:
        spectrogram_length = 0
    else:
        FLAGS.spectrogram_length = 1 + int(FLAGS.length_minus_window / FLAGS.window_stride_samples)
    if FLAGS.preprocess == 'average':
        fft_bin_count = 1 + (_next_power_of_two(FLAGS.window_size_samples) / 2)
        FLAGS.average_window_width = int(math.floor(fft_bin_count / FLAGS.feature_bin_count))
        FLAGS.fingerprint_width = int(math.ceil(fft_bin_count / FLAGS.average_window_width))
    elif FLAGS.preprocess in ['mfcc', 'fbank']:
        FLAGS.average_window_width = -1
        FLAGS.fingerprint_width = FLAGS.feature_bin_count
    elif FLAGS.preprocess == 'micro':
        FLAGS.average_window_width = -1
        FLAGS.fingerprint_width = FLAGS.feature_bin_count
    else:
        raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                        ' "average", or "micro")' % (FLAGS.preprocess))
    FLAGS.fingerprint_size = FLAGS.fingerprint_width * FLAGS.spectrogram_length

    audio_processor = input_data.AudioProcessor(
        data_url=FLAGS.data_url, data_dir=FLAGS.data_dir,
        silence_percentage=FLAGS.silence_percentage, unknown_percentage=FLAGS.unknown_percentage,
        wanted_words=FLAGS.wanted_words.split(','), validation_percentage=FLAGS.validation_percentage,
        testing_percentage=FLAGS.testing_percentage, 
        n_thr_spikes=FLAGS.n_thr_spikes, n_repeat=FLAGS.in_repeat, seed=FLAGS.seed
    )

    FLAGS.label_count = len(input_data.prepare_words_list(FLAGS.wanted_words.split(',')))

    FLAGS.time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)


    epochs_list = list(map(int, FLAGS.n_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(epochs_list) != len(learning_rates_list):
        raise Exception(
            '--n_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(epochs_list),
                                                        len(learning_rates_list)))
    
    
    steps_list = [math.ceil(epochs * audio_processor.set_size("training")/FLAGS.batch_size) for epochs in epochs_list]
    

    n_thr_spikes = max(1, FLAGS.n_thr_spikes)

    # - Define trainable variables
    rng_key = random.PRNGKey(FLAGS.seed)
    _, *sks = random.split(rng_key, 5)
    W_in = onp.array(random.truncated_normal(sks[1],-2,2,(FLAGS.fingerprint_width, FLAGS.n_hidden))* (onp.sqrt(2/(FLAGS.fingerprint_width + FLAGS.n_hidden)) / .87962566103423978))
    W_rec = onp.array(random.truncated_normal(sks[2],-2,2,(FLAGS.n_hidden, FLAGS.n_hidden))* (onp.sqrt(1/(FLAGS.n_hidden)) / .87962566103423978))
    onp.fill_diagonal(W_rec, 0.)
    W_out = onp.array(random.truncated_normal(sks[3],-2,2,(FLAGS.n_hidden, FLAGS.label_count))*0.01)
    b_out = onp.zeros((FLAGS.label_count,))

    # - Create the model
    rnn = RNN(vars(FLAGS))

    init_params = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    color_range = onp.linspace(0,1,onp.sum(iteration))
    
    if(FLAGS.optimizer == "adam"):
        opt_init, opt_update, get_params = optimizers.adam(get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    elif(FLAGS.optimizer == "sgd"):
        opt_init, opt_update, get_params = optimizers.sgd(get_lr_schedule(iteration,lrs))
    else:
        print("Invalid optimizer")
        sys.exit(0)
    opt_state = opt_init(init_params)

    best_val_acc = 0.0
    for i in range(sum(iteration)):
        # - Get training data
        train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, vars(FLAGS), FLAGS.background_frequency,FLAGS.background_volume, FLAGS.time_shift_samples, 'training')
        X = train_fingerprints.numpy()
        y = train_ground_truth.numpy()

        opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, rnn, FLAGS, rnn._rng_key)
        rnn._rng_key, _ = random.split(rnn._rng_key)

        if((i+1) % 10 == 0):
            params = get_params(opt_state)
            logits, spikes = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **params)
            avg_firing = jnp.mean(spikes, axis=1)
            loss = loss_normal(y, logits, avg_firing, FLAGS.reg)
            lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS, rnn._rng_key)
            rnn._rng_key, _ = random.split(rnn._rng_key)
            lip_loss_over_time = list(onp.array(lip_loss_over_time, dtype=onp.float64))
            training_accuracy = get_batched_accuracy(y, logits)
            attacked_accuracy = get_batched_accuracy(y, logits_theta_star)
            print(f"Loss is {loss} Lipschitzness loss over time {lip_loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_accuracy}",flush=True)
            log(FLAGS.session_id,"training_accuracy",onp.float64(training_accuracy))
            log(FLAGS.session_id,"attacked_training_accuracy",onp.float64(attacked_accuracy))
            log(FLAGS.session_id,"kl_over_time",lip_loss_over_time)


        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = audio_processor.set_size('validation')
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data(FLAGS.batch_size, i, vars(FLAGS), 0.0, 0.0, 0.0, 'validation'))
                X = validation_fingerprints.numpy()
                y = validation_ground_truth.numpy()
                logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS, rnn._rng_key)
                rnn._rng_key, _ = random.split(rnn._rng_key)
                llot.append(lip_loss_over_time)
                predicted_labels = jnp.argmax(logits, axis=1)
                correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
                batched_validation_acc = get_batched_accuracy(y, logits)
                attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
                total_accuracy += (batched_validation_acc * FLAGS.batch_size) / set_size
                attacked_total_accuracy += (attacked_batched_validation_acc * FLAGS.batch_size) / set_size

            # - Logging
            log(FLAGS.session_id,"validation_accuracy",onp.float64(total_accuracy))
            log(FLAGS.session_id,"attacked_validation_accuracies",onp.float64(attacked_total_accuracy))
            mean_llot = onp.mean(onp.asarray(llot), axis=0)
            log(FLAGS.session_id,"validation_kl_over_time",list(onp.array(mean_llot, dtype=onp.float64)))

            # - Save the model
            if(total_accuracy > best_val_acc):
                best_val_acc = total_accuracy
                rnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")

            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")

        
