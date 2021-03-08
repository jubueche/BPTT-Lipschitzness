from jax import config
config.FLAGS.jax_log_compiles=True

import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from ECG.ecg_data_loader import ECGDataLoader
from TensorCommands import input_data
from six.moves import xrange
from RNN_Jax import RNN
from jax import random
import jax.numpy as jnp
from loss_jax import loss_normal, compute_gradient_and_update, attack_network, compute_gradients
from jax.experimental import optimizers
from EntropySGD.entropy_sgd import EntropySGD_Jax
import ujson as json
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from random import randint
import time
import string
import sqlite3
import math
from architectures import ecg_lsnn as arch
from architectures import log
from concurrent.futures import ThreadPoolExecutor
from experiment_utils import get_mismatch_accuracy

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

    FLAGS.l2_weight_decay_params = str(FLAGS.l2_weight_decay_params[1:-1]).split(",")
    FLAGS.l1_weight_decay_params = str(FLAGS.l1_weight_decay_params[1:-1]).split(",")
    FLAGS.contractive_params = str(FLAGS.contractive_params[1:-1]).split(",")
    if(FLAGS.l1_weight_decay_params == ['']):
        FLAGS.l1_weight_decay_params = []
    if(FLAGS.l2_weight_decay_params == ['']):
        FLAGS.l2_weight_decay_params = []
    if(FLAGS.contractive_params == ['']):
        FLAGS.contractive_params = []

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
    
    ecg_processor = ECGDataLoader(path=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    flags_dict = vars(FLAGS)
    epochs_list = list(map(int, FLAGS.n_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(epochs_list) != len(learning_rates_list):
        raise Exception(
            '--n_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(epochs_list),
                                                        len(learning_rates_list)))
    
    steps_list = [math.ceil(epochs * ecg_processor.N_train/FLAGS.batch_size) for epochs in epochs_list]

    # - Define trainable variables
    d_In = ecg_processor.n_channels
    d_Out = ecg_processor.n_labels
    rng_key = random.PRNGKey(FLAGS.seed)
    _, *sks = random.split(rng_key, 5)
    W_in = onp.array(random.truncated_normal(sks[1],-2,2,(d_In, FLAGS.n_hidden))* (onp.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978))
    W_rec = onp.array(random.truncated_normal(sks[2],-2,2,(FLAGS.n_hidden, FLAGS.n_hidden))* (onp.sqrt(1/(FLAGS.n_hidden)) / .87962566103423978))
    onp.fill_diagonal(W_rec, 0.)
    W_out = onp.array(random.truncated_normal(sks[3],-2,2,(FLAGS.n_hidden, d_Out))*0.01)
    b_out = onp.zeros((d_Out,))

    FLAGS.spectrogram_length = ecg_processor.T
    FLAGS.fingerprint_width = ecg_processor.n_channels

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
    elif(FLAGS.optimizer == "esgd"):
        config = dict(momentum=0.9, damp=0.0, nesterov=True, weight_decay=0.0, L=10, eps=1e-4, g0=1e-2, g1=1e-3, langevin_lr=0.1, langevin_beta1=0.75)
        opt_init, opt_update, get_params = EntropySGD_Jax(get_lr_schedule(iteration,lrs), config)
    else:
        print("Invalid optimizer")
        sys.exit(0)
    opt_state = opt_init(init_params)

    best_val_acc = best_mean_mm_val_acc = 0.0

    pp = lambda x : ("%.3f" % x)
    ppl = lambda x : [pp(xx) for xx in x]
    for i in range(sum(iteration)):
        # - Get training data
        X,y = ecg_processor.get_batch("train")
        
        def get_grads(params):
            grads = compute_gradients(X, y, params, rnn, FLAGS, rnn._rng_key)
            return grads

        if(FLAGS.optimizer == "esgd"):
            opt_state = opt_update(i, get_params(opt_state), opt_state, get_grads)
        else:
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
            print(f"Loss is {pp(loss)} Lipschitzness loss over time {ppl(lip_loss_over_time)} Accuracy {pp(training_accuracy)} Attacked accuracy {pp(attacked_accuracy)}",flush=True)
            log(FLAGS.session_id,"training_accuracy",onp.float64(training_accuracy))
            log(FLAGS.session_id,"attacked_training_accuracy",onp.float64(attacked_accuracy))
            log(FLAGS.session_id,"kl_over_time",lip_loss_over_time)

        if((i+1) % (2*FLAGS.eval_step_interval) == 0):
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Start the load operations and mark each future with its URL
                futures = [executor.submit(get_mismatch_accuracy, 0.3, params, FLAGS, ecg_processor, rnn) for i in range(100)]
                mismatch_accuracies = onp.array([f.result() for f in futures])
                mean_mm_val_acc = onp.mean(mismatch_accuracies)
                log(FLAGS.session_id,"mm_val_robustness",list(mismatch_accuracies))
                print(f"MM robustness @0.3 {mean_mm_val_acc}+-{onp.std(mismatch_accuracies)}")
                # - Save the model
                if(mean_mm_val_acc > best_mean_mm_val_acc):
                    best_mean_mm_val_acc = mean_mm_val_acc
                    rnn.save(model_save_path, params)
                    print(f"Saved model under {model_save_path}")

        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = ecg_processor.N_val
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            val_bs = 200
            for i in range(0, set_size // val_bs):
                X,y = ecg_processor.get_batch("val", batch_size=val_bs)
                logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS, rnn._rng_key)
                rnn._rng_key, _ = random.split(rnn._rng_key)
                llot.append(lip_loss_over_time)
                predicted_labels = jnp.argmax(logits, axis=1)
                correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
                batched_validation_acc = get_batched_accuracy(y, logits)
                attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
                total_accuracy += batched_validation_acc
                attacked_total_accuracy += attacked_batched_validation_acc
            
            ecg_processor.reset("val")

            total_accuracy = total_accuracy / (set_size // val_bs)
            attacked_total_accuracy = attacked_total_accuracy / (set_size // val_bs)
            # - Logging
            log(FLAGS.session_id,"validation_accuracy",onp.float64(total_accuracy))
            log(FLAGS.session_id,"attacked_validation_accuracies",onp.float64(attacked_total_accuracy))
            mean_llot = onp.mean(onp.asarray(llot), axis=0)
            log(FLAGS.session_id,"validation_kl_over_time",list(onp.array(mean_llot, dtype=onp.float64)))

            # # - Save the model
            # if(total_accuracy > best_val_acc):
            #     best_val_acc = total_accuracy
            #     rnn.save(model_save_path, params)
            #     print(f"Saved model under {model_save_path}")

            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")