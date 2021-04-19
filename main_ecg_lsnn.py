from jax import config
config.FLAGS.jax_log_compiles=True

import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from ECG.ecg_data_loader import ECGDataLoader
from RNN_Jax import RNN
from jax import random
import jax.numpy as jnp
from loss_jax import compute_gradient_and_update, compute_gradients
from jax.experimental import optimizers
from EntropySGD.entropy_sgd import EntropySGD_Jax
from ABCD.abcd import ABCD_Jax
import math
import time
from architectures import ecg_lsnn as arch
from architectures import log
from concurrent.futures import ThreadPoolExecutor, as_completed
from experiment_utils import get_batched_accuracy, _get_acc_batch, get_val_acc, _get_mismatch_data, get_val_acc, get_lr_schedule

if __name__ == '__main__':
    t0 = time.time()
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
    FLAGS.network = rnn # TODO changing this would require additional methods for get_acc etc. is there a better way?
    FLAGS.architecture = "ecg_lsnn" # - . -
 
    init_params = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    
    if(FLAGS.optimizer == "adam"):
        opt_init, opt_update, get_params = optimizers.adam(get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    elif(FLAGS.optimizer == "sgd"):
        opt_init, opt_update, get_params = optimizers.sgd(get_lr_schedule(iteration,lrs))
    elif(FLAGS.optimizer == "esgd"):
        config = dict(momentum=0.9, damp=0.0, nesterov=True, weight_decay=0.0, L=10, eps=1e-4, g0=1e-2, g1=1e-3, langevin_lr=0.1, langevin_beta1=0.75, b1=0.0, b2=1.0, eps_adam=1e-8)
        opt_init, opt_update, get_params = EntropySGD_Jax(get_lr_schedule(iteration,lrs), config)
    elif(FLAGS.optimizer == "abcd"):
        config = dict(L=FLAGS.abcd_L, eta_A=FLAGS.abcd_etaA, b1=0.9, b2=0.999, eps=1e-8)
        opt_init, opt_update, get_params = ABCD_Jax(get_lr_schedule(iteration,lrs), config)
    else:
        print("Invalid optimizer")
        sys.exit(0)
    opt_state = opt_init(init_params)

    best_val_acc = best_mean_mm_val_acc = 0.0
    for i in range(sum(iteration)):
        # - Get training data
        X,y = ecg_processor.get_batch("train")
        
        def get_grads(params):
            grads = compute_gradients(X, y, params, rnn, FLAGS, rnn._rng_key)
            return grads

        if(FLAGS.optimizer == "esgd"):
            opt_state = opt_update(i, get_params(opt_state), opt_state, get_grads)
        elif(FLAGS.optimizer == "abcd"):
            opt_state = opt_update(i, get_params(opt_state), opt_state, get_grads, FLAGS, rnn._rng_key)
        else:
            opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, rnn, FLAGS, rnn._rng_key)
        rnn._rng_key, _ = random.split(rnn._rng_key)

        if((i+1) % 10 == 0):
            params = get_params(opt_state)
            elapsed_time = float(onp.round(100 * (time.time() - t0) / 3600) / 100)
            training_accuracy, attacked_training_accuracy, loss_over_time, loss = _get_acc_batch(X, y, params, FLAGS, ATTACK=True)
            print(f"{elapsed_time} h Epoch {ecg_processor.n_epochs} i {i} / {sum(iteration)} Loss is {loss} Lipschitzness loss over time {loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_training_accuracy}",flush=True)
            log(FLAGS.session_id,"training_accuracy",training_accuracy)
            log(FLAGS.session_id,"attacked_training_accuracy",attacked_training_accuracy)
            log(FLAGS.session_id,"kl_over_time",loss_over_time)

        if((i+1) % (2*FLAGS.eval_step_interval) == 0):
            params = get_params(opt_state)
            mismatch_accuracies = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_get_mismatch_data, vars(FLAGS), params, 0.3, FLAGS.data_dir, "val") for i in range(50)]
                for future in as_completed(futures):
                    mismatch_accuracies.append(future.result())
            mismatch_accuracies = onp.array(mismatch_accuracies, dtype=onp.float64)
            
            mean_mm_val_acc = onp.mean(mismatch_accuracies)
            log(FLAGS.session_id,"mm_val_robustness",list(mismatch_accuracies))
            print(f"Epoch {ecg_processor.n_epochs} i {i} MM robustness @0.3 {mean_mm_val_acc}+-{onp.std(mismatch_accuracies)}")
            

        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            val_acc, attacked_val_acc, loss_over_time, loss = get_val_acc(vars(FLAGS), params, FLAGS.data_dir, ATTACK=True)
            log(FLAGS.session_id,"validation_accuracy",val_acc)
            log(FLAGS.session_id,"attacked_validation_accuracies",attacked_val_acc)
            log(FLAGS.session_id,"validation_kl_over_time",list(loss_over_time))
            print(f"Epoch {ecg_processor.n_epochs} i {i} Validation accuracy {val_acc} Attacked val. accuracy {attacked_val_acc}")
            if(val_acc > best_val_acc):
                best_val_acc = val_acc
                rnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")