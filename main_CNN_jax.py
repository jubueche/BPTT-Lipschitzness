from jax import config
config.FLAGS.jax_log_compiles=True

from random import randint
import time
import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from TensorCommands import input_data
from six.moves import xrange
from CNN_Jax import CNN
from jax import random
import jax.numpy as jnp
from loss_jax import categorical_cross_entropy, compute_gradient_and_update, attack_network
from jax.experimental import optimizers
import ujson as json
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from CNN.import_data import CNNDataLoader
from jax import lax
import math
import sqlite3
from architectures import cnn as arch
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
    FLAGS.Kernels = json.loads(FLAGS.Kernels)
    FLAGS.Dense = json.loads(FLAGS.Dense)

    data_loader = CNNDataLoader(FLAGS.batch_size, FLAGS.data_dir)
    flags_dict = vars(FLAGS)
    epochs_list = list(map(int, FLAGS.n_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(epochs_list) != len(learning_rates_list):
        raise Exception(
            '--n_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(epochs_list),
                                                        len(learning_rates_list)))
    
    steps_list = [math.ceil(epochs * data_loader.N_train/FLAGS.batch_size) for epochs in epochs_list]

    d_Out = data_loader.y_train.shape[1]

    Kernels = FLAGS.Kernels
    Dense   = FLAGS.Dense

    rng_key = random.PRNGKey(FLAGS.seed)
    _, *sks = random.split(rng_key, 10)
    K1 = onp.array(random.truncated_normal(sks[1],-2,2,(Kernels[0][0], Kernels[0][1], Kernels[0][2], Kernels[0][3]))* (onp.sqrt(6/(Kernels[0][0]*Kernels[0][1]*Kernels[0][2] + Kernels[0][0]*Kernels[0][1]*Kernels[0][3]))))
    K2 = onp.array(random.truncated_normal(sks[2],-2,2,(Kernels[1][0], Kernels[1][1], Kernels[1][2], Kernels[1][3]))* (onp.sqrt(6/(Kernels[1][0]*Kernels[1][1]*Kernels[1][2] + Kernels[1][0]*Kernels[1][1]*Kernels[1][3]))))
    CB1 = onp.array(random.truncated_normal(sks[6],-2,2,(1, Kernels[0][1], 1, 1))* (onp.sqrt(6/(Kernels[0][0]*Kernels[0][1]*Kernels[0][2] + Kernels[0][0]*Kernels[0][1]*Kernels[0][3]))))
    CB2 = onp.array(random.truncated_normal(sks[7],-2,2,(1, Kernels[1][1], 1, 1))* (onp.sqrt(6/(Kernels[1][0]*Kernels[1][1]*Kernels[1][2] + Kernels[1][0]*Kernels[1][1]*Kernels[1][3]))))
    W1 = onp.array(random.truncated_normal(sks[3],-2,2,(Dense[0][0], Dense[0][1]))* (onp.sqrt(6/(Dense[0][0] + Dense[0][1]))))
    W2 = onp.array(random.truncated_normal(sks[4],-2,2,(Dense[1][0], Dense[1][1]))* (onp.sqrt(6/(Dense[1][0] + Dense[1][1]))))
    W3 = onp.array(random.truncated_normal(sks[4],-2,2,(Dense[2][0], d_Out))* (onp.sqrt(6/(Dense[2][0] + d_Out))))
    B1 = onp.zeros((Dense[0][1],))
    B2 = onp.zeros((Dense[1][1],))
    B3 = onp.zeros((d_Out,))

    # - Create the model
    cnn = CNN(vars(FLAGS))

    init_params = {"K1": K1, "CB1": CB1, "K2": K2, "CB2": CB2, "W1": W1, "W2": W2, "W3": W3, "B1": B1, "B2": B2, "B3": B3}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    
    opt_init, opt_update, get_params = optimizers.adam(get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)
    best_val_acc = 0.0
    for i in range(sum(iteration)):
        # - Get training data
        (X,y) = data_loader.get_batch("train")
        y = jnp.argmax(y, axis=1)

        if(X.shape[0] == 0):
            continue
        opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, cnn, FLAGS, cnn._rng_key)
        cnn._rng_key, _ = random.split(cnn._rng_key)

        if((i+1) % 10 == 0):
            params = get_params(opt_state)
            logits, _ = cnn.call(X, [[0]], **params)
            loss = categorical_cross_entropy(y, logits)
            lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, cnn, FLAGS, cnn._rng_key)
            cnn._rng_key, _ = random.split(cnn._rng_key)
            lip_loss_over_time = list(onp.array(lip_loss_over_time, dtype=onp.float64))
            training_accuracy = get_batched_accuracy(y, logits)
            attacked_accuracy = get_batched_accuracy(y, logits_theta_star)

            log(FLAGS.session_id,"training_accuracy",onp.float64(training_accuracy))
            log(FLAGS.session_id,"attacked_training_accuracy",onp.float64(attacked_accuracy))
            if(not onp.isnan(lip_loss_over_time).any()):
                log(FLAGS.session_id,"kl_over_time",lip_loss_over_time)
                print(f"Loss is {loss} Lipschitzness loss over time {lip_loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_accuracy}",flush=True)

        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = data_loader.N_val
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            val_bs = 200
            for i in range(0, set_size // val_bs):
                X,y = data_loader.get_batch("val", batch_size=val_bs)
                y = jnp.argmax(y, axis=1)
                logits, _ = cnn.call(X, [[0]], **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, cnn, FLAGS, cnn._rng_key)
                cnn._rng_key, _ = random.split(cnn._rng_key)
                if(not onp.isnan(lip_loss_over_time).any()):
                    llot.append(lip_loss_over_time)
                batched_validation_acc = get_batched_accuracy(y, logits)
                attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
                total_accuracy += batched_validation_acc
                attacked_total_accuracy += attacked_batched_validation_acc

            total_accuracy = total_accuracy / (set_size // val_bs)
            attacked_total_accuracy = attacked_total_accuracy / (set_size // val_bs)
            # - Logging
            log(FLAGS.session_id,"validation_accuracy",onp.float64(total_accuracy))
            log(FLAGS.session_id,"attacked_validation_accuracies",onp.float64(attacked_total_accuracy))
            if(len(llot) > 0):
                mean_llot = onp.mean(onp.asarray(llot), axis=0)
            else:
                mean_llot = [0.0]
            log(FLAGS.session_id,"validation_kl_over_time",list(onp.array(mean_llot, dtype=onp.float64)))

            # - Save the model
            if(total_accuracy > best_val_acc):
                best_val_acc = total_accuracy
                cnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")

            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")