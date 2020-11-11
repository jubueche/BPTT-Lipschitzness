from jax import config
config.FLAGS.jax_log_compiles=True

import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
from ECG.ecg_data_loader import ECGDataLoader
import input_data_eager as input_data
from six.moves import xrange
from RNN_Jax import RNN
from GraphExecution import utils
from jax import random
import jax.numpy as jnp
from loss_jax import loss_normal, compute_gradient_and_update, attack_network
from jax.experimental import optimizers
import ujson as json
import wandb
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from random import randint
import time
import string
import sqlite3
import math

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

if __name__ == '__main__':

    parser = utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...",flush=True)
        print(unparsed,flush=True)
        sys.exit(0)

    if(FLAGS.model_architecture in ["lsnn","cnn"]):
        print("Please provide model_architecture=lsnn_ecg")
        sys.exit(0)

    # - Paths
    base_path = path.dirname(path.abspath(__file__))
    if FLAGS.session_id == 0:
        FLAGS.session_id = randint(1000000000, 9999999999)
    print("Session ID is ", FLAGS.session_id)
    model_name = f"{FLAGS.session_id}_model.json"
    track_name = f"{FLAGS.session_id}_track.json"
    FLAGS.start_time = int(time.time()*1000) #.strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = path.join(base_path, f"Resources/{model_name}")
    track_save_path = path.join(base_path, f"Resources/Plotting/{track_name}")

    def create_table_from_dict(dictionary, name):
        fieldset = []
        for key, val in dictionary.items():
            if type(val) == int:
                definition = "INTEGER"
            if type(val) == float:
                definition = "REAL"
            else:
                definition = "TEXT"
            
            if key == "session_id":
                fieldset.append("'{0}' {1} PRIMARY KEY".format(key, definition))
            else:
                fieldset.append("'{0}' {1}".format(key, definition))

        if len(fieldset) > 0:
            return "CREATE TABLE IF NOT EXISTS {0} ({1});".format(name, ", ".join(fieldset))

    def insert_row_from_dict(dictionary,name):
        column_names = list(dictionary.keys())
        column_values = list(dictionary.values())
        def format_value(val):
            if type(val) == int:
                return str(val)
            if type(val) == float:
                return str(val)
            return "\"" + str(val) + "\""
        
        return "INSERT INTO {0} ({1}) VALUES({2});".format(name,", ".join(column_names), ", ".join(map(format_value,column_values)))

    try:
        print("registering session...")
        conn = sqlite3.connect("sessions_"+ FLAGS.db + ".db", timeout=100)
        c = conn.cursor()
        c.execute(create_table_from_dict(vars(FLAGS),"sessions"))
        c.execute(insert_row_from_dict(vars(FLAGS), "sessions"))
        conn.commit()
        c.close()
    except sqlite3.Error as error:
        print("Failed to insert data into sqlite table", error)
    finally:
        if (conn):
            conn.close()
            print("Registering Complete")

    
    model_settings = utils.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )
    ecg_processor = ECGDataLoader(path=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    # model_settings["tau_adaptation"] = model_settings['spectrogram_length'] * model_settings['in_repeat']
    model_settings["spectrogram_length"] = ecg_processor.T
    model_settings['fingerprint_width'] = ecg_processor.n_channels

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

    # - Create the model
    rnn = RNN(model_settings)
 
    init_params = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    color_range = onp.linspace(0,1,onp.sum(iteration))
    
    opt_init, opt_update, get_params = optimizers.adam(utils.get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)

    track_dict = {"training_accuracies": [], "attacked_training_accuracies": [], "kl_over_time": [], "validation_accuracy": [], "attacked_validation_accuracy": [], "validation_kl_over_time": [], "model_parameters": model_settings}
    best_val_acc = 0.0

    pp = lambda x : ("%.3f" % x)
    ppl = lambda x : [pp(xx) for xx in x]
    for i in range(sum(iteration)):
        # - Get training data
        X,y = ecg_processor.get_batch("train")
        
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
            # print(f"Loss is {loss} Accuracy {training_accuracy}",flush=True)
            track_dict["training_accuracies"].append(onp.float64(training_accuracy))
            track_dict["attacked_training_accuracies"].append(onp.float64(attacked_accuracy))
            track_dict["kl_over_time"].append(lip_loss_over_time)

        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = ecg_processor.N_val
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            val_bs = 200
            for i in range(0, int(onp.ceil(set_size/val_bs))):
                X,y = ecg_processor.get_batch("val", batch_size=val_bs)
                logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS, rnn._rng_key)
                rnn._rng_key, _ = random.split(rnn._rng_key)
                llot.append(lip_loss_over_time)
                predicted_labels = jnp.argmax(logits, axis=1)
                correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
                batched_validation_acc = get_batched_accuracy(y, logits)
                attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
                total_accuracy += (batched_validation_acc * val_bs) / set_size
                attacked_total_accuracy += (attacked_batched_validation_acc * val_bs) / set_size

            # - Logging
            track_dict["validation_accuracy"].append(onp.float64(total_accuracy))
            track_dict["attacked_validation_accuracy"].append(onp.float64(attacked_total_accuracy))
            mean_llot = onp.mean(onp.asarray(llot), axis=0)
            track_dict["validation_kl_over_time"].append(list(onp.array(mean_llot, dtype=onp.float64)))

            # - Save the model
            if(total_accuracy > best_val_acc):
                best_val_acc = total_accuracy
                rnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")
                with open(track_save_path, "w") as f:
                    json.dump(track_dict, f)
                print(f"Saved track dict under {track_save_path}")


            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")