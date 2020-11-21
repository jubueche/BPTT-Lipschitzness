from jax import config
config.FLAGS.jax_log_compiles=True

from random import randint
import time
import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
import input_data_eager as input_data
from six.moves import xrange
from CNN_Jax import CNN
from GraphExecution import utils
from jax import random
import jax.numpy as jnp
from loss_jax import categorical_cross_entropy, compute_gradient_and_update, attack_network
from jax.experimental import optimizers
import ujson as json
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from CNN.import_data import DataLoader
from jax import lax
import math
import sqlite3

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


    if(FLAGS.model_architecture in ["lsnn","lsnn_ecg"]):
        print("Please provide model_architecture=cnn")
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

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    epochs_list = list(map(int, FLAGS.n_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(epochs_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(epochs_list),
                                                        len(learning_rates_list)))

    data_loader = DataLoader(FLAGS.batch_size, FLAGS.data_dir)
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
    cnn = CNN(model_settings)

    init_params = {"K1": K1, "CB1": CB1, "K2": K2, "CB2": CB2, "W1": W1, "W2": W2, "W3": W3, "B1": B1, "B2": B2, "B3": B3}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    color_range = onp.linspace(0,1,onp.sum(iteration))
    
    opt_init, opt_update, get_params = optimizers.adam(utils.get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)

    track_dict = {"training_accuracies": [], "attacked_training_accuracies": [], "kl_over_time": [], "validation_accuracy": [], "attacked_validation_accuracy": [], "validation_kl_over_time": [], "model_parameters": model_settings}
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
            track_dict["training_accuracies"].append(onp.float64(training_accuracy))
            track_dict["attacked_training_accuracies"].append(onp.float64(attacked_accuracy))
            if(not onp.isnan(lip_loss_over_time).any()):
                track_dict["kl_over_time"].append(lip_loss_over_time)
                print(f"Loss is {loss} Lipschitzness loss over time {lip_loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_accuracy}",flush=True)

        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = data_loader.N_val
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            val_bs = 200
            for i in range(0, int(onp.ceil(set_size/val_bs))):
                X,y = data_loader.get_batch("val", batch_size=val_bs)
                y = jnp.argmax(y, axis=1)
                logits, _ = cnn.call(X, [[0]], **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, cnn, FLAGS, cnn._rng_key)
                cnn._rng_key, _ = random.split(cnn._rng_key)
                if(not onp.isnan(lip_loss_over_time).any()):
                    llot.append(lip_loss_over_time)
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
                cnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")
                with open(track_save_path, "w") as f:
                    json.dump(track_dict, f)
                print(f"Saved track dict under {track_save_path}")


            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")
        
