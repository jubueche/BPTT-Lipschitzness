from jax import config
config.FLAGS.jax_log_compiles=True

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
from import_data import DataLoader
from jax import lax


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


    # - Paths
    base_path = path.dirname(path.abspath(__file__))
    stored_name = '{}_{}_h{}_b{}_s{}'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),FLAGS.model_architecture, FLAGS.n_hidden,FLAGS.beta_lipschitzness,FLAGS.seed)
    model_name = f"{stored_name}_model.json"
    track_name = f"{stored_name}_track.json"
    model_save_path = path.join(base_path, f"Resources/{model_name}")
    track_save_path = path.join(base_path, f"Resources/Plotting/{track_name}")


    model_settings = utils.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                        len(learning_rates_list)))

    data_loader = DataLoader(FLAGS.batch_size)

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
    iteration = onp.array(FLAGS.how_many_training_steps.split(","), int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    color_range = onp.linspace(0,1,onp.sum(iteration))
    
    opt_init, opt_update, get_params = optimizers.adam(utils.get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)

    track_dict = {"training_accuracies": [], "attacked_training_accuracies": [], "kl_over_time": [], "validation_accuracy": [], "attacked_validation_accuracy": [], "validation_kl_over_time": [], "model_parameters": model_settings}
    best_val_acc = 0.0

    for i in range(sum(iteration)):
        # - Get training data
        (X,y) = data_loader.get_batch()
        y = jnp.argmax(y, axis=1)

        opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, cnn, FLAGS, cnn._rng_key)
        cnn._rng_key, _ = random.split(cnn._rng_key)

        if((i+1) % 10 == 0):
            params = get_params(opt_state)
            logits, _ = cnn.call(X, **params)
            loss = categorical_cross_entropy(y, logits)
            lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, cnn, FLAGS, cnn._rng_key)
            cnn._rng_key, _ = random.split(cnn._rng_key)
            lip_loss_over_time = list(onp.array(lip_loss_over_time, dtype=onp.float64))
            training_accuracy = get_batched_accuracy(y, logits)
            attacked_accuracy = get_batched_accuracy(y, logits_theta_star)
            print(f"Loss is {loss} Lipschitzness loss over time {lip_loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_accuracy}",flush=True)
            track_dict["training_accuracies"].append(onp.float64(training_accuracy))
            track_dict["attacked_training_accuracies"].append(onp.float64(attacked_accuracy))
            track_dict["kl_over_time"].append(lip_loss_over_time)
        
