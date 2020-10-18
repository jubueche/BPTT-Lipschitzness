import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
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

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

if __name__ == '__main__':

    parser = utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...")
        print(unparsed)
        sys.exit(0)

    # - Paths
    base_path = path.dirname(path.abspath(__file__))
    postfix = ""
    if(FLAGS.lipschitzness):
        postfix += "_lipschitzness"
    stored_name = '{}_{}_l{}_h{}_w{}str{}_do{}{}'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        FLAGS.model_architecture, FLAGS.n_layer, FLAGS.n_hidden, FLAGS.window_size_ms, FLAGS.window_stride_ms,
        FLAGS.dropout_prob,postfix)
    model_name = f"{stored_name}_model.json"
    track_name = f"{stored_name}_track.json"
    model_save_path = path.join(base_path, f"Resources/{model_name}")
    track_save_path = path.join(base_path, f"Resources/Plotting/{track_name}")

    wandb.init(project="robust-lipschitzness", config=vars(FLAGS))

    model_settings = utils.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    model_settings["tau_adaptation"] = model_settings['spectrogram_length'] * model_settings['in_repeat']
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
    #W_in = onp.random.randn(d_In, FLAGS.n_hidden)*(onp.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978)
    W_in = onp.array(random.truncated_normal(random.PRNGKey(0),-2,2,(d_In, FLAGS.n_hidden))* (onp.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978))
    W_rec = onp.array(random.truncated_normal(random.PRNGKey(1),-2,2,(FLAGS.n_hidden, FLAGS.n_hidden))* (onp.sqrt(1/(FLAGS.n_hidden)) / .87962566103423978))
    #W_rec = onp.random.randn(FLAGS.n_hidden,FLAGS.n_hidden) * (onp.sqrt(1/FLAGS.n_hidden) / .87962566103423978)
    onp.fill_diagonal(W_rec, 0.)
    W_out = onp.array(random.truncated_normal(random.PRNGKey(2),-2,2,(FLAGS.n_hidden, d_Out))*0.01)
    b_out = onp.zeros((d_Out,))

    # - Create the model
    rnn = RNN(model_settings)

    init_params = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}
    iteration = onp.array(FLAGS.how_many_training_steps.split(","), int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    color_range = onp.linspace(0,1,onp.sum(iteration))
    
    opt_init, opt_update, get_params = optimizers.adam(utils.get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)

    track_dict = {"training_accuracies": [], "attacked_training_accuracies": [], "kl_over_time": [], "validation_accuracy": [], "attacked_validation_accuracy": [], "validation_kl_over_time": []}
    best_val_acc = 0.0
    for i in range(sum(iteration)):
        # - Get training data
        train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,FLAGS.background_volume, time_shift_samples, 'training')
        X = train_fingerprints.numpy()
        y = train_ground_truth.numpy()

        opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, rnn, FLAGS)
        
        if(i % 10 == 0):
            params = get_params(opt_state)
            logits, spikes = rnn.call(X, **params)
            avg_firing = jnp.mean(spikes, axis=1)
            loss = loss_normal(y, logits, avg_firing, FLAGS.reg)
            lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS)
            lip_loss_over_time = list(onp.array(lip_loss_over_time, dtype=onp.float64))
            training_accuracy = get_batched_accuracy(y, logits)
            attacked_accuracy = get_batched_accuracy(y, logits_theta_star)
            print(f"Loss is {loss} Lipschitzness loss over time {lip_loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_accuracy}")
            
            # - Logging for wandb
            track_dict["training_accuracies"].append(onp.float64(training_accuracy))
            track_dict["attacked_training_accuracies"].append(onp.float64(attacked_accuracy))
            track_dict["kl_over_time"].append(lip_loss_over_time)
            
            plt.subplot(121)
            plt.plot(track_dict["training_accuracies"], color="g", label="Training acc.")
            plt.plot(track_dict["attacked_training_accuracies"], color="r", label="Attacked training acc.")
            plt.legend()
            plt.subplot(122)
            for idx,l in enumerate(track_dict["kl_over_time"]):
                plt.plot(l, color=(1.0,1.0,color_range[idx]))
            wandb.log({"train": plt})


        if((i+1) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            set_size = audio_processor.set_size('validation')
            llot = []
            total_accuracy = attacked_total_accuracy = 0
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0.0, 'validation'))
                X = validation_fingerprints.numpy()
                y = validation_ground_truth.numpy()
                logits, _ = rnn.call(X, **params)
                lip_loss_over_time, logits_theta_star = attack_network(X, params, logits, rnn, FLAGS)
                llot.append(lip_loss_over_time)
                predicted_labels = jnp.argmax(logits, axis=1)
                correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
                batched_validation_acc = get_batched_accuracy(y, logits)
                attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
                total_accuracy += (batched_validation_acc * FLAGS.batch_size) / set_size
                attacked_total_accuracy += (attacked_batched_validation_acc * FLAGS.batch_size) / set_size

            # - Logging
            color_range_val = onp.linspace(0,1,int(sum(iteration)/FLAGS.eval_step_interval))
            track_dict["validation_accuracy"].append(onp.float64(total_accuracy))
            track_dict["attacked_validation_accuracy"].append(onp.float64(attacked_total_accuracy))
            mean_llot = onp.mean(onp.asarray(llot), axis=0)
            track_dict["validation_kl_over_time"].append(mean_llot)
            plt.subplot(121)
            plt.plot(track_dict["validation_accuracy"], color="g", label="Val. acc.")
            plt.plot(track_dict["attacked_validation_accuracy"], color="r", label="Attacked val. acc.")
            plt.legend()
            plt.subplot(122)
            for idx,l in enumerate(track_dict["validation_kl_over_time"]):
                plt.plot(l, color=(1.0,1.0,color_range_val[idx]))
            wandb.log({"val": plt})

            # - Save the model
            if(total_accuracy > best_val_acc):
                best_val_acc = total_accuracy
                rnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")


            print(f"Validation accuracy {total_accuracy} Attacked val. accuracy {attacked_total_accuracy}")

    with open(track_save_path, "w") as f:
        json.dump(track_dict, f)

        