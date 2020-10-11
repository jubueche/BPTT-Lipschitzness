import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
import input_data_eager as input_data
import tensorflow as tf
from RNN import RNN
import ujson as json
import numpy as np
from utils import get_parser, prepare_model_settings


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



    train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,FLAGS.background_volume, time_shift_samples, 'training')
    
    
    
    # - Define trainable variables
    d_In = model_settings['fingerprint_width']
    d_Out = model_settings["label_count"]
    W_in = tf.Variable(initial_value=tf.random.truncated_normal(shape=(d_In,FLAGS.n_hidden), mean=0.0, stddev= tf.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978), trainable=True)
    W_rec = tf.Variable(initial_value=tf.linalg.set_diag(tf.random.truncated_normal(shape=(FLAGS.n_hidden,FLAGS.n_hidden), mean=0., stddev= tf.sqrt(1/FLAGS.n_hidden) / .87962566103423978), tf.zeros([FLAGS.n_hidden])), trainable=True)
    W_out = tf.Variable(initial_value=tf.random.truncated_normal(shape=(FLAGS.n_hidden,d_Out), mean=0.0, stddev=0.01), trainable=True)
    b_out = tf.Variable(initial_value=tf.zeros(shape=(d_Out,)), trainable=True)

    # - Create the model
    rnn = RNN(model_settings)
    rnn_batched = RNN(model_settings)

    # - Define loss function
    def loss_normal(target_output, logits):
        cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
        return cross_entropy_mean

    #logits_graz = tf.cast(graz_dict["logits"],dtype=tf.float32)

    @tf.function
    def get_loss_and_gradients():
        with tf.GradientTape(persistent=False) as tape:
            logits, spikes = rnn.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out, batch_sized=False)
            loss = loss_class.normal_loss(train_ground_truth, logits)
        gradients = tape.gradient(loss, [W_in,W_rec,W_out,b_out])
        return loss, logits, spikes, gradients

    def get_loss_and_gradients_batched():
        with tf.GradientTape(persistent=False) as tape:
            logits2, spikes2 = rnn_batched.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            loss2 = loss_class.normal_loss(train_ground_truth, logits2)
        gradients2 = tape.gradient(loss2, [W_in,W_rec,W_out,b_out])
        return loss2, logits2, spikes2, gradients2

    loss_batched, logits_batched, spikes_batched, gradients_batched = get_loss_and_gradients_batched()
    loss, logits, spikes, gradients = get_loss_and_gradients()

    gradients = [g.numpy() for g in gradients]
    gradients_batched = [g.numpy() for g in gradients_batched]

    print('Checking gradients')
    pass_grad = True
    for i in range(len(gradients)):
        assert(gradients[i].shape == gradients_batched[i].shape)
        if (not (np.isclose(gradients[i],gradients_batched[i])).all()):
            pass_grad = False


    print("Checking loss...")
    # print(f"Loss ours {loss.numpy()} and theirs {loss_graz.numpy()}")
    d = tf.reduce_sum(tf.math.abs((logits-logits_batched))).numpy()
    # print(f"Sum of absolute differences is {d}")
    if(abs(d) < 1e-6 and pass_grad):
        print("\033[92mPASSED\033[0m")
    else:
        print("\033[91mFAILED\033[0m")
    print("=========================================================")