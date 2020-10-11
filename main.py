# - Implementation of custom RNN with one hidden layer
import tensorflow as tf
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import function
from collections import namedtuple
import numpy as np
from GraphExecution import utils
import sys
import os
from datetime import datetime
import input_data_eager as input_data
import tensorflow_probability as tfp
from six.moves import xrange
import tensorflow.keras.backend as K
from RNN import RNN
from GraphExecution import loss as loss_class

DEBUG = False

if __name__ == '__main__':

    print(f"Tensorflow version {tf.__version__} Using eager evalation {tf.executing_eagerly()} should be True")

    parser = utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...")
        print(unparsed)
        sys.exit(0)

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
    W_in = tf.Variable(initial_value=tf.random.truncated_normal(shape=(d_In,FLAGS.n_hidden), mean=0.0, stddev= tf.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978), trainable=True)
    W_rec = tf.Variable(initial_value=tf.linalg.set_diag(tf.random.truncated_normal(shape=(FLAGS.n_hidden,FLAGS.n_hidden), mean=0., stddev= tf.sqrt(1/FLAGS.n_hidden) / .87962566103423978), tf.zeros([FLAGS.n_hidden])), trainable=True)
    W_out = tf.Variable(initial_value=tf.random.truncated_normal(shape=(FLAGS.n_hidden,d_Out), mean=0.0, stddev=0.01), trainable=True)
    b_out = tf.Variable(initial_value=tf.zeros(shape=(d_Out,)), trainable=True)

    # - Create the model
    rnn = RNN(model_settings)

    iteration = [1000,500]; lrs = [0.001, 0.0001]; current_idx = 0 ; cum_sum = 0
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    @tf.function(autograph=not DEBUG)
    def compute_and_apply_gradients(train_fingerprints, train_ground_truth, W_in, W_rec, W_out, b_out):
        with tf.GradientTape(persistent=False) as tape_normal:
            logits, spikes = rnn.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            average_fr = tf.reduce_mean(spikes, axis=(1,2))
            loss = loss_class.normal_loss(train_ground_truth, logits, average_fr, FLAGS)
        # - Get the gradients
        gradients = tape_normal.gradient(loss, [W_in,W_rec,W_out,b_out])
        optimizer.apply_gradients(zip(gradients,[W_in,W_rec,W_out,b_out]))

    for i in range(sum(iteration)):
        # - Get some data
        if(i >= cum_sum + iteration[current_idx]):
            current_idx += 1
            cum_sum += iteration[current_idx-1]
            optimizer.lr= lrs[current_idx]
        # - Get training data
        train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,FLAGS.background_volume, time_shift_samples, 'training')
        
        compute_and_apply_gradients(train_fingerprints, train_ground_truth, W_in, W_rec, W_out, b_out)
        
        if(i % 10 == 0):
            logits, spikes = rnn.call(fingerprint_input=train_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            average_fr = tf.reduce_mean(spikes, axis=(1,2))
            loss = loss_class.normal_loss(train_ground_truth, logits, average_fr, FLAGS)
            predicted_indices = tf.cast(tf.argmax(input=logits, axis=1), dtype=tf.int32)
            correct_prediction = tf.cast(tf.equal(predicted_indices, train_ground_truth), dtype=tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            print(f"Loss is {loss.numpy()} Accuracy is {accuracy.numpy()}")

        if((i+1) % 399 == 0):
            set_size = audio_processor.set_size('validation')
            total_accuracy = 0
            for i in xrange(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                            0.0, 0.0, 'validation'))
                logits, _ = rnn.call(fingerprint_input=validation_fingerprints, W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
                predicted_indices = tf.cast(tf.argmax(input=logits, axis=1), dtype=tf.int32)
                correct_prediction = tf.cast(tf.equal(predicted_indices, validation_ground_truth), dtype=tf.float32)
                validation_accuracy = tf.reduce_mean(correct_prediction)
                total_accuracy += (validation_accuracy * FLAGS.batch_size) / set_size

            print(f"Validation accuracy is {total_accuracy}")