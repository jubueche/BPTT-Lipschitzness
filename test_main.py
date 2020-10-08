import input_data_eager as input_data
import tensorflow as tf
from RNN import RNN
from utils import get_parser
import sys
import os
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
    
    # - Read the weights and input from the Graz output
    with open("tests/graz_output.json", "r") as f:
        graz_dict = json.load(f)

    model_settings_graz = graz_dict["model_settings"]

    for key in model_settings_graz.keys():
        print(f"{key}: Graz {model_settings_graz[key]} Ours {model_settings[key]}")
        assert(model_settings_graz[key] == model_settings[key]), "Model settings differ"

    # - Define trainable variables
    d_In = model_settings['fingerprint_width']
    d_Out = model_settings["label_count"]
    W_in = tf.Variable(initial_value=graz_dict["W_in"], trainable=True)
    W_rec = tf.Variable(initial_value=graz_dict["W_rec"], trainable=True)
    W_out = tf.Variable(initial_value=graz_dict["W_out"], trainable=True)
    b_out = tf.Variable(initial_value=graz_dict["b_out"], trainable=True)

    # - Create the model
    rnn = RNN(model_settings)

    # - Define loss function
    def loss_normal(target_output, logits):
        cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
        # - TODO Need to adapt call method so that it also returns spikes
        # if FLAGS.model_architecture == 'lsnn':
        #     regularization_f0 = 10 / 1000  # 10Hz
        #     loss_reg = tf.reduce_sum(tf.square(average_fr - regularization_f0) * FLAGS.reg)
        #     cross_entropy_mean += loss_reg
        return cross_entropy_mean

    logits_graz = tf.cast(graz_dict["logits"],dtype=tf.float32)
    logits = rnn.call(fingerprint_input=graz_dict["train_input"], W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
    loss = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits) # + loss_lip(logits, logits_adv1)
    loss_graz = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits_graz)

    print("=========================================================")
    print(f"Loss ours {loss.numpy()} and theirs {loss_graz.numpy()}")
    print(f"Sum of absolute differences is {tf.reduce_sum(tf.math.abs((logits-logits_graz))).numpy()}")
    print("=========================================================")