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
    
    # - Read the weights and input from the Graz output
    fn = os.path.join(os.path.dirname(__file__), "tests/graz_output.json")
    with open(fn, "r") as f:
        graz_dict = json.load(f)

    model_settings_graz = graz_dict["model_settings"]

    for key in model_settings_graz.keys():
        # print(f"{key}: Graz {model_settings_graz[key]} Ours {model_settings[key]}")
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
        return cross_entropy_mean

    logits_graz = tf.cast(graz_dict["logits"],dtype=tf.float32)

    @tf.function
    def get_loss_and_gradients():
        with tf.GradientTape(persistent=False) as tape:
            logits, spikes = rnn.call(fingerprint_input=graz_dict["train_input"], W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            loss = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits)
        gradients = tape.gradient(loss, [W_in,W_rec,W_out,b_out])
        return loss, logits, spikes, gradients

    loss, logits, spikes, gradients = get_loss_and_gradients()

    gradients = [g.numpy() for g in gradients]

    print("Checking gradients...")
    graz_gradients = [np.asarray(g) for g in graz_dict["gradients"]]
    pass_grad = True
    for i in range(len(gradients)):
        assert(gradients[i].shape == graz_gradients[i].shape)
        if (not (np.isclose(gradients[i],graz_gradients[i])).all()):
            pass_grad = False

    loss_graz = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits_graz)
    spikes_graz = np.asarray(graz_dict["spikes"])

    print("Checking loss...")
    # print(f"Loss ours {loss.numpy()} and theirs {loss_graz.numpy()}")
    d = tf.reduce_sum(tf.math.abs((logits-logits_graz))).numpy()
    # print(f"Sum of absolute differences is {d}")
    if(abs(d) < 1e-6 and pass_grad):
        print("\033[92mPASSED\033[0m")
    else:
        print("\033[91mFAILED\033[0m")
    print("=========================================================")