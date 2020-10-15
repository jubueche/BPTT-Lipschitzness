import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/Jax")
import input_data_eager as input_data
from RNN_Jax import RNN
from GraphExecution import utils
from jax import random
import ujson as json
import tempfile

if __name__ == '__main__':

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
    W_in = onp.random.randn(d_In, FLAGS.n_hidden)*(onp.sqrt(2/(d_In + FLAGS.n_hidden)) / .87962566103423978)
    W_rec = onp.random.randn(FLAGS.n_hidden,FLAGS.n_hidden) * (onp.sqrt(1/FLAGS.n_hidden) / .87962566103423978)
    onp.fill_diagonal(W_rec, 0.)
    W_out = onp.random.randn(FLAGS.n_hidden,d_Out)*0.01
    b_out = onp.zeros((d_Out,))

    theta_init = {"W_in": W_in, "W_rec": W_rec, "W_out": W_out, "b_out": b_out}
    # - Create the model
    rnn = RNN(model_settings)

    train_fingerprints, train_ground_truth = audio_processor.get_data(FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,FLAGS.background_volume, time_shift_samples, 'training')
    X = train_fingerprints.numpy()
    y = train_ground_truth.numpy()

    logits_initial, spikes_initial = rnn.call(X, **theta_init)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        fn = path.join(tmp_dir_name, "tmp_model.json")
        rnn.save(fn, theta_init)

        rnn_loaded, theta_loaded = RNN.load(fn)
        logits_loaded, spikes_loaded = rnn_loaded.call(X, **theta_loaded)

    if(onp.isclose(logits_initial, logits_loaded).all() and onp.isclose(spikes_initial, spikes_loaded).all()):
        print("\033[92mPASSED\033[0m Load Save")
    else:
        print("\033[91mFAILED\033[0m Load Save")

        