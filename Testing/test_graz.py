import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
import models
import utils
import input_data
import tensorflow as tf
import loss as loss_class
import ujson as json
import numpy as np

def execute_graz_rnn(model_settings, FLAGS):
    tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)
    sess = tf.compat.v1.InteractiveSession()

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir,
        FLAGS.silence_percentage, FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings, FLAGS.summaries_dir,
        FLAGS.n_thr_spikes, FLAGS.in_repeat
    )
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    n_thr_spikes = max(1, FLAGS.n_thr_spikes)
    training_placeholder = tf.compat.v1.placeholder(tf.bool, name='is_training')
    input_placeholder = tf.compat.v1.placeholder(tf.float32, [None, model_settings['fingerprint_size'] * (2 * n_thr_spikes - 1) * model_settings['in_repeat']],name='fingerprint_input')
    fingerprint_input = input_placeholder
    model_out = models.create_model(fingerprint_input,model_settings,FLAGS.model_architecture,is_training=training_placeholder)
    logits, spikes, dropout_prob = model_out
    av = tf.reduce_mean(spikes, axis=(0, 1))
    ground_truth_input = tf.compat.v1.placeholder(tf.int64, [None], name='groundtruth_input')
    with tf.compat.v1.name_scope('loss'):
        loss = loss_class.evaluate_loss_function(target_output=ground_truth_input, logits=logits, average_fr=0, FLAGS=FLAGS) # - FIXME Introduce av instead of 0

    # - Would define optimizers here
    gradients = tf.compat.v1.train.AdamOptimizer().compute_gradients(loss)

    tf.compat.v1.global_variables_initializer().run()
    train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)

    fd = {
        fingerprint_input: train_fingerprints,
        ground_truth_input: train_ground_truth,
        dropout_prob: FLAGS.dropout_prob,
        training_placeholder: True,}

    logits_value, loss_value, spikes_value, gradients_value = sess.run([logits, loss, spikes, gradients], feed_dict=fd)

    # - Get the weights
    W_in, W_rec, W_out, b_out = sess.run([e for e in tf.compat.v1.trainable_variables()])
    sess.close()
    np.fill_diagonal(W_rec, 0.)
    gradients_value = [e[0].tolist() for e in gradients_value]

    return_dict = {"logits": logits_value.tolist(), "loss": loss_value.tolist(), "spikes": spikes_value.tolist(),
                        "train_input": train_fingerprints.tolist(), "train_groundtruth": train_ground_truth.tolist(),
                        "W_in": W_in.tolist(), "W_rec": W_rec.tolist(), "W_out": W_out.tolist(), "b_out": b_out.tolist(),
                        "gradients": gradients_value}



    return return_dict


if __name__ == '__main__':
    tf.random.set_seed(42)
    
    parser = utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...")
        print(unparsed)
        sys.exit(0)
    
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )
    model_settings['n_hidden'] = FLAGS.n_hidden
    model_settings['n_layer'] = FLAGS.n_layer
    model_settings['dropout_prob'] = FLAGS.dropout_prob
    model_settings['n_lif_frac'] = FLAGS.n_lif_frac
    model_settings['tau'] = FLAGS.tau
    model_settings['refr'] = FLAGS.refr
    model_settings['beta'] = FLAGS.beta
    model_settings['n_thr_spikes'] = FLAGS.n_thr_spikes
    model_settings['n_delay'] = FLAGS.n_delay
    model_settings['eprop'] = FLAGS.eprop
    model_settings['random_eprop'] = FLAGS.random_eprop
    model_settings['avg_spikes'] = FLAGS.avg_spikes

    graz_values = execute_graz_rnn(model_settings, FLAGS)

    graz_values["model_settings"] = model_settings

    fn = os.path.join(os.path.dirname(__file__), "tests/graz_output.json")
    print(fn)
    with open(fn, "w") as f:
        json.dump(graz_values, f)