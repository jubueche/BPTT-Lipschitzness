import argparse
import tensorflow as tf

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='tmp/speech_dataset/',
        help="""\
        Where to download the speech training data to.
        """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
        How loud the background noise should be, between 0 and 1.
        """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
        How many of the training samples have background noise mixed in.
        """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be silence.
        """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be unknown words.
        """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
        Range to randomly shift the training audio by in time.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectogram timeslices.',)
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC / FBANK fingerprint',
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='15000,3000',
        help='How many training loops to run',)
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.001,0.0001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='results/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--train_dir',
        type=str,
        default='results/speech_commands_train',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=1000,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    parser.add_argument(
        '--quantize',
        type=bool,
        default=False,
        help='Whether to train the model for eight-bit deployment')
    parser.add_argument(
        '--preprocess',
        type=str,
        default='mfcc',
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
    parser.add_argument(
        '--n_hidden',
        type=int,
        default=2048,
        help='Number of hidden units in recurrent models.')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=1,
        help='Number of stacked layers in recurrent models.')
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.0,
        help='Dropoout probability for recurrent models.',)
    parser.add_argument(
        '--print_every',
        type=int,
        default=10,
        help='How often to print the training step results.',)
    parser.add_argument(
        '--reg',
        type=float,
        default=0.001,
        help='Firing rate regularization coefficient.',)
    parser.add_argument(
        '--n_lif_frac',
        type=float,
        default=0.0,
        help='Fraction of non-adaptive LIF neurons in LSNN.',)
    parser.add_argument(
        '--beta',
        type=float,
        default=2.,
        help='Adaptation coefficient of ALIF neurons in LSNN.',)
    parser.add_argument(
        '--comment',
        type=str,
        default='',
        help='String to append to output dir.')
    parser.add_argument(
        '--n_thr_spikes',
        type=int,
        default=-1,
        help='Number of thresholds in thr-crossing analog to spike encoding.',)
    parser.add_argument(
        '--tau',
        type=float,
        default=20.,
        help='Membrane time constant of ALIF neurons in LSNN.',)
    parser.add_argument(
        '--refr',
        type=int,
        default=2,
        help='Number of refractory time steps of ALIF neurons in LSNN.',)
    parser.add_argument(
        '--in_repeat',
        type=int,
        default=1,
        help='Number of time steps to repeat every input feature.',)
    parser.add_argument(
        '--n_delay',
        type=int,
        default=0,
        help='Maximum number of timesteps for synapse delay in LSNN.',)
    parser.add_argument(
        '--random_eprop',
        type=bool,
        default=False,
        help='Use random eprop for LSNN training')
    parser.add_argument(
        '--eprop',
        type=bool,
        default=False,
        help='Use symmetric eprop for LSNN training')
    parser.add_argument(
        '--avg_spikes',
        type=bool,
        default=True,
        help='Average spikes over time for readout')
    parser.add_argument(
        '--lipschitzness',
        default=False,
        action='store_true',
        help='Use lipschitzness loss or not. Default: False'
    )
    parser.add_argument(
        '--lipschitzness_loss',
        default="kl",
        type=str,
        help='Loss to use for Lipschitzness measure (mse, kl)'
    )
    parser.add_argument(
        '--beta_lipschitzness',
        default=1.0,
        type=float,
        help='Beta used for weighting lipschitzness term'
    )
    parser.add_argument(
        '--step_size_lipschitzness',
        default=0.0001,
        type=float,
        help='Step size used to update Theta*'
    )

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
            value: A member of tf.logging.
        Raises:
            ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == 'INFO':
            return tf.compat.v1.logging.INFO
        elif value == 'DEBUG':
            return tf.compat.v1.logging.DEBUG
        elif value == 'ERROR':
            return tf.compat.v1.logging.ERROR
        elif value == 'FATAL':
            return tf.compat.v1.logging.FATAL
        elif value == 'WARN':
            return tf.compat.v1.logging.WARN
        else:
            raise argparse.ArgumentTypeError('Not an expected value')
    parser.add_argument(
        '--verbosity',
        type=verbosity_arg,
        default=tf.compat.v1.logging.INFO,
        help='Log verbosity. Can be "INFO", "DEBUG", "ERROR", "FATAL", or "WARN"')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='Optimizer (gradient_descent, momentum, adam)')

    return parser