import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import os
import gzip

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
from python_speech_features import fbank

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'

def prepare_words_list(wanted_words):
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words

def which_set(filename, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                        (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def load_wav_file(filename):
    wav_loader = io_ops.read_file(filename)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1).audio.flatten()
    return wav_decoder

def save_wav_file(filename, wav_data, sample_rate):
    wav_encoder = tf.audio.encode_wav(wav_data,sample_rate)
    io_ops.write_file(filename, wav_encoder)

def get_features_range(model_settings):
    if model_settings['preprocess'] == 'average':
        features_min = 0.0
        features_max = 127.5
    elif model_settings['preprocess'] == 'mfcc':
        features_min = -247.0
        features_max = 30.0
    elif model_settings['preprocess'] == 'micro':
        features_min = 0.0
        features_max = 26.0
    else:
        raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                       ' "average", or "micro")' % (model_settings['preprocess']))
    return features_min, features_max

def load_mnist(path, kind='train'):


    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class AudioProcessor(object):
    def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
                   wanted_words, validation_percentage, testing_percentage,
                   model_settings, summaries_dir, n_thr_spikes=-1, n_repeat=1, seed=59185):
        self.RANDOM_SEED = seed
        np.random.seed(seed)
        if data_dir:
            self.data_dir = data_dir
            self.maybe_download_and_extract_dataset(data_url, data_dir)
            self.prepare_data_index(silence_percentage, unknown_percentage,
                              wanted_words, validation_percentage,
                              testing_percentage)
            self.prepare_background_data()
        self.n_thr_spikes = max(1, n_thr_spikes)
        self.n_repeat = max(1, n_repeat)

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                tf.compat.v1.logging.error(
                    'Failed to download URL: %s to folder: %s', data_url, filepath)
                tf.compat.v1.logging.error(
                    'Please make sure you have enough free space and'
                    ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            tf.compat.v1.logging.info('Successfully downloaded %s (%d bytes)',
                                        filename, statinfo.st_size)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def prepare_data_index(self, silence_percentage, unknown_percentage,
                         wanted_words, validation_percentage,
                         testing_percentage):
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(self.RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}
        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))
        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })
            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data(self):
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data
        search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,'*.wav')
        for wav_path in gfile.Glob(search_path):
            wav_loader = io_ops.read_file(wav_path)
            wav_data = tf.audio.decode_wav(wav_loader, desired_channels=1)
            self.background_data.append(wav_data.audio.numpy().flatten())
        if not self.background_data:
            raise Exception('No background wav files were found in ' + search_path)

    def set_size(self, mode):
        return len(self.data_index[mode])

    def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift, mode):
    
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))
        # Data and labels will be populated and returned.
        data = np.zeros((sample_count, model_settings['fingerprint_size']))
        labels = np.zeros(sample_count)
        desired_samples = model_settings['desired_samples']
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')
        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in xrange(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]
            # If we're time shifting, set up the offset for this sample.
            if time_shift > 0:
                time_shift_amount = np.random.randint(-time_shift, time_shift)
            else:
                time_shift_amount = 0
            if time_shift_amount > 0:
                time_shift_padding = [[time_shift_amount, 0], [0, 0]]
                time_shift_offset = [0, 0]
            else:
                time_shift_padding = [[0, -time_shift_amount], [0, 0]]
                time_shift_offset = [-time_shift_amount, 0]
            # Choose a section of background noise to mix in.
            if use_background or sample['label'] == SILENCE_LABEL:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                if len(background_samples) <= model_settings['desired_samples']:
                    raise ValueError(
                        'Background sample is too short! Need more than %d'
                        ' samples but only %d were found' %
                        (model_settings['desired_samples'], len(background_samples)))
                background_offset = np.random.randint(
                    0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[background_offset:(
                    background_offset + desired_samples)]
                background_reshaped = background_clipped.reshape([desired_samples, 1])
                if sample['label'] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0, 1)
                elif np.random.uniform(0, 1) < background_frequency:
                    background_volume = np.random.uniform(0, background_volume_range)
                else:
                    background_volume = 0
            else:
                background_reshaped = np.zeros([desired_samples, 1])
                background_volume = 0
            
            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == SILENCE_LABEL:
                foreground_volume = 0
            else:
                foreground_volume = 1
            data_tensor = self.get_processing(model_settings, foreground_volume=foreground_volume, time_shift_padding=time_shift_padding, time_shift_offset=time_shift_offset, background_data=background_reshaped, background_volume=background_volume, wav_filename=sample['file'])
            # - We just need data_tensor

            if model_settings['preprocess'] == 'fbank':
                def compute_fbs(int16_wav_input):
                    fbs, energy = fbank(int16_wav_input, model_settings['sample_rate'],
                                    nfilt=int(model_settings['fingerprint_width'] / 3) - 1,
                                    winstep=model_settings['window_stride_samples'] / model_settings['sample_rate'],
                                    winlen=model_settings['window_size_samples'] / model_settings['sample_rate'],
                                    nfft=1024,
                                    lowfreq=64)
                    fbs = np.log(fbs)
                    energy = np.log(energy)
                    features = np.concatenate([fbs, energy[:, None]], axis=1)
                    # add derivatives:
                    get_delta = lambda v: np.concatenate([np.zeros((1, v.shape[1])), v[2:] - v[:-2], np.zeros((1, v.shape[1]))], axis=0)
                    d_features = get_delta(features)
                    d2_features = get_delta(d_features)

                    return np.concatenate([features, d_features, d2_features], axis=1)

                data_tensor = compute_fbs(data_tensor)

            data[i - offset, :] = data_tensor.numpy().flatten()
            label_index = self.word_to_index[sample['label']]
            labels[i - offset] = label_index

        # - End big for loop
        if self.n_repeat > 1:
            data = np.repeat(data, self.n_repeat, axis=1)

        if self.n_thr_spikes > 1:
            # GENERATE THRESHOLD CROSSING SPIKES
            # print(data.shape)
            num_thrs = self.n_thr_spikes
            thrs = np.linspace(0, 1, num_thrs)  # number of input neurons determines the resolution
            spike_stack = []
            for img in data:  # shape img = (3920)
                Sspikes = None
                for thr in thrs:
                    if Sspikes is not None:
                        Sspikes = np.concatenate((Sspikes, self.find_onset_offset(img, thr)))
                    else:
                        Sspikes = self.find_onset_offset(img, thr)
                Sspikes = np.array(Sspikes)  # shape Sspikes = (2*num_thrs-1, 3920)
                Sspikes = np.swapaxes(Sspikes, 0, 1)
                spike_stack.append(Sspikes)
            spike_stack = np.array(spike_stack)  # (64, 3920, 2*num_thrs-1)
            # print(spike_stack.shape)
            spike_stack = np.reshape(spike_stack, [sample_count, -1])  # (64, 74480) how_many, spec_time * 2*num_thrs-1
            # print(spike_stack.shape)
            data = spike_stack
        data = tf.cast(data, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int32)
        return data, labels


    def get_processing(self, model_settings, foreground_volume, time_shift_padding, time_shift_offset, background_data, background_volume, wav_filename):
        desired_samples = model_settings['desired_samples']
        wav_loader = io_ops.read_file(wav_filename)
        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
            
        scaled_foreground = tf.multiply(wav_decoder.audio,foreground_volume)
        # Shift the sample's start position, and pad any gaps with zeros.

        padded_foreground = tf.pad(tensor=scaled_foreground,paddings=time_shift_padding,mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,time_shift_offset,[desired_samples, -1])
        # Mix in background noise.
        
        background_mul = tf.cast(tf.multiply(background_data,background_volume), dtype=tf.float32)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.

        def periodic_hann_window(window_length, dtype):
            return 0.5 - 0.5 * tf.math.cos(2.0 * np.pi * tf.range(tf.cast(window_length, dtype=dtype), dtype=dtype) / tf.cast(window_length, dtype=dtype))

        signal_stft = tf.signal.stft(tf.transpose(background_clamp, [1, 0]),
                                    frame_length=model_settings['window_size_samples'],
                                    frame_step=model_settings['window_stride_samples'],
                                    window_fn=periodic_hann_window)
        signal_spectrograms = tf.abs(signal_stft)
        spectrogram = signal_spectrograms

        if model_settings['preprocess'] == 'average':
            final_output = tf.nn.pool(
                input=tf.expand_dims(spectrogram, -1),
                window_shape=[1, model_settings['average_window_width']],
                strides=[1, model_settings['average_window_width']],
                pooling_type='AVG',
                padding='SAME')
        elif model_settings['preprocess'] == 'fbank':
            int16_input = tf.cast(tf.multiply(background_clamp, 32768), tf.int16)
            final_output = int16_input
        elif model_settings['preprocess'] == 'mfcc':
            num_spectrogram_bins = signal_stft.shape[-1]
            num_mel_bins = num_mfccs = model_settings['fingerprint_width']
            log_noise_floor = 1e-12
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,model_settings['sample_rate'])
            mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(mel_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
            log_mel_spectrograms = tf.math.log(mel_spectrograms + log_noise_floor)
            signal_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]
            final_output = signal_mfccs
        else:
            raise ValueError('Unknown preprocess mode "%s" (should be "mfcc", '
                            ' "average", or "micro")' %
                            (model_settings['preprocess']))

        return final_output

    def find_onset_offset(self, y, threshold):
        if threshold == 1:
            equal = y == threshold
            transition_touch = np.where(equal)[0]
            touch_spikes = np.zeros_like(y)
            touch_spikes[transition_touch] = 1
            return np.expand_dims(touch_spikes, axis=0)
        else:
            # Find where y crosses the threshold (increasing).
            lower = y < threshold
            higher = y >= threshold
            transition_onset = np.where(lower[:-1] & higher[1:])[0]
            transition_offset = np.where(higher[:-1] & lower[1:])[0]
            onset_spikes = np.zeros_like(y)
            offset_spikes = np.zeros_like(y)
            onset_spikes[transition_onset] = 1
            offset_spikes[transition_offset] = 1
            return np.stack((onset_spikes, offset_spikes))