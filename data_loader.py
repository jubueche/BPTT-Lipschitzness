import os
import json
import soundfile
import warnings
import torch
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from typing import Dict, Callable, List, Optional
import datetime

def get_latest_model(base_path):
    models = os.listdir(base_path)
    dates = []
    for idx,model in enumerate(models):
        try:
            date_time_obj = datetime.datetime.strptime(model, '%Y-%m-%d-%H-%M-%S.model')
        except:
            continue
        else:
            dates.append((date_time_obj,idx))
    # - Sort in ascending order
    dates_sorted = sorted(dates, key= lambda x : x[0])
    if(len(dates_sorted) == 0):
        return None
    return models[dates_sorted[-1][1]]

def gen_label_map(data_list, key_words: Optional[List[str]] = None) -> Dict:
    """
    Generate a dict of all the labels in the data and a corresponding unique `int` id

    :param data_list:
    :return: dictionary of {`label`: id}
    """
    label_map = dict.fromkeys(data_list[:, 1])
    if key_words is None:
        for label_id, label in enumerate(label_map.keys()):
            label_map[label] = label_id
    else:
        null_class_label = len(key_words)
        for label in label_map.keys():
            try:
                label_id = key_words.index(label)
            except ValueError as e:
                label_id = null_class_label
            label_map[label] = label_id
    return label_map


def get_label_distribution(label_list, key_words: List[str]) -> (Dict, List):
    """
    Get the label distribution and the weights for learning on this data

    :param label_list:
    :param key_words:
    :return:
    """
    distribution = {k: list(label_list).count(k) for k in key_words}
    weights = [len(label_list) / distribution[k] for k in key_words]
    if sum(distribution.values()) != len(label_list):
        # Unknow class
        weights.append(len(label_list) / (len(label_list) - sum(distribution.values())))
    return distribution, weights


class AudioDataset(Dataset):
    def __init__(
        self,
        path: str = "/home/aiCTX Dropbox/Engineering/Datasets/TensorCommands/",
        config: str = "tensorcommands.json",
        data_partition: str = "train",
        label_map: Callable = gen_label_map,
        key_words: Optional[List] = None,
        transform: Optional = None,
        target_signal_transform: Optional = None,
        cache: Optional[str] = None,
        peak_min_height: float = 0.1,
        target_signal_duration: float = 0.4,
    ):
        """
        Load MyTensorCommands dataset

        :param path: Path to the dataset
        :param config: Name of the config file ie json file with all the paths to individual files and their labels
        :param data_partition: train/test/validation set to load
        :param label_map: a callable that maps the labels of the dataset to ids. This method is used to build a dict `label_map`, which is then used to generate int labels for the dataset
        :param key_words: A list of labels that are to be given unique ids for training, all data points that don't belong to the given labels will be labeled as null class.
        :param transform: callable to transform the data
        :param target_signal_transform: callable to transform signal
        :param cache: If a path is given, the data is saved/loaded from the path specified by cache
        :param peak_min_height: minimum height of a peak for the target signal generation
        :param target_signal_duration: 0.4ms by default
        """
        super().__init__()
        self.transform = transform
        self.target_signal_transform = target_signal_transform
        with open(os.path.join(path, config), "r") as f:
            dataset = json.load(f)
        self.path = path
        self.dataset = dataset
        self.data_partition = data_partition

        self.data_list = np.array(dataset.get(data_partition, None))
        self.label_map = label_map(self.data_list, key_words)
        self.cache = cache
        # Parameters for target signal generation
        self.peak_min_height = peak_min_height
        self.target_signal_duration = target_signal_duration

    def __getitem__(self, indx):
        data_fname, label = self.data_list[indx]
        label_idx = self.label_map[label]
        if self.cache:
            # Acquire signal
            try:
                signal = self.get_cached_signal(indx)
            except FileNotFoundError as e:
                warnings.warn(
                    f"{self.__class__}: Cached file not found for sample {data_fname}. Regenerating it."
                )
                signal = self.save_signal_to_cache(indx)
            # Acquire target
            try:
                target_signal = self.get_cached_target_signal(indx)
            except FileNotFoundError:
                warnings.warn(
                    f"{self.__class__}: Cached file not found for target {data_fname}. Regenerating it."
                )
                target_signal = self.save_target_signal_to_cache(indx)
            return signal, label_idx, target_signal
        else:
            raw_signal, sampling_freq = self.load_raw_sample(data_fname)
            target_signal = self.gen_target_signal(raw_signal, sampling_freq)
            if self.transform:
                signal = self.transform(raw_signal)
            else:
                signal = raw_signal
            return signal, label_idx, target_signal

    def save_signal_to_cache(self, indx):
        data_fname, _ = self.data_list[indx]
        signal, sampling_freq = self.load_raw_sample(data_fname)
        if self.transform:
            signal = self.transform(signal)
        data_fname_cached = Path(self.cache) / Path(data_fname).with_suffix(".npy")
        if not data_fname_cached.parents[0].exists():
            # Create subdirectories required
            data_fname_cached.parents[0].mkdir(parents=True)
        np.save(data_fname_cached, signal)
        return signal

    def get_cached_signal(self, indx):
        data_fname, _= self.data_list[indx]
        data_fname_cached = Path(self.cache) / Path(data_fname).with_suffix(".npy")
        signal = np.load(data_fname_cached)
        return signal

    def __len__(self):
        return len(self.data_list)

    def load_raw_sample(self, filename):
        signal, sampling_freq = soundfile.read(os.path.join(self.path, filename))
        return signal, sampling_freq

    def save_target_signal_to_cache(self, indx):
        data_fname, _= self.data_list[indx]
        target_signal_fname_cached = Path(str(Path(self.cache) / Path(data_fname).with_suffix("")) + "_target.npy")
        signal, sampling_freq = self.load_raw_sample(data_fname)
        target_signal = self.gen_target_signal(signal, sampling_freq)
        if self.target_signal_transform:
            target_signal = self.target_signal_transform(target_signal)
        if not target_signal_fname_cached.parents[0].exists():
            # Create subdirectories required
            target_signal_fname_cached.parents[0].mkdir(parents=True)
        np.save(target_signal_fname_cached, target_signal)
        return target_signal

    def get_cached_target_signal(self, indx):
        data_fname, _= self.data_list[indx]
        target_signal_fname_cached = Path(str(Path(self.cache) / Path(data_fname).with_suffix("")) + "_target.npy")
        target_signal = np.load(target_signal_fname_cached)
        return target_signal

    def gen_target_signal(self, signal, sampling_freq):
        # normalization
        signal = signal - np.mean(signal)
        signal /= np.max(np.abs(signal))

        # create target signal
        power = signal ** 2
        peaks = find_peaks(power, height=self.peak_min_height)[0]

        # no peak
        if len(peaks) == 0:
            t_mean = len(signal)
        else:
            t_mean = peaks[-1]

        t_half_width = int(sampling_freq * self.target_signal_duration / 2.0)
        if t_mean + t_half_width >= len(signal):
            t_mean = len(signal) - t_half_width

        if t_mean - t_half_width < 0:
            t_mean = t_half_width

        t_tgt_start = t_mean - t_half_width
        t_tgt_stop = t_mean + t_half_width

        target_signals = np.zeros(len(signal))
        target_signals[t_tgt_start:t_tgt_stop] = 1

        return target_signals


class ClassWeightedRandomSampler(Sampler):
    def __init__(self, labels, class_weights, n_samples=None):
        """
        Given weights of all the classes in the dataset, sample the dataset based on the weights
        If weights are are 1, then you have uniform sampling independent of the distribution of the dataset

        :param dataset:
        :param class_weights:
        :param n_samples: 
        """
        self.weights = class_weights / sum(class_weights)
        self.labels = torch.tensor(labels)

        # Get indices of all labels
        self.list_of_indices = [
            torch.where(self.labels == l)[0] for l in range(len(class_weights))
        ]

        # Determine total number of samples
        if n_samples is None:
            # Determine the class with the maximum number of samples
            n_sample_max = max([len(x) for x in self.list_of_indices])
            self.n_samples = n_sample_max * len(class_weights)
        else:
            self.n_samples = n_samples

        # Given n samples, determine howmany samples from each class
        self.num_samples_per_class = (self.weights * self.n_samples).int()

    def __iter__(self):
        # Given n samples in each class, determine which indices these samples should be
        # TODO: This random picking of elements does not guarantee that all the samples will be used.
        # FIXME
        list_of_sample_indices = []
        for label, l_indx in enumerate(self.list_of_indices):
            n = self.num_samples_per_class[label]
            indices = l_indx[torch.randint(0, len(l_indx), (n,))]
            list_of_sample_indices.append(indices)

        # Add all these indices
        all_indices = torch.cat(list_of_sample_indices)

        # Shuffle indices
        x = all_indices.numpy()
        np.random.shuffle(x)
        all_indices = torch.from_numpy(x)

        return iter(all_indices)

    def __len__(self):
        return self.n_samples
