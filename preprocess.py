from scipy.signal import butter, sosfilt
import numpy as np
from typing import Optional


def hz2mel(x: float):
    """Takes value from hz and returns mel
    """
    return 2595 * np.log10(1 + x / 700)


def mel2hz(x: float):
    """Takes value from mel and returns hz
    """
    return 700 * (10 ** (x / 2595) - 1)


class ButterMel:
    def __init__(
        self,
        sampling_freq: float = 16000,
        cutoff_freq: float = 100,
        num_filters: int = 64,
        order: int = 2,
        downsample: Optional[int] = None,
    ):
        """
        Filterbank of Butterworth filters in Mel-scale

        :param sampling_freq: sampling frequency of the signal in Hertz,
        defaults to 16000
        :type sampling_freq: int, optional
        :param cutoff_freq: cutoff frequency for lowpass filtering, defaults to 100
        :param downsample: downsampling of the filtered signal, defaults to (sampling_freq/cutoff_freq)
        :type cutoff_freq: float, optional
        :param num_filters: number of filters, defaults to 64
        :type num_filters: int, optional
        :param order: order of the Butterworth filters, defaults to 2
        :type order: int, optional
        """
        self.sampling_freq = sampling_freq
        self.cutoff_freq = cutoff_freq
        self.num_filters = num_filters
        self.order = order

        self.filter_bandwidth = 5 / num_filters
        self.nyquist = sampling_freq / 2
        if downsample:
            self.downsample = downsample
        else:
            self.downsample = int(sampling_freq / cutoff_freq)

        low_freq = hz2mel(cutoff_freq)
        high_freq = hz2mel(sampling_freq / 2 / (1 + self.filter_bandwidth) - 1)
        self.freqs = mel2hz(np.linspace(low_freq, high_freq, num_filters))

        self.freq_bands = (
            np.array([self.freqs, self.freqs * (1 + self.filter_bandwidth)])
            / self.nyquist
        )
        self.filters = list(
            map(
                lambda fb: butter(order, fb, analog=False, btype="band", output="sos"),
                self.freq_bands.T,
            )
        )
        self.filter_lowpass = butter(
            3, cutoff_freq / self.nyquist, analog=False, btype="low", output="sos"
        )

    def __call__(self, signals):
        # signals, filter_lowpass, downsample
        filters_output = []
        for f in self.filters:
            sig = sosfilt(f, signals)
            sig = np.abs(sig)
            sig = sosfilt(self.filter_lowpass, sig)[:: self.downsample]
            filters_output.append(sig)
        filters_output = np.array(filters_output)
        return filters_output


class StandardizeDataLength:
    def __init__(self, data_length=16000):
        """
        Zero pad the data so as to ensure standard data length

        :param data_length:
        """
        self.data_length = data_length

    def __call__(self, signal):
        if len(signal) > self.data_length:
            return NotImplementedError
        elif len(signal) < self.data_length:
            return np.concatenate((signal, np.zeros(self.data_length - len(signal))))
        else:
            return signal


class Subsample:
    def __init__(self,downsample=160):
        """
        Subsample data
        """
        self.downsample = downsample

    def __call__(self, signal):
        return signal[..., ::self.downsample]
