from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np

class noisy_forward_experiment:
    
    @staticmethod
    def train_grid():

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg = ecg0

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        speech = speech0

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        cnn_grid = cnn_grid0

        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "beta_robustness", [0.1,0.5])

        return final_grid
