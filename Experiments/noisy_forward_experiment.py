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
        seeds = [0]
        initial_std_mismatch = [0.3]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0, "noisy_forward":True})
        ecg = ecg0

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0, "noisy_forward":True})
        speech = speech0

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0, "noisy_forward":True, "learning_rate":"0.0001,0.00001"})
        cnn_grid = cnn_grid0

        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "initial_std_mismatch", initial_std_mismatch)
        final_grid = split(final_grid, "seed", seeds)

        return final_grid
