from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import dj, split, configure, query, djm
from experiment_utils import *

class ecg_experiment:
    
    @staticmethod
    def train_grid():
        ecg = ecg_lsnn.make()
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "beta_robustness", [0.0,0.1,0.2,0.3,0.4,0.5])
        ecg = split(ecg, "seed", [0])
        return ecg

    @staticmethod
    def visualize():
        pass