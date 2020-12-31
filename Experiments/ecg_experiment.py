from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *

class ecg_experiment:
    
    @staticmethod
    def train_grid():
        ecg = ecg_lsnn.make()
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [1.0,2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "n_epochs", ["64,16"])
        ecg = split(ecg, "beta_robustness", [0.7,0.9,1.1,1.3])
        ecg = split(ecg, "seed", [0])
        return ecg

    @staticmethod
    def visualize():
        pass