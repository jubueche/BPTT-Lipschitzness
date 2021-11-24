from architectures import speech_lsnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
import numpy as np
import matplotlib.pyplot as plt

seeds = [0]
class awp_sweep:
    
    @staticmethod
    def train_grid():

        cnn_grid = [cnn.make()]
        cnn_awp = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry", "eps_pga":0.0, "nb_iter":3})

        ecg = [ecg_lsnn.make()]
        ecg_awp = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1, "eps_pga":0.0, "nb_iter":3})

        speech = [speech_lsnn.make()]
        speech_awp = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1, "eps_pga":0.0, "nb_iter":3})

        final_grid = ecg_awp+speech_awp
        final_grid = split(final_grid, "seed", seeds)
        final_grid = split(final_grid, "attack_size_mismatch", [0.01, 0.03, 0.05, 0.1])
        return final_grid

    @staticmethod
    def visualize():

        pass

