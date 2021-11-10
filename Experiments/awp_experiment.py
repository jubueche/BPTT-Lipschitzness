from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np
import seaborn as sns

seeds = [0]

class awp_experiment:
    
    @staticmethod
    def train_grid():

        ecg = [ecg_lsnn.make()]
        ecg = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})

        speech = [speech_lsnn.make()]
        speech = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})

        cnn_grid = [cnn.make()]
        cnn_grid = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry"})
        
        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "seed", seeds)
        return final_grid

    @staticmethod
    def visualize():

        speech_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]

        grid = [model for model in awp_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_mm = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid_mm = split(grid_mm, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid_mm = split(grid_mm, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid_mm = configure(grid_mm, {"mode":"direct", "n_iterations":100})    
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")