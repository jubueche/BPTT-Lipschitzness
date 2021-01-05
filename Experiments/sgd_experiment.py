from architectures import speech_lsnn
from datajuicer import dj, split, configure, query, djm
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class sgd_experiment:
    
    @staticmethod
    def train_grid():
        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0])
        speech = split(speech, "eval_step_interval", [1000])
        speech = split(speech, "learning_rate", ["0.01,0.001"])
        speech = split(speech, "optimizer", ["sgd"])
        speech1 = split(speech, "batch_size", [8])
        speech2 = split(speech, "batch_size", [128])
        speech1 = split(speech1, "n_epochs",["200,56"])
        speech2 = split(speech2, "n_epochs", ["200,88"])
        speech = speech1 + speech2
        speech = split(speech, "seed", [0,1,2,3,4,5,6,7,8,9])
        return speech

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.5,0.7,0.9,1.1,1.5]
        seeds = [0]
        grid = [model for model in sgd_experiment.train_grid() if model["seed"] in seeds] 
        grid = djm(grid, "train", run_mode="load")("{*}")

        grid = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})

        grid = configure(grid, {"n_iterations":50})
        grid = configure(grid, {"n_iterations":1}, where={"mm_level":0.0})
        grid = configure(grid, {"mode":"direct"})
        
        grid = djm(grid, get_mismatch_list, n_threads=1, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}", 100)
