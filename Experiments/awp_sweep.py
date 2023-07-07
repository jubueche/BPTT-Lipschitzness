from architectures import speech_lsnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
import numpy as np
import matplotlib.pyplot as plt

seeds = [1]
class awp_sweep:
    
    @staticmethod
    def train_grid():

        cnn_grid = [cnn.make()]
        cnn_awp = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry", "eps_pga":0.0, "nb_iter":3})

        ecg = [ecg_lsnn.make()]
        ecg_awp = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1, "eps_pga":0.0, "nb_iter":3})

        speech = [speech_lsnn.make()]
        speech_awp = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1, "eps_pga":0.0, "nb_iter":3})

        final_grid = speech_awp + ecg_awp + cnn_awp
        final_grid = split(final_grid, "seed", seeds)
        final_grid = split(final_grid, "attack_size_mismatch", [0.01, 0.03, 0.05, 0.1])
        return final_grid

    @staticmethod
    def visualize():
        label_dict = {
            "beta_robustness": r"beta",
            "optimizer": "Optimizer",
            "mismatch_list_mean": "Mean Acc.",
            "mismatch_list_std":"Std.",
            "mismatch_list_min":"Min.",
            "dropout_prob":"Dropout",
            "mm_level": "Mismatch",
            "cnn" : "CNN",
            "speech_lsnn": "Speech LSNN",
            "ecg_lsnn": "ECG LSNN",
            "awp": "AWP",
            "AWP = True":"AWP",
            "noisy_forward_std": r"Forward",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD",
            "attack_size_constant" : "Attack"
        }

        speech_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]

        grid = [model for model in awp_sweep.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_mm = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid_mm = split(grid_mm, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid_mm = split(grid_mm, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid_mm = configure(grid_mm, {"n_iterations":100})
        grid_mm = configure(grid_mm, {"n_iterations":1}, where={"mm_level":0.0})
        grid_mm = configure(grid_mm, {"mode":"direct"})        
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=20, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        group_by = ["architecture", "awp", "mm_level", "attack_size_mismatch"]
        for g in grid_mm:
            g["mismatch_list"] = list(100 * np.array(g["mismatch_list"])) 
        reduced = reduce_keys(grid_mm, "mismatch_list", reduction={"mean": lambda l: float(np.mean(l)), "std": lambda l: float(np.std(l)), "min": lambda l: float(np.min(l))}, group_by=group_by)

        independent_keys = ["architecture",Table.Deviation_Var({"awp":False,"attack_size_mismatch":0.0}, label="Method"),  "mm_level"]
        dependent_keys = ["mismatch_list_mean", "mismatch_list_std","mismatch_list_min"]

        print(latex(reduced, independent_keys, dependent_keys, bold_order=[max,min,max], label_dict= label_dict))