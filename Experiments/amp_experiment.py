from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
import numpy as np

class amp_experiment:
    
    @staticmethod
    def train_grid():
        seeds = [0,1]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        ecg2 = configure(ecg, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg3 = configure(ecg, {"beta_robustness":9999, "treat_as_constant":True, "attack_size_mismatch":0.0, "attack_size_constant":0.001, "boundary_loss":"madry", "p_norm":"2"})
        ecg3 = split(ecg3, "attack_size_constant", [0.02, 0.01, 0.005, 0.001, 0.0005]) #0.02 is training
        ecg = ecg0 + ecg1 + ecg2 + ecg3

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1})
        speech2 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        speech3 = configure(speech, {"beta_robustness":9999, "treat_as_constant":True, "attack_size_mismatch":0.0, "attack_size_constant":0.001, "boundary_loss":"madry", "p_norm":"2"})
        speech3 = split(speech3, "attack_size_constant", [0.02, 0.01, 0.005, 0.001, 0.0005]) #0.02 is training
        speech = speech0 + speech1 + speech2 + speech3

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        cnn_grid3 = configure(cnn_grid, {"beta_robustness":9999, "treat_as_constant":True, "attack_size_mismatch":0.0, "attack_size_constant":0.01, "boundary_loss":"madry", "p_norm":"2"})
        cnn_grid3 = split(cnn_grid3, "attack_size_constant", [0.02, 0.01, 0.005, 0.001, 0.0005])
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3

        final_grid = ecg3 + speech3 + cnn_grid3
        final_grid = split(final_grid, "seed", seeds)

        return final_grid
    
    @staticmethod
    def visualize():
        label_dict = {
            "beta_robustness": "Beta",
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
            "Beta = 0.25":"Beta 0.25",
            "Beta = 0.5":"Beta 0.5",
            "Beta = 0.1":"Beta 0.1",
            "Beta 0.25, Forward Noise": "Forward Noise + Beta 0.25",
            "Beta 0.5, Forward Noise": "Forward Noise + Beta 0.5",
            "Beta 0.1, Forward Noise": "Forward Noise + Beta 0.1",
            "noisy_forward_std = 0.3": "Forward Noise",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD",
            "attack_size_constant" : "Attack"
        }

        speech_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        seeds = [0,1]

        grid = [model for model in amp_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", store_key="*")("{*}")

        grid_mm = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid_mm = split(grid_mm, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid_mm = split(grid_mm, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid_mm = configure(grid_mm, {"n_iterations":100})
        grid_mm = configure(grid_mm, {"n_iterations":1}, where={"mm_level":0.0})
        grid_mm = configure(grid_mm, {"mode":"direct"})        
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        group_by = ["architecture", "awp", "beta_robustness", "dropout_prob", "optimizer", "noisy_forward_std", "mm_level", "attack_size_constant"]
        for g in grid_mm:
            g["mismatch_list"] = list(100 * np.array(g["mismatch_list"])) 
        reduced = reduce_keys(grid_mm, "mismatch_list", reduction={"mean": lambda l: float(np.mean(l)), "std": lambda l: float(np.std(l)), "min": lambda l: float(np.min(l))}, group_by=group_by)

        independent_keys = ["architecture",Table.Deviation_Var({"beta_robustness":9999, "awp":False, "dropout_prob":0.0, "optimizer":"adam", "noisy_forward_std":0.0, "attack_size_constant":0.0}, label="Method"),  "mm_level"]
        dependent_keys = ["mismatch_list_mean", "mismatch_list_std","mismatch_list_min"]

        print(latex(reduced, independent_keys, dependent_keys, bold_order=[max,min,max], label_dict= label_dict))