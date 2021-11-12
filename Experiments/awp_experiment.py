from architectures import speech_lsnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
import numpy as np
import matplotlib.pyplot as plt

seeds = [0]
eps_pgas = [0.01,0.1,1.0]
nb_iter = 3

class awp_experiment:
    
    @staticmethod
    def train_grid():

        ecg = [ecg_lsnn.make()]
        ecg_awp = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg_awp_eps = split(ecg_awp, "eps_pga", eps_pgas)
        ecg_awp_eps = configure(ecg_awp_eps, {"nb_iter":3})

        speech = [speech_lsnn.make()]
        speech_awp = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        speech_awp_eps = split(speech_awp, "eps_pga", eps_pgas)
        speech_awp_eps = configure(speech_awp_eps, {"nb_iter":3})

        final_grid = ecg_awp + ecg_awp_eps + speech_awp + speech_awp_eps
        final_grid = split(final_grid, "seed", seeds)
        return final_grid

    @staticmethod
    def visualize():

        speech_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]

        grid = [model for model in awp_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_mm = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})

        grid_mm = configure(grid_mm, {"mode":"direct", "n_iterations":100})    
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")\
            ("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        fig = plt.figure(figsize=(5,5), constrained_layout=False)
        ax = plt.gca()
        ax.set_ylabel("Test acc. (%)")
        ax.set_xlabel("Mismatch level")

        eps_pgas_list = [0] + eps_pgas
        for eps_pga in eps_pgas_list:
            accs = np.empty(shape=(len(speech_mm_levels),))
            stds = np.empty(shape=(len(speech_mm_levels),))
            for idx,mm_level in enumerate(speech_mm_levels):
                accuracy_list = query(grid_mm, "mismatch_list", where={"eps_pga":eps_pga, "mm_level":mm_level})
                accs[idx] = np.mean(accuracy_list)
                stds[idx] = np.std(accuracy_list)
            ax.errorbar(x=speech_mm_levels, y=accs, yerr=stds, label=r"$\epsilon_{pga}=$" + ("%.2f"%eps_pga))
        
        plt.legend()
        plt.show()
            

