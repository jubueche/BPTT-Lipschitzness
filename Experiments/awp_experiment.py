from architectures import speech_lsnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
import numpy as np
import matplotlib.pyplot as plt

seeds = [0]
eps_pgas = [0.01,0.1,1.0]
eps_pgas_cnn = [0.001,0.01,0.1]
nb_iter = 3

class awp_experiment:
    
    @staticmethod
    def train_grid():

        cnn_grid = [cnn.make()]
        cnn_awp = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry"})
        cnn_awp_eps = split(cnn_awp, "eps_pga", eps_pgas_cnn)
        cnn_awp_eps = configure(cnn_awp_eps, {"nb_iter":3})

        ecg = [ecg_lsnn.make()]
        ecg_awp = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg_awp_eps = split(ecg_awp, "eps_pga", eps_pgas)
        ecg_awp_eps = configure(ecg_awp_eps, {"nb_iter":3})

        speech = [speech_lsnn.make()]
        speech_awp = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        speech_awp_eps = split(speech_awp, "eps_pga", eps_pgas)
        speech_awp_eps = configure(speech_awp_eps, {"nb_iter":3})

        final_grid = ecg_awp+ecg_awp_eps+speech_awp+speech_awp_eps+cnn_awp+cnn_awp_eps
        final_grid = split(final_grid, "seed", seeds)
        return final_grid

    @staticmethod
    def visualize():

        title_dict = {
            "speech_lsnn": "Speech LSNN",
            "ecg_lsnn": "ECG LSNN",
            "cnn": "FMNIST CNN"
        }

        architectures = ["speech_lsnn","ecg_lsnn","cnn"]
        mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]

        grid = [model for model in awp_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_mm = split(grid, "mm_level", mm_levels)

        grid_mm = configure(grid_mm, {"mode":"direct", "n_iterations":100})    
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")\
            ("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        _ = plt.figure(figsize=(13,4), constrained_layout=False)
        for i,arch in enumerate(architectures):
            eps_pgas_arch = eps_pgas_cnn if arch == "cnn" else eps_pgas
            ax = plt.subplot(1, len(architectures), i+1)
            
            ax.set_ylabel("Test acc.")
            ax.set_xlabel("Mismatch level")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(title_dict[arch])

            eps_pgas_list = [0] + eps_pgas_arch
            for eps_pga in eps_pgas_list:
                accs = np.empty(shape=(len(mm_levels),))
                stds = np.empty(shape=(len(mm_levels),))
                for idx,mm_level in enumerate(mm_levels):
                    accuracy_list = query(grid_mm, "mismatch_list", where={"architecture":arch, "eps_pga":eps_pga, "mm_level":mm_level})
                    accs[idx] = np.mean(accuracy_list)
                    stds[idx] = np.std(accuracy_list)
                ax.errorbar(x=mm_levels, y=accs, yerr=stds, label=r"$\epsilon_{pga}=$" + ("%.3f"%eps_pga))
        
            ax.legend()


        plt.savefig("Resources/Figures/AWP_sweep.pdf", dpi=1200)
        plt.show()
            

