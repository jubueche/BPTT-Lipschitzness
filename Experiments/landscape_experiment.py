from architectures import speech_lsnn, ecg_lsnn, cnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib as mpl

class landscape_experiment:

    @staticmethod
    def train_grid():
        betas = [0.0,0.05,0.125,0.25,0.5]
        seeds = [0]
        grid_speech = speech_lsnn.make()
        grid_speech = configure([grid_speech], dictionary={"attack_size_mismatch":0.1})
        grid_speech = split(grid_speech, "seed", seeds)
        grid_speech = split(grid_speech, "beta_robustness", betas)

        grid_ecg = ecg_lsnn.make()
        grid_ecg = configure([grid_ecg], dictionary={"attack_size_mismatch":0.1})
        grid_ecg = split(grid_ecg, "seed", seeds)
        grid_ecg = split(grid_ecg, "beta_robustness", betas)

        grid_cnn = cnn.make()
        grid_cnn = configure([grid_cnn], dictionary={"attack_size_mismatch":0.1})
        grid_cnn = split(grid_cnn, "seed", seeds)
        grid_cnn = split(grid_cnn, "beta_robustness", betas)

        return grid_speech + grid_ecg + grid_cnn

    @staticmethod
    def visualize():
        seeds = [0]
        betas = [0.0,0.05,0.125,0.25,0.5]
        colors = ["#4c84e6","#fc033d","#03fc35","#77fc03","#f803fc"]
        grid = [model for model in landscape_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        num_steps = 10
        std = 0.2

        grid = run(grid, get_landscape_sweep, n_threads=1, store_key="landscape")("{*}", num_steps, "{data_dir}", std)

        def get_data(arch):
            data_dict = {}
            for beta in betas:
                data_tmp = query(grid, "landscape", where={"beta_robustness":beta, "architecture":arch})
                data_dict[beta] = data_tmp
            return data_dict

        data_speech = get_data(arch="speech_lsnn")
        data_ecg = get_data(arch="ecg_lsnn")
        data_cnn = get_data(arch="cnn")

        fig = plt.figure(figsize=(10,3), constrained_layout=True)
        axes_speech = plt.subplot(1,3,1) # - Speech
        axes_speech.set_xlabel(r"$\alpha$")
        axes_speech.set_ylabel("Loss")
        axes_speech.spines['right'].set_visible(False)
        axes_speech.spines['top'].set_visible(False)

        axes_ecg = plt.subplot(1,3,2)
        axes_ecg.spines['right'].set_visible(False)
        axes_ecg.spines['top'].set_visible(False)

        axes_cnn = plt.subplot(1,3,3)
        axes_cnn.spines['right'].set_visible(False)
        axes_cnn.spines['top'].set_visible(False)

        for beta_idx,beta in enumerate(betas):
            data_beta_speech = data_speech[beta]
            data_beta_ecg = data_ecg[beta]
            data_beta_cnn = data_cnn[beta]
            for idx_d,d in enumerate(data_beta_speech):
                label = None
                if idx_d == 0:
                    label = ("%s" % str(beta))               
                axes_speech.plot(onp.linspace(-1,1,num_steps), d, c=colors[beta_idx], alpha=0.5, label=label)
            
            for idx_d,d in enumerate(data_beta_ecg):
                axes_ecg.plot(onp.linspace(-1,1,num_steps), d, c=colors[beta_idx], alpha=0.5)

            for idx_d,d in enumerate(data_beta_cnn):
                axes_cnn.plot(onp.linspace(-1,1,num_steps), d, c=colors[beta_idx], alpha=0.5)

        plt.title("Weight loss landscape")
        plt.legend()
        plt.savefig("Resources/Figures/landscape.pdf", dpi=1200)
        plt.plot()