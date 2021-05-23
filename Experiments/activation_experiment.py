from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from TensorCommands.data_loader import SpeechDataLoader
from experiment_utils import *
import numpy as onp
import matplotlib as mpl
mpl.rcParams["hist.bins"] = 100
mpl.rcParams["lines.linewidth"] = 2.0
import matplotlib.pyplot as plt
from datajuicer.visualizers import METHOD_COLORS, METHOD_LINESTYLE, METHOD_LINEWIDTH
from jax import config
config.FLAGS.jax_log_compiles=True
config.update('jax_disable_jit', False)

class activation_experiment:

    @staticmethod
    def train_grid():
        grid = speech_lsnn.make()
        grid1 = configure([grid], dictionary={"beta_robustness":0.0})
        grid2 = configure([grid], dictionary={"beta_robustness":0.5, "noisy_forward_std":0.3, "attack_size_mismatch":0.1})
        grid3 = configure([grid], dictionary={"beta_robustness":0.0, "noisy_forward_std":0.3})
        return grid1 + grid2 + grid3

    @staticmethod
    def visualize():
        grid = [model for model in activation_experiment.train_grid()]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        zetas = [0.0,0.00001,0.00005,0.0001,0.0005,0.001,0.005]
        grid = split(grid, "zeta", zetas)

        grid = run(grid, get_IBP_test_acc, n_threads=1, store_key="interval_accuracy")("{*}","{zeta}","{data_dir}")

        test_accuracies_standard = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.0})
        test_accuracies_robust = query(grid, "interval_accuracy", where={"beta_robustness":0.5, "noisy_forward_std":0.3})
        test_accuracies_robust_no_adversary = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.3})

        model_standard = query(grid, "network", where={"beta_robustness":0.0, "noisy_forward_std":0.0})[0]
        model_robust = query(grid, "network", where={"beta_robustness":0.5})[0]
        theta_standard = query(grid, "theta", where={"beta_robustness":0.0, "noisy_forward_std":0.0})[0]
        theta_robust = query(grid, "theta", where={"beta_robustness":0.5})[0]
        loader = SpeechDataLoader(path=query(grid, "data_dir", where={"beta_robustness":0.0})[0], batch_size=1)
        X,_ = loader.get_batch("test")
        rnn_out_robust, output_dic_robust = model_robust.call_verbose(X, model_robust.unmasked(), **theta_robust)
        rnn_out_standard, output_dic_standard = model_standard.call_verbose(X, model_standard.unmasked(), **theta_standard)

        _, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(10,3))
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        ax = axes[0]
        ax.plot(test_accuracies_robust, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Beta + Forward")
        ax.plot(test_accuracies_standard, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.plot(test_accuracies_robust_no_adversary, color=METHOD_COLORS["Forward Noise"], linewidth=METHOD_LINEWIDTH["Forward Noise"], linestyle=METHOD_LINESTYLE["Forward Noise"], label="Forward Noise")
        ax.set_xticks(onp.arange(0,len(zetas),1))
        ax.set_xticklabels(zetas)
        ax.set_xlabel(r"Attack size $\zeta$")
        ax.set_ylabel("Verified test acc.")
        ax.legend()

        ax = axes[1]
        ax.hist(output_dic_robust["V"].flatten(), density=True, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Robust")
        ax.hist(output_dic_standard["V"].flatten(), density=True, alpha=0.5, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.set_ylabel("Normalized bin count")
        ax.set_xlabel("Membrane potential")
        ax.legend()

        plt.show()