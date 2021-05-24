from architectures import speech_lsnn, ecg_lsnn
from datajuicer import run, split, configure, query
from TensorCommands.data_loader import SpeechDataLoader
from ECG.ecg_data_loader import ECGDataLoader
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

class IBP_experiment:

    @staticmethod
    def train_grid():
        grid = [speech_lsnn.make()]
        grid0 = configure(grid, {"beta_robustness":0.0})
        grid1 = configure(grid, {"beta_robustness":0.5, "noisy_forward_std":0.3, "attack_size_mismatch":0.1})
        grid2 = configure(grid, {"beta_robustness":0.0, "noisy_forward_std":0.3})
        speech = grid0 + grid1 + grid2

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        ecg2 = configure(ecg, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg = ecg0 + ecg1 + ecg2

        return speech + ecg

    @staticmethod
    def visualize():
        grid = [model for model in IBP_experiment.train_grid()]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        zetas = [0.0,0.00001,0.00005,0.0001,0.0005,0.001]
        grid = split(grid, "zeta", zetas)

        grid = run(grid, get_IBP_test_acc, n_threads=1, store_key="interval_accuracy")("{*}","{zeta}","{data_dir}")

        test_accuracies_standard_speech = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"speech_lsnn"})
        test_accuracies_robust_speech = query(grid, "interval_accuracy", where={"beta_robustness":0.5, "noisy_forward_std":0.3, "architecture":"speech_lsnn"})
        test_accuracies_robust_no_adversary_speech = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.3, "architecture":"speech_lsnn"})

        test_accuracies_standard_ecg = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"ecg_lsnn"})
        test_accuracies_robust_ecg = query(grid, "interval_accuracy", where={"beta_robustness":0.1, "attack_size_mismatch":0.1, "noisy_forward_std":0.3, "architecture":"ecg_lsnn"})
        test_accuracies_robust_no_adversary_ecg = query(grid, "interval_accuracy", where={"beta_robustness":0.0, "noisy_forward_std":0.3, "architecture":"ecg_lsnn"})

        model_standard_speech = query(grid, "network", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"speech_lsnn"})[0]
        model_robust_speech = query(grid, "network", where={"beta_robustness":0.5, "architecture":"speech_lsnn"})[0]
        theta_standard_speech = query(grid, "theta", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"speech_lsnn"})[0]
        theta_robust_speech = query(grid, "theta", where={"beta_robustness":0.5, "architecture":"speech_lsnn"})[0]
        loader_speech = SpeechDataLoader(path=query(grid, "data_dir", where={"beta_robustness":0.0, "architecture":"speech_lsnn"})[0], batch_size=1)
        X_speech,_ = loader_speech.get_batch("test")
        _, output_dic_robust_speech = model_robust_speech.call_verbose(X_speech, model_robust_speech.unmasked(), **theta_robust_speech)
        _, output_dic_standard_speech = model_standard_speech.call_verbose(X_speech, model_standard_speech.unmasked(), **theta_standard_speech)

        model_standard_ecg = query(grid, "network", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"ecg_lsnn"})[0]
        model_robust_ecg = query(grid, "network", where={"beta_robustness":0.1, "noisy_forward_std":0.3, "attack_size_mismatch":0.1, "architecture":"ecg_lsnn"})[0]
        theta_standard_ecg = query(grid, "theta", where={"beta_robustness":0.0, "noisy_forward_std":0.0, "architecture":"ecg_lsnn"})[0]
        theta_robust_ecg = query(grid, "theta", where={"beta_robustness":0.1, "noisy_forward_std":0.3, "attack_size_mismatch":0.1, "architecture":"ecg_lsnn"})[0]
        loader_ecg = ECGDataLoader(path=query(grid, "data_dir", where={"beta_robustness":0.0, "architecture":"ecg_lsnn"})[0], batch_size=1)
        X_ecg,_ = loader_ecg.get_batch("test")
        _, output_dic_robust_ecg = model_robust_ecg.call_verbose(X_ecg, model_robust_ecg.unmasked(), **theta_robust_ecg)
        _, output_dic_standard_ecg = model_standard_ecg.call_verbose(X_ecg, model_standard_ecg.unmasked(), **theta_standard_ecg)

        _, axes = plt.subplots(ncols=4, nrows=1, constrained_layout=True,figsize=(14,3))
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        ax = axes[0]
        ax.plot(test_accuracies_robust_speech, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Beta + Forward")
        ax.plot(test_accuracies_standard_speech, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.plot(test_accuracies_robust_no_adversary_speech, color=METHOD_COLORS["Forward Noise"], linewidth=METHOD_LINEWIDTH["Forward Noise"], linestyle=METHOD_LINESTYLE["Forward Noise"], label="Forward Noise")
        ax.set_xticks(onp.arange(0,len(zetas),1))
        ax.set_xticklabels(zetas)
        ax.set_xlabel(r"Attack size $\zeta$")
        ax.set_ylabel("Verified test acc.")
        ax.set_title("SRNN Speech")
        ax.legend()

        ax = axes[1]
        ax.hist(output_dic_robust_speech["V"].flatten(), density=True, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Robust")
        ax.hist(output_dic_standard_speech["V"].flatten(), density=True, alpha=0.5, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.set_ylabel("Normalized bin count")
        ax.set_xlabel("Membrane potential")
        ax.legend()

        ax = axes[2]
        ax.plot(test_accuracies_robust_ecg, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Beta + Forward")
        ax.plot(test_accuracies_standard_ecg, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.plot(test_accuracies_robust_no_adversary_ecg, color=METHOD_COLORS["Forward Noise"], linewidth=METHOD_LINEWIDTH["Forward Noise"], linestyle="dotted", label="Forward Noise")
        ax.set_xticks(onp.arange(0,len(zetas),1))
        ax.set_xticklabels(zetas)
        ax.set_xlabel(r"Attack size $\zeta$")
        ax.set_ylabel("Verified test acc.")
        ax.set_title("SRNN ECG")
        ax.legend()

        ax = axes[3]
        ax.hist(output_dic_robust_ecg["V"].flatten(), density=True, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Robust")
        ax.hist(output_dic_standard_ecg["V"].flatten(), density=True, alpha=0.5, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.set_ylabel("Normalized bin count")
        ax.set_xlabel("Membrane potential")
        ax.legend()

        plt.savefig("Resources/Figures/figure_interval_propagation.pdf", dpi=1200)
        plt.show()