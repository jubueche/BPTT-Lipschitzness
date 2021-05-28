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
from datajuicer.visualizers import *
from jax import config
config.FLAGS.jax_log_compiles=True
config.update('jax_disable_jit', False)
import numpy as np

class IBP_experiment:

    @staticmethod
    def train_grid():
        seeds = [0,1]

        grid = [speech_lsnn.make()]
        grid0 = configure(grid, {"beta_robustness":0.0})
        grid1 = configure(grid, {"beta_robustness":0.5, "noisy_forward_std":0.3, "attack_size_mismatch":0.1})
        grid2 = configure(grid, {"beta_robustness":0.0, "noisy_forward_std":0.3})
        grid3 = configure(grid, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1})
        speech = grid0 + grid1 + grid2 + grid3

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        ecg2 = configure(ecg, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg3 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        ecg = ecg0 + ecg1 + ecg2 + ecg3
        final_grid = split(ecg+speech, "seed", seeds)
        return final_grid

    @staticmethod
    def visualize():
        grid = [model for model in IBP_experiment.train_grid()]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        zetas = [0.0,0.00001,0.00005,0.0001,0.0005,0.001]
        grid = split(grid, "zeta", zetas)

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
            "Beta = 0.25":"Beta",
            "Beta = 0.5":"Beta",
            "Beta = 0.1":"Beta",
            "noisy_forward_std = 0.3": "Forward Noise",
            "Beta, Forward Noise": "Forward Noise + Beta",
            "noisy_forward_std = 0.0": "No Forward Noise",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD"
        }

        @visualizer(dim=4)
        def ibp_plot(table, axes_dict, zetas):
            table_shape = table.shape()
            for i0 in range(table_shape[0]):
                ax = axes_dict[table.get_label(0,i0)]
                ax.set_title(table.get_label(0,i0))
                ax.set_xlabel(r"Attack size $\zeta$")
                ax.set_ylabel("Verified test acc.")
                ax.set_xticks(onp.arange(0,len(zetas),1))
                ax.set_xticklabels(zetas)
                arch = table.get_label(0, i0)
                for i1 in range(table_shape[1]):
                    data = [table.get_val(i0,i1,i2,0) for i2 in range(table_shape[2])]
                    if None in data or data == []:
                        continue
                    else:
                        data = [np.mean(a) for a in data]
                        method = table.get_label(1,i1)
                        ax.plot(data, color=METHOD_COLORS[method], linewidth=METHOD_LINEWIDTH[method], linestyle=METHOD_LINESTYLE[method], label=method)
            axes_dict[table.get_label(0,0)].legend()
            plt.savefig("Resources/Figures/figure_interval_propagation.pdf", dpi=1200)
            plt.show()

        grid = run(grid, get_IBP_test_acc, n_threads=1, store_key="interval_accuracy")("{*}","{zeta}","{data_dir}")

        _, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(9,3))
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        independent_keys = ["architecture", Table.Deviation_Var(default={"beta_robustness":0.0, "noisy_forward_std":0.0},label="method"), "zeta"]
        dependent_keys = ["interval_accuracy"]
        axes_dict = {"Speech LSNN":axes[0], "ECG LSNN":axes[1]}
        ibp_plot(grid, independent_keys=independent_keys,dependent_keys=dependent_keys,label_dict=label_dict, axes_dict=axes_dict, order=None, zetas=zetas)

        def get_voltage(model, data_dir):
            class Namespace:
                def __init__(self,d):
                    self.__dict__.update(d)
            FLAGS = Namespace(model)
            loader, set_size = get_loader(FLAGS, data_dir)
            X,_ = loader.get_batch("test")
            _, output_dic = model["network"].call_verbose(X, model["network"].unmasked(), **model["theta"])
            model["V"] = output_dic["V"]
            return model

        grid = run(grid, get_voltage, n_threads=1, store_key="V")("{*}","{data_dir}")

        _, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True,figsize=(9,3))
        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        V_speech_rob = onp.array(query(grid, "V", where={"architecture":"speech_lsnn", "noisy_forward_std":0.3})).flatten()
        V_speech_rob[V_speech_rob < -20]=-20
        V_speech_def = onp.array(query(grid, "V", where={"architecture":"speech_lsnn", "beta_robustness":0.0, "noisy_forward_std":0.0})).flatten()

        ax = axes[0]
        ax.hist(V_speech_rob, density=True, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Robust")
        ax.hist(V_speech_def, density=True, alpha=0.5, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.set_ylabel("Normalized bin count")
        ax.set_xlabel("Membrane potential")
        ax.legend()

        V_ecg_rob = onp.array(query(grid, "V", where={"architecture":"speech_lsnn", "noisy_forward_std":0.3})).flatten()
        V_ecg_rob[V_ecg_rob < -20]=-20
        V_ecg_def = onp.array(query(grid, "V", where={"architecture":"speech_lsnn", "beta_robustness":0.0, "noisy_forward_std":0.0})).flatten()

        ax = axes[1]
        ax.hist(V_ecg_rob, density=True, color=METHOD_COLORS["Beta + Forward"], linewidth=METHOD_LINEWIDTH["Beta + Forward"], linestyle=METHOD_LINESTYLE["Beta + Forward"], label="Robust")
        ax.hist(V_ecg_def, density=True, alpha=0.5, color=METHOD_COLORS["Standard"], linewidth=METHOD_LINEWIDTH["Standard"], linestyle=METHOD_LINESTYLE["Standard"], label="Standard")
        ax.set_ylabel("Normalized bin count")
        ax.set_xlabel("Membrane potential")
        ax.legend()

        plt.savefig("Resources/Figures/membrane_potential_distribution.pdf", dpi=1200)
        plt.show()