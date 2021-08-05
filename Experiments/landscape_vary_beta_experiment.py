from architectures import speech_lsnn, ecg_lsnn, cnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib.pyplot as plt
from datajuicer.table import Table
from datajuicer.visualizers import latex, visualizer, METHOD_COLORS, METHOD_LINESTYLE, METHOD_LINEWIDTH
from datajuicer.utils import reduce_keys

seeds = [0]

class landscape_vary_beta_experiment:

    @staticmethod
    def train_grid():
        
        ecg = [ecg_lsnn.make()]
        ecg = configure(ecg, {"attack_size_mismatch": 0.1})
        ecg = split(ecg, "beta_robustness", [0.1,0.2,0.3])

        speech = [speech_lsnn.make()]
        speech = configure(speech, {"attack_size_mismatch": 0.1})
        speech = split(speech, "beta_robustness", [0.1,0.2,0.3])

        cnn_grid = [cnn.make()]
        cnn_grid = configure(cnn_grid, {"attack_size_mismatch": 0.1})
        cnn_grid = split(cnn_grid, "beta_robustness", [0.1,0.2,0.3])

        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "seed", seeds)

        return final_grid

    @staticmethod
    def visualize():
        grid = [model for model in landscape_vary_beta_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        num_steps = 50
        std = 0.2
        alpha_val = 0.2
        from_ = -2.0
        to_ = 2.0
        n_repeat = 5

        grid = run(grid, get_landscape_sweep, n_threads=10, run_mode="normal", store_key="landscape")("{*}", num_steps, "{data_dir}", std, from_, to_, n_repeat)

        label_dict = {
            "beta_robustness": "Beta",
            "n_attack_steps": "Attack steps",
            "attack_size": "Attack size",
            "optimizer": "Optimizer",
            "acc": "Mean Acc.",
            "dropout_prob":"Dropout",
            "cnn" : "F-MNIST CNN",
            "speech_lsnn": "Speech SRNN",
            "ecg_lsnn": "ECG SRNN",
            "awp": "AWP",
            "AWP = True":"AWP",
            "Beta = 0.25":"Beta",
            "Beta = 0.5":"Beta",
            "Dropout = 0.3": "Dropout",
            "noisy_forward_std = 0.3": "Forward Noise",
            "Beta = 0.5, Forward Noise": "Forward Noise + Beta",
            "Beta = 0.25, Forward Noise": "Forward Noise + Beta",
            "Beta = 0.1, Forward Noise": "Forward Noise + Beta",
            "noisy_forward_std = 0.0": "No Forward Noise",
            "Beta, Forward Noise":"Forward Noise + Beta",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD"
        }

        def ma(x, N, fill=True):
            return onp.concatenate([x for x in [ [None]*(N // 2 + N % 2)*fill, onp.convolve(x, onp.ones((N,))/N, mode='valid'), [None]*(N // 2 -1)*fill, ] if len(x)]) 
        def moving_average(x, N):
            result = onp.zeros_like(x)
            if x.ndim > 1:
                over = x.shape[0]
                for i in range(over):
                    result[i] = ma(x[i], N)
            else:
                result = ma(x, N)
            return result
        def get_ma(data,N=5):
            smoothed_mean = moving_average(onp.mean(onp.array(data), axis=0), N=N)
            return smoothed_mean

        fig = plt.figure(figsize=(12,3), constrained_layout=True)
        gridspec = fig.add_gridspec(1, 3, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        axes = [fig.add_subplot(gridspec[0,j]) for j in range(3)]
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_xlabel(r"$\alpha$")
        axes[1].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].set_xlabel(r"$\alpha$")

        @visualizer(dim=3)
        def grid_plot(table, axes, mean_only):
            shape = table.shape()
            for i0 in range(shape[0]):
                data_dic = {table.get_label(axis=1, index=idx): table.get_val(i0,idx,0) for idx in range(shape[1]) if not table.get_val(i0,idx,0) is None}
                for key in data_dic:
                    data_dic[key] = onp.mean(onp.stack(data_dic[key]), 0)
                for idx,label in enumerate(data_dic):
                    if not data_dic[label] is None:
                        if not mean_only:
                            d_raw = data_dic[label]
                            for d in d_raw:
                                axes[i0].plot(onp.linspace(from_,to_,len(d)), d, c=METHOD_COLORS[label], linestyle=METHOD_LINESTYLE[label], linewidth=METHOD_LINEWIDTH[label], alpha=0.2)    
                        d = get_ma(data_dic[label])
                        axes[i0].plot(onp.linspace(from_,to_,len(d)), d, c=METHOD_COLORS[label], linestyle=METHOD_LINESTYLE[label], linewidth=METHOD_LINEWIDTH[label], label=label)
                axes[i0].grid(axis='y', which='both')
                axes[i0].grid(axis='x', which='major')
                axes[i0].set_ylabel("Cross Entropy Loss")
                axes[i0].set_title(table.get_label(axis=0, index=i0))
                axes[i0].set_xlabel(r"$\alpha$")
            axes[0].legend(frameon=True, prop={'size': 7})

        independent_keys = ["architecture", Table.Deviation_Var({"beta_robustness":0.0, "awp":False, "dropout_prob":0.0, "optimizer":"adam", "noisy_forward_std":0.0}, label="Method")]
        dependent_keys = ["landscape"]
        grid_plot(grid, independent_keys=independent_keys, dependent_keys=dependent_keys, label_dict=label_dict, axes=axes, order=None, mean_only=True)

        plt.savefig("Resources/Figures/landscape_vary_beta.pdf", dpi=1200)
        plt.plot()

        for ax in axes:
            ax.clear()
        grid_plot(grid, independent_keys=independent_keys, dependent_keys=dependent_keys, label_dict=label_dict, axes=axes, order=None, mean_only=False)
        plt.savefig("Resources/Figures/landscape_vary_beta_raw.pdf", dpi=1200)
        plt.plot()