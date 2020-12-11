from architectures import speech_lsnn
from datajuicer import dj, split, configure, query, djm
from experiment_utils import *
import numpy as onp

class constant_attack_experiment:

    @staticmethod
    def train_grid():
        grid = speech_lsnn.make()
        grid = split(grid, "attack_size_constant", [0.01])
        grid = split(grid, "attack_size_mismatch", [0.0])
        grid = split(grid, "initial_std_constant", [0.001])
        grid = split(grid, "initial_std_mismatch", [0.0])
        grid = split(grid, "beta_robustness", [0.0, 0.1, 1.0, 10.0])
        grid = split(grid, "seed", [0,1,2,3,4,5,6,7,8,9])
        return grid

    @staticmethod
    def visualize():

        grid = [model for model in constant_attack_experiment.train_grid() if model["seed"] in [0,1,2,3,4,5,6,7,8,9]]
        grid = djm(grid, "train", run_mode="load")("{*}")
        grid = configure(grid, {"surface_dist_short": 0.01})
        grid = configure(grid, {"surface_dist_long": 0.05})
        grid = configure(grid, {"n_iterations": 2})
        grid = configure(grid, {"mode":"direct"})

        grid = djm(grid, get_attacked_test_acc, n_threads=1, store_key="attacked_test_acc")("{*}", "{data_dir}")
        grid = djm(grid, get_surface_mean, n_threads=1, store_key="surface_mean_short")("{n_iterations}", "{*}", "{surface_dist_short}", "{data_dir}")
        grid = djm(grid, get_surface_mean, n_threads=1, store_key="surface_mean_long")("{n_iterations}", "{*}", "{surface_dist_long}", "{data_dir}")

        # - Get the data
        betas = [0.0, 0.1, 1.0, 10.0]
        attacked_test_acc_data = [query(grid, "attacked_test_acc", where={"beta_robustness": beta}) for beta in betas]
        surface_mean_short_data = [query(grid, "surface_mean_short", where={"beta_robustness": beta}) for beta in betas]
        surface_mean_long_data = [query(grid, "surface_mean_long", where={"beta_robustness": beta}) for beta in betas]

        fig = plt.figure(figsize=(10, 5), constrained_layout=False)
        gridspec = fig.add_gridspec(2, 3, left=0.05, right=0.98, hspace=0.3, wspace=0.2)

        def plot_training_evolution(ax, beta, acc_norm_list, acc_robust_list, filter_length=50, alpha=0.3):
            for idx,(acc_norm, acc_robust) in enumerate(zip(acc_norm_list,acc_robust_list)):
                ma_acc_norm = onp.convolve(acc_norm, onp.ones(filter_length)/filter_length, mode="full")[:len(acc_norm)]
                ma_acc_robust = onp.convolve(acc_robust, onp.ones(filter_length)/filter_length, mode="full")[:len(acc_robust)]
                x = onp.linspace(0,len(acc_robust)-1, len(acc_robust))
                label1 = label2 = None
                if(idx == 0):
                    label1 = "Normal"; label2 = "Attacked"
                ax.plot(x, ma_acc_norm, label=label1, color="C1", alpha=alpha)
                ax.plot(x, ma_acc_robust, label=label2, color="C2", alpha=alpha)
                ax.plot(x, acc_norm, color="C1", alpha=alpha/3)
                ax.plot(x, acc_robust, color="C2", alpha=alpha/3)
            acc_norm_list = onp.array([onp.convolve(acc_norm, onp.ones(filter_length)/filter_length, mode="full")[:len(acc_norm)] for acc_norm in acc_norm_list])
            acc_robust_list = onp.array([onp.convolve(acc_robust, onp.ones(filter_length)/filter_length, mode="full")[:len(acc_robust)] for acc_robust in acc_robust_list])
            ax.plot(x, onp.mean(acc_norm_list, axis=0), color="C1")
            ax.plot(x, onp.mean(acc_robust_list, axis=0), color="C2")
            ax.set_ylim([0.3,1.0])
            if(beta > 0):
                ax.set_ylabel(str(beta)+r" [$\beta$]")

        # - Get the top axes
        top_axes,bottom_axes = get_axes_constant_attack_figure(fig, gridspec, betas)
        for idx,beta in enumerate([0.0,0.1,10.0]):
            plot_training_evolution(top_axes[idx], beta, query(grid, "training_accuracies", where={"beta_robustness":beta}), query(grid, "attacked_training_accuracies", where={"beta_robustness":beta}))
        top_axes[-1].legend(frameon=False)

        def plot_bottom_data(data, linestyle, color, marker, label, betas):
            x = onp.linspace(0,len(betas)-1,len(betas))
            bottom_axes.errorbar(x,onp.median(data, axis=1),onp.std(data,axis=1), label=label, color=color, marker=marker, linestyle=linestyle, markevery=list(onp.array(x,int)), capsize=3)

        plot_bottom_data(onp.array(attacked_test_acc_data), linestyle="dashed", color="C2", marker="o", label="Attacked 0.01", betas=betas)
        plot_bottom_data(onp.array(surface_mean_short_data), linestyle="dotted", color="C4", marker=".", label="Surface 0.01", betas=betas)
        plot_bottom_data(onp.array(surface_mean_long_data), linestyle="solid", color="C6", marker="s", label="Surface 0.05", betas=betas)
        bottom_axes.legend(frameon=False)

        plt.savefig("Resources/Figures/constant_attack_figure.pdf", dpi=1200)
        plt.savefig("Resources/Figures/constant_attack_figure.png", dpi=1200)
        plt.show()
        