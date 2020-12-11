from architectures import speech_lsnn
from datajuicer import dj, split, configure, query, djm
from experiment_utils import *
import numpy as onp
import matplotlib as mpl

class methods_experiment:

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
        betas = [0.0, 0.1, 1.0]
        grid = [model for model in methods_experiment.train_grid() if model["seed"] in [0,1,2,3,4,5,6,7,8,9]]
        grid = djm(grid, "train", run_mode="load")("{*}")
        grid = configure(grid, {"mode":"direct"})

        fig = plt.figure(figsize=(12, 9), constrained_layout=False)
        gridspec = fig.add_gridspec(3, 6, left=0.05, right=0.98, hspace=0.3, wspace=0.4)
        _,axes_middle,axes_bottom = get_axes_method_figure(fig, gridspec)

        kl_over_time = [onp.array(query(grid, "kl_over_time", where={"beta_robustness": beta})) for beta in betas]

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0.0,vmax=1.0), cmap="RdBu"), ax=axes_bottom[0], ticks=[0.0,1.0])
        cbar.ax.set_yticklabels(["t=0","t=T"])

        def plot_kl_over_time(ax, kl_over_time_beta, cmap, beta):
            t = onp.linspace(0,1.0,num=kl_over_time_beta.shape[1])
            n_seeds = kl_over_time_beta.shape[0]
            for seed in range(n_seeds):
                for idx in range(kl_over_time_beta.shape[1]):
                    ax.plot(kl_over_time_beta[seed,idx], color=cmap(t[idx]), alpha=0.2)
            ax.set_xlabel(r"$\beta=$"+str(beta))

        def plot_end_kl_over_time(ax, kl_over_time_beta, beta, filter_length=50):
            n_seeds = kl_over_time_beta.shape[0]
            T = kl_over_time_beta.shape[1]
            data = onp.zeros((T,n_seeds)); data_ma = onp.zeros((T,n_seeds))
            for seed in range(n_seeds):
                data[:,seed] = onp.array([el[-1] for el in kl_over_time_beta[seed]])
                data_ma[:,seed] = onp.convolve(data[:,seed], onp.ones(filter_length)/filter_length, mode="full")[:T]
            ax.plot(data, alpha=0.1, color="C2")
            ax.plot(data_ma, alpha=0.5, color="C2")
            ax.plot(onp.mean(data_ma, axis=1), alpha=1.0, color="C2", label=r"$\beta$="+str(beta))
            ax.legend(frameon=False, loc=0)

        for i in range(len(betas)):
            plot_kl_over_time(axes_bottom[i], kl_over_time[i], cbar.cmap, betas[i])
        
        plot_end_kl_over_time(axes_middle[0], kl_over_time[0], betas[0])
        plot_end_kl_over_time(axes_middle[1], kl_over_time[1], betas[1])
        
        axes_middle[0].set_xticks([0,kl_over_time[0].shape[1]])
        axes_middle[0].set_xticklabels(["t=0","t=T"])
        axes_middle[1].ticklabel_format(style="sci", scilimits=(0,0))

        plt.savefig("Resources/Figures/methods_figure.pdf", dpi=1200)
        plt.savefig("Resources/Figures/methods_figure.png", dpi=1200)
        plt.show()