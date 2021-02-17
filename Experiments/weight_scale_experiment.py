from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib as mpl
from scipy import stats

class weight_scale_experiment:

    @staticmethod
    def train_grid():
        grid = speech_lsnn.make()
        grid = split(grid, "attack_size_constant", [0.01])
        grid = split(grid, "attack_size_mismatch", [0.0])
        grid = split(grid, "initial_std_constant", [0.001])
        grid = split(grid, "initial_std_mismatch", [0.0])
        grid = split(grid, "beta_robustness", [0.0, 0.1, 1.0, 10.0])
        grid = split(grid, "seed", [0,1,2,3,4,5,6,7,8,9])

        grid2 = speech_lsnn.make()
        grid2 = split(grid2, "attack_size_constant", [0.0])
        grid2 = split(grid2, "attack_size_mismatch", [2.0])
        grid2 = split(grid2, "initial_std_constant", [0.0])
        grid2 = split(grid2, "initial_std_mismatch", [0.001])
        grid2 = split(grid2, "beta_robustness", [0.0, 0.1, 1.0, 10.0])
        grid2 = split(grid2, "seed", [0,1,2,3,4,5,6,7,8,9])

        return grid + grid2

    @staticmethod
    def visualize():
        betas = [0.0, 0.1, 1.0, 10.0]
        seeds = [0,1,2,3,4,5,6,7,8,9]
        grid = [model for model in weight_scale_experiment.train_grid() if model["seed"] in [0,1,2,3,4,5,6,7,8,9]]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        N_rows = len(grid[0]["theta"])
        N_cols = len(betas)
        theta =  list(grid[0]["theta"].keys())
        params = ["".join(list(map(lambda x : " " if x=="_" else x,p))) for p in theta]

        def ij2idx(i,j):
            return i*N_cols + j

        fig = plt.figure(figsize=(14, 5), constrained_layout=False)
        axes = get_axes_weight_scale_exp(fig, N_rows, N_cols)
        for idx,b in enumerate(betas):
            axes[ij2idx(N_rows-1,idx)].set_xlabel(r"$\beta$ %.1f" % b)
        for idx,p in enumerate(params):
            axes[ij2idx(idx,0)].set_ylabel(r"$\textnormal{%s}$" % p)

        for i,param in enumerate(theta):
            for j,beta in enumerate(betas):
                weights = onp.array(query(grid, "theta", where={"seed":0, "beta_robustness": beta})[0][param])
                mu, std = stats.norm.fit(weights)
                xmin = onp.min(weights) ; xmax = onp.max(weights)
                x = onp.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                axes[ij2idx(i,j)].plot(x, p, color="k", linewidth=2.0, linestyle="dashed")
                axes[ij2idx(i,j)].hist(onp.ravel(weights), bins=10, density=True, alpha=0.3)
        plt.savefig("Resources/Figures/figure_weight_distributions.pdf", dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        gridspec = fig.add_gridspec(N_rows, 1, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        axes = [fig.add_subplot(gridspec[i,0]) for i in range(N_rows)]
        remove_all_but_left_btm(axes)
        for i,param in enumerate(theta):
            x = [x  for j in [len(seeds)*[b] for b in range(len(betas))] for x in j]
            min_max = []
            for b in betas:
                weights = query(grid, "theta", where={"beta_robustness": b})
                for el in weights:
                    w = onp.array(el[param])
                    min_max.append(onp.max(w)-onp.min(w))
            axes[i].scatter(x, min_max)
            axes[i].set_xticks([x for x in range(len(betas))])
            axes[i].set_xticklabels([(r"$\beta$ %.1f" % b) for b in betas])
            axes[i].set_xlim(-0.5,len(betas)-0.5)
            axes[i].set_ylabel(params[i])

        axes[0].set_title(r"Max($\Theta$)-Min($\Theta$)")
        plt.savefig("Resources/Figures/figure_weight_min_max.pdf", dpi=1200)
        plt.show()

