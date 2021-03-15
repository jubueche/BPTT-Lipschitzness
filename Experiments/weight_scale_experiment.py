from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib as mpl
from scipy import stats

class weight_scale_experiment:

    @staticmethod
    def train_grid():
        seeds = [0,1,2,3,4]
        beta_robustness = [0.0,0.125,1.0,10.0]

        grid = speech_lsnn.make()
        grid1 = configure([grid], dictionary={"attack_size_constant":0.01,"attack_size_mismatch":0.0,"initial_std_constant":0.001, "initial_std_mismatch":0.0})
        grid1 = split(grid1, "beta_robustness", beta_robustness)
        grid1 = split(grid1, "seed", seeds)

        grid2 = configure([grid], dictionary={"attack_size_constant":0.0,"attack_size_mismatch":0.3,"initial_std_constant":0.0, "initial_std_mismatch":0.001})
        grid2 = split(grid2, "beta_robustness", beta_robustness)
        grid2 = split(grid2, "seed", seeds)

        return grid1 + grid2

    @staticmethod
    def visualize():
        betas = [0.0, 0.1, 1.0, 10.0]
        seeds = [0,1,2,3,4]
        grid = [model for model in weight_scale_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        N_cols = len(betas)
        theta =  [el for el in list(grid[0]["theta"].keys()) if el != "b_out"]
        params = ["".join(list(map(lambda x : " " if x=="_" else x,p))) for p in theta] # - Exclude the bias
        N_rows = len(params)

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
                weights_constant = onp.array(query(grid, "theta", where={"seed":0, "attack_size_constant":0.01, "beta_robustness": beta})[0][param])
                weights_relative = onp.array(query(grid, "theta", where={"seed":0, "attack_size_mismatch":0.3, "beta_robustness": beta})[0][param])
                mu_constant, std_constant = stats.norm.fit(weights_constant)
                mu_relative, std_relative = stats.norm.fit(weights_relative)
                xmin = min(onp.min(weights_constant),onp.min(weights_relative))
                xmax = max(onp.max(weights_constant),onp.max(weights_relative))
                x = onp.linspace(xmin, xmax, 100)
                p_constant = stats.norm.pdf(x, mu_constant, std_constant)
                p_relative = stats.norm.pdf(x, mu_relative, std_relative)
                axes[ij2idx(i,j)].plot(x, p_constant, color="k", linewidth=2.0, linestyle="dashed")
                axes[ij2idx(i,j)].plot(x, p_relative, color="k", linewidth=2.0, linestyle="dotted")
                axes[ij2idx(i,j)].hist(onp.ravel(weights_constant), bins=10, density=True, color="b", alpha=0.3)
                axes[ij2idx(i,j)].hist(onp.ravel(weights_relative), bins=10, density=True, color="r", alpha=0.3)
        plt.savefig("Resources/Figures/figure_weight_distributions.pdf", dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        gridspec = fig.add_gridspec(N_rows, 1, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        axes = [fig.add_subplot(gridspec[i,0]) for i in range(N_rows)]
        remove_all_but_left_btm(axes)
        for i,param in enumerate(theta):
            x = [x  for j in [len(seeds)*[b] for b in range(len(betas))] for x in j]
            min_max_constant = []
            min_max_relative = []
            for b in betas:
                weights_constant = query(grid, "theta", where={"attack_size_constant":0.01, "beta_robustness": b})
                weights_relative = query(grid, "theta", where={"attack_size_mismatch":0.3, "beta_robustness": b})
                for el_constant, el_relative in zip(weights_constant,weights_relative):
                    w_constant = onp.array(el_constant[param])
                    w_relative = onp.array(el_relative[param])
                    min_max_constant.append(onp.max(w_constant)-onp.min(w_constant))
                    min_max_relative.append(onp.max(w_relative)-onp.min(w_relative))
            axes[i].scatter(x, min_max_constant, color="b")
            axes[i].scatter(x, min_max_relative, color="r")
            axes[i].set_xticks([x for x in range(len(betas))])
            axes[i].set_xticklabels([(r"$\beta$ %.1f" % b) for b in betas])
            axes[i].set_xlim(-0.5,len(betas)-0.5)
            axes[i].set_ylabel(params[i])

        axes[0].set_title(r"Max($\Theta$)-Min($\Theta$)")
        plt.savefig("Resources/Figures/figure_weight_min_max.pdf", dpi=1200)
        plt.show()

