from architectures import ecg_lsnn, speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib as mpl
from scipy import stats

class weight_scale_experiment:

    @staticmethod
    def train_grid():
        seeds = [0,1,2,3,4]
        beta_robustness = [0.0,0.125,0.5,1.0]

        grid = speech_lsnn.make()
        grid1 = configure([grid], dictionary={"attack_size_constant":0.01,"attack_size_mismatch":0.0,"initial_std_constant":0.001, "n_epochs":"80,20", "initial_std_mismatch":0.0})
        grid1 = split(grid1, "beta_robustness", beta_robustness)
        grid1 = split(grid1, "seed", seeds)

        grid2 = configure([grid], dictionary={"attack_size_constant":0.0,"attack_size_mismatch":0.2,"initial_std_constant":0.0, "n_epochs":"80,20", "initial_std_mismatch":0.001})
        grid2 = split(grid2, "beta_robustness", beta_robustness)
        grid2 = split(grid2, "seed", seeds)

        return grid1 + grid2

    @staticmethod
    def visualize():
        ATTACK_SIZE_CONST = 0.01
        ATTACK_SIZE_MM = 0.2
        betas = [0.0,0.125,0.5,1.0]
        seeds = [0,1,2,3,4]
        grid = [model for model in weight_scale_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        theta = ["W_in", "W_rec", "W_out"]
        params = ["".join(list(map(lambda x : " " if x=="_" else x,p))) for p in theta] # - Exclude the bias
        N_cols = len(params)

        fig = plt.figure(figsize=(10, 2), constrained_layout=True)
        gridspec = fig.add_gridspec(1, N_cols, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        axes = [fig.add_subplot(gridspec[0,i]) for i in range(N_cols)]
        remove_all_but_left_btm(axes)
        for i,param in enumerate(theta):
            x = [x  for j in [len(seeds)*[b] for b in range(len(betas))] for x in j]
            d_constant = []
            d_relative = []
            for b in betas:
                weights_constant = query(grid, "theta", where={"attack_size_constant":ATTACK_SIZE_CONST, "beta_robustness": b})
                weights_relative = query(grid, "theta", where={"attack_size_mismatch":ATTACK_SIZE_MM, "beta_robustness": b})
                for el_constant, el_relative in zip(weights_constant,weights_relative):
                    w_constant = onp.array(el_constant[param])
                    w_relative = onp.array(el_relative[param])
                    d_constant.append(onp.mean(onp.abs(w_constant)))
                    d_relative.append(onp.mean(onp.abs(w_relative)))
            label_const = label_rel = None
            if i == 0:
                label_const = "Constant"
                label_rel = "Relative"
            axes[i].scatter(x, d_constant, color="b", label=label_const, alpha=0.5)
            axes[i].scatter(x, d_relative, color="r", label=label_rel, alpha=0.5)
            axes[i].set_xticks([x for x in range(len(betas))])
            r = lambda x : float(onp.round(x * 100) / 100)
            axes[i].set_yticks(onp.linspace(r(axes[i].get_ylim()[0]),r(axes[i].get_ylim()[1]),5))
            axes[i].set_xticklabels([(r"$\beta$ %.1f" % b) for b in betas])
            axes[i].set_xlim(-0.5,len(betas)-0.5)
            axes[i].set_ylabel(params[i])
            axes[i].grid(axis='y', which='both')
            axes[i].grid(axis='x', which='major')
            if i == 0:
                axes[i].legend(loc=2)

        axes[0].set_title(r"Sum(Abs($\Theta$))")
        plt.savefig("Resources/Figures/figure_weight_magnitudes.pdf", dpi=1200)
        plt.show()

