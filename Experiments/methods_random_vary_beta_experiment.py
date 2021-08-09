from architectures import cnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np
import seaborn as sns
from datajuicer.visualizers import METHOD_COLORS, METHOD_LINESTYLE, METHOD_LINEWIDTH

seeds = [0]

class methods_random_vary_beta_experiment:
    
    @staticmethod
    def train_grid():
        betas = [0.1,0.3,0.5]
        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0, "attack_size_mismatch": 0.1})
        cnn_grid1 = configure(cnn_grid, {"attack_size_mismatch": 0.1})
        cnn_grid1 = split(cnn_grid1, "beta_robustness", betas)
        cnn_grid = cnn_grid0 + cnn_grid1
        cnn_grid = split(cnn_grid, "seed", seeds)
        return cnn_grid

    @staticmethod
    def visualize():

        attack_sizes = [0.0,0.01,0.05,0.1,0.2,0.3,0.5,0.7]
        grid = [model for model in methods_random_vary_beta_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct","boundary_loss":"madry", "n_attack_steps":10})
        grid = split(grid, "attack_size", attack_sizes)

        grid = run(grid, min_whole_attacked_test_acc, n_threads=2, store_key="min_acc_test_set_acc")(1, "{*}", "{data_dir}", "{n_attack_steps}", "{attack_size}", 0.0, 0.001, 0.0, "{boundary_loss}")
        grid = run(grid, get_surface_mean, n_threads=2, store_key="random_attack")(5,"{*}","{attack_size}","{data_dir}")

        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        axes = get_axes_weight_scale_exp(fig, 2, 2)

        def plot_attack(ax, lab, beta, labels=[None,None]):
            test_attack = [[t[0] for t in query(grid, "min_acc_test_set_acc", where={"beta_robustness":beta, "seed":s})] for s in seeds]
            test_random_attack = [query(grid, "random_attack", where={"beta_robustness":beta, "seed":s}) for s in seeds]
            
            test_attack_mean = onp.mean(onp.array(test_attack), 0)
            test_random_attack_mean = onp.mean(onp.array(test_random_attack), 0)
            print("===== Attack =====")
            print(f"beta {beta} Baseline {test_attack_mean[0]}")
            print("->",test_attack_mean)
            print("===== Random =====")
            print(f"beta {beta} Baseline {test_random_attack_mean[0]}")
            print("->",test_random_attack_mean)

            for t_a,t_a_r in zip(test_attack,test_random_attack):
                ax.plot(t_a, alpha=1./len(test_attack), color=METHOD_COLORS["Forward Noise + Beta"], linestyle=METHOD_LINESTYLE["Forward Noise + Beta"],linewidth=2.0)
                ax.plot(t_a_r, alpha=1./len(test_random_attack), color=METHOD_COLORS["AWP"], linestyle=METHOD_LINESTYLE["AWP"],linewidth=2.0)
            
            ax.plot(test_attack_mean, color=METHOD_COLORS["Forward Noise + Beta"], linestyle=METHOD_LINESTYLE["Forward Noise + Beta"],linewidth=2.0,label=labels[0])
            ax.plot(test_random_attack_mean, color=METHOD_COLORS["AWP"], linestyle=METHOD_LINESTYLE["AWP"],linewidth=2.0,label=labels[1])

            ax.set_title(r"Training $\beta_{\textnormal{rob}}$=" + str(beta))
            ax.set_xlabel(r"Attack size $\zeta$")
            ax.set_xticklabels(attack_sizes)
            ax.set_ylabel("Test acc.")
            ax.grid(axis='y', which='both')
            ax.grid(axis='x', which='major')
            ax.text(s=lab, x=0, y=1.0)
            if not labels[0] == None:
                ax.legend(frameon=True, loc=0, prop={'size': 7})

        plot_attack(axes[0], lab=r"\bf{a}", beta=0.0, labels=["Adversarial","Random"])
        plot_attack(axes[1], lab=r"\bf{b}", beta=0.1)
        plot_attack(axes[2], lab=r"\bf{c}", beta=0.3)
        plot_attack(axes[3], lab=r"\bf{d}", beta=0.5)
        plt.savefig(f"Resources/Figures/methods_random_vary_beta_experiment.pdf", dpi=1200)
        plt.show()
