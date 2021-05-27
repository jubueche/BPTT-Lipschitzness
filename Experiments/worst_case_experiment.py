from jax import config
config.update('jax_disable_jit', False)

from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np
from datajuicer.table import Table
from datajuicer.visualizers import latex, visualizer, METHOD_COLORS, METHOD_LINESTYLE, METHOD_LINEWIDTH
from datajuicer.utils import reduce_keys

class worst_case_experiment:

    @staticmethod
    def train_grid():
        seeds = [0,1]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        ecg2 = []#configure(ecg, {"beta_robustness": 0.0, "dropout_prob": 0.3})
        ecg3 = []#configure(ecg, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.1,0.01", "n_epochs":"20,10"})
        ecg4 = configure(ecg, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        ecg5 = []#configure(ecg, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001"})
        ecg6 = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg7 = configure(ecg, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg = ecg0 + ecg1 + ecg2 + ecg3 + ecg4 + ecg5 + ecg6 + ecg7

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1})
        speech2 = [] # configure(speech, {"beta_robustness": 0.0, "dropout_prob":0.3})
        speech3 = [] # configure(speech, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"40,10"})
        speech4 = configure(speech, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        speech5 = [] #configure(speech, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001"})
        speech6 = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        speech7 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        speech = speech0 + speech1 + speech2 + speech3 + speech4  + speech5 + speech6 + speech7

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        cnn_grid2 = [] #configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = [] #configure(cnn_grid, {"beta_robustness": 0.0, "optimizer": "esgd", "n_epochs":"10,5"})
        cnn_grid4 = configure(cnn_grid, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        cnn_grid5 = [] #configure(cnn_grid, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"10,2"})
        cnn_grid6 = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry"})
        cnn_grid7 = configure(cnn_grid, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3 + cnn_grid4 + cnn_grid5 + cnn_grid6 + cnn_grid7

        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "seed", seeds)

        return final_grid

    @staticmethod
    def visualize():
        attack_sizes = [0.0,0.005,0.01,0.05,0.1,0.2,0.3,0.5]
        n_attack_steps = [10]
        seeds = [0,1]
        attack_size_mismatch_speech = 0.1
        attack_size_mismatch_ecg = 0.1
        attack_size_mismatch_cnn = 0.1
        boundary_loss = ["madry","kl"]

        grid = [model for model in worst_case_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_worst_case = configure(grid, {"mode":"direct"})
        grid_worst_case = split(grid_worst_case, "attack_size", attack_sizes)
        grid_worst_case = split(grid_worst_case, "n_attack_steps", n_attack_steps)
        grid_worst_case = split(grid_worst_case, "boundary_loss", boundary_loss)

        grid_worst_case = run(grid_worst_case, min_whole_attacked_test_acc, n_threads=2, store_key="min_acc_test_set_acc")(1, "{*}", "{data_dir}", "{n_attack_steps}", "{attack_size}", 0.0, 0.001, 0.0, "{boundary_loss}")
        for g in grid_worst_case:
            acc,loss = g["min_acc_test_set_acc"]
            g["acc"] = 100 * acc
            g["loss"] = loss

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
            "Beta = 0.25":"Beta 0.25",
            "Beta = 0.5":"Beta 0.5",
            "Beta = 0.2, Forward": "Forward + Beta 0.2",
            "Beta = 0.3, Forward": "Forward + Beta 0.3",
            "Beta = 0.5, Forward": "Forward + Beta 0.5",
            "Beta 0.25, Forward": "Forward + Beta 0.25",
            "Beta 0.5, Forward": "Forward + Beta 0.5",
            "Beta = 0.1, Forward": "Forward + Beta 0.1",
            "noisy_forward_std = 0.3": "Forward",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD"
        }

        order = {
            "architecture": ["speech_lsnn", "ecg_lsnn", "cnn"],
            "Method": ["Forward + Beta 0.1", "Forward + Beta 0.5", "Beta 0.25", "Beta 0.5", "Standard", "Forward"]
        }

        def get_table(architecture, boundary_loss, loss_or_acc):
            sub_grid = [g for g in grid_worst_case if g["architecture"]==architecture and g["boundary_loss"]==boundary_loss]
            group_by = ["awp", "beta_robustness", "dropout_prob", "optimizer", "noisy_forward_std", "attack_size", "n_attack_steps"]
            reduced = reduce_keys(sub_grid, loss_or_acc, reduction={"mean":lambda l : float(np.mean(l))}, group_by=group_by)
            independent_keys = ["n_attack_steps",Table.Deviation_Var({"beta_robustness":0.0, "awp":False, "dropout_prob":0.0, "optimizer":"adam", "noisy_forward_std":0.0}, label="Attack"),  "attack_size"]
            dependent_keys = [loss_or_acc+"_mean"]

            print(latex(reduced, independent_keys, dependent_keys, label_dict, bold_order=[max if loss_or_acc=="acc" else min]))

        @visualizer(dim=4)
        def grid_plot(table, axes):
            shape = table.shape()
            for i0 in range(shape[0]):
                n_attack_steps = table.get_label(axis=0, index=i0)
                data_dic = {table.get_label(axis=2, index=idx): [table.get_val(i0,i1,idx,0) for i1 in range(shape[1])] for idx in range(shape[2])}
                for idx,label in enumerate(data_dic):
                    d = data_dic[label]
                    if not None in d:
                        d = [onp.mean(a) for a in d]
                        axes[i0].plot(range(len(d)), d, color=METHOD_COLORS[label], label=label, linestyle=METHOD_LINESTYLE[label], linewidth=METHOD_LINEWIDTH[label]) 
                axes[i0].grid(axis='y', which='both')
                axes[i0].grid(axis='x', which='major')
                axes[i0].set_ylabel("Test Acc.")
                axes[i0].set_title(table.get_label(axis=0, index=i0))
                axes[i0].set_xlabel(r"Attack size $\zeta$")
            axes[0].legend(frameon=True, prop={'size': 7})

        def plot(boundary_loss, loss_or_acc):
            sub_grid = [g for g in grid_worst_case if g["boundary_loss"]==boundary_loss and g["optimizer"]=="adam"]
            fig = plt.figure(figsize=(10, 3), constrained_layout=True)
            axes = get_axes_worst_case(fig, N_rows=1, N_cols=3, attack_sizes=attack_sizes)
            independent_keys = ["architecture","attack_size", Table.Deviation_Var({"beta_robustness":0.0, "awp":False, "dropout_prob":0.0, "optimizer":"adam", "noisy_forward_std":0.0}, label="Method")]
            dependent_keys = [loss_or_acc]
            grid_plot(sub_grid, independent_keys=independent_keys, dependent_keys=dependent_keys, label_dict=label_dict, axes=axes, order=order)
            plt.savefig(f"Resources/Figures/figure_worst_case_{loss_or_acc}_boundary_{boundary_loss}.pdf", dpi=1200)
            plt.show()

        plot("madry", loss_or_acc="acc")
        plot("madry", loss_or_acc="loss")
        plot("kl", loss_or_acc="acc")
        plot("kl", loss_or_acc="loss")

        print("--------- Speech SRNN ---------")
        get_table("speech_lsnn","kl",loss_or_acc="acc")
        get_table("speech_lsnn","madry",loss_or_acc="acc")
        print("--------- ECG SRNN ---------")
        get_table("ecg_lsnn","kl",loss_or_acc="acc")
        get_table("ecg_lsnn","madry",loss_or_acc="acc")
        print("--------- CNN ---------")
        get_table("cnn","kl",loss_or_acc="acc")
        get_table("cnn","madry",loss_or_acc="acc")