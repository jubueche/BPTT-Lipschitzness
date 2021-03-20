from jax import config
# config.update("jax_enable_x64", True)
config.update('jax_disable_jit', False)

from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
from Experiments.mismatch_experiment import mismatch_experiment

class worst_case_experiment(mismatch_experiment):

    @staticmethod
    def visualize():

        ACC = 0
        LOSS = 1

        attack_sizes = [0.0,0.005,0.01,0.05,0.1,0.2,0.3,0.5]
        n_attack_steps = [5,10,15]
        seeds = [0]
        beta = 0.125
        dropout = 0.3

        grid = [model for model in mismatch_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_worst_case = configure(grid, {"mode":"direct"})
        grid_worst_case = split(grid_worst_case, "attack_size", attack_sizes)
        grid_worst_case = split(grid_worst_case, "n_attack_steps", n_attack_steps)
        
        if(config.FLAGS.jax_enable_x64):
            for g in grid_worst_case:
                for p in g["theta"]:
                    g["theta"][p] = g["theta"][p].astype(jnp.float64)

        grid_worst_case = run(grid_worst_case, min_whole_attacked_test_acc, n_threads=1, store_key="min_acc_test_set_acc")(5, "{*}", "{data_dir}", "{n_attack_steps}", "{attack_size}", 0.0, 0.001, 0.0)

        def _get_data_acc(architecture, beta, identifier, grid, n_attack_steps):
            robust_data = query(grid, identifier, where={"beta_robustness":beta, "attack_size_mismatch":0.2, "dropout_prob":0.0, "architecture":architecture, "n_attack_steps":n_attack_steps})
            robust_data_acc = onp.array([el[0] for el in robust_data])
            robust_data_loss = onp.array([el[1] for el in robust_data])

            vanilla_data = query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.0, "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_data_acc = onp.array([el[0] for el in vanilla_data])
            vanilla_data_loss = onp.array([el[1] for el in vanilla_data])

            vanilla_dropout_data = query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.3, "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_dropout_data_acc = onp.array([el[0] for el in vanilla_dropout_data])
            vanilla_dropout_data_loss = onp.array([el[1] for el in vanilla_dropout_data])

            return (vanilla_data_acc,vanilla_data_loss), (vanilla_dropout_data_acc,vanilla_dropout_data_loss), (robust_data_acc,robust_data_loss)

        def print_worst_case_test(data, attack_sizes, beta, dropout, n_attack_steps, typ):
            if(typ == ACC):
                print("%s \t\t %s \t %s \t %s" % ("Attack size","Test acc. ($\\beta=0$)",f"Test acc. (dropout = {dropout})",f"Test acc. ($\\beta={beta}$)"))
            else:
                print("%s \t\t %s \t %s \t\t %s" % ("Attack size","Loss ($\\beta=0$)",f"Loss (dropout = {dropout})",f"Loss ($\\beta={beta}$)"))
            for idx,attack_size in enumerate(attack_sizes):
                m = 1
                if(typ == ACC):
                    m = 100 # - Percentage
                dn = 100*onp.ravel(data[0][typ])[idx]
                dnd = 100*onp.ravel(data[1][typ])[idx]
                dr = 100*onp.ravel(data[2][typ])[idx]
                print("%.3f \t\t\t %.2f \t\t\t %.2f \t\t\t\t %.2f" % (attack_size,dn,dnd,dr))

        def _plot(ax, data, typ=ACC, labels=None, ylabel=None, title=None):
            colors = ["#4c84e6","#fc033d","#03fc35"]
            for idx in range(len(data)):
                d = onp.ravel(data[idx][typ])
                if(labels is None):
                    ax.plot(range(len(d)), d, color=colors[idx])
                else:
                    ax.plot(range(len(d)), d, color=colors[idx], label=labels[idx])
            ax.grid(axis='y', which='both')
            ax.grid(axis='x', which='major')
            if(ylabel is not None):
                ax.set_ylabel(ylabel)
            if(labels is not None):
                ax.legend(frameon=False, prop={'size': 7})
            if(title is not None):
                ax.set_title(title)

        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        axes = get_axes_worst_case(fig, N_rows=6, N_cols=3, attack_sizes=attack_sizes)
        def ij_x(i,j):
            return i*3+j

        for idx,n in enumerate(n_attack_steps):

            data_ecg_worst_case = _get_data_acc("ecg_lsnn", beta, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_speech_worst_case = _get_data_acc("speech_lsnn", beta, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_cnn_worst_case = _get_data_acc("cnn", beta, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)

            if(idx == 0):
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=ACC, ylabel="Test acc. Speech", labels=["Normal","Dropout","Robust"], title=(r"$N_{\textnormal{steps}}=$ %s" % str(n)))
                _plot(axes[ij_x(2,idx)], data_ecg_worst_case, typ=ACC, ylabel="Test acc. ECG")
                _plot(axes[ij_x(4,idx)], data_cnn_worst_case, typ=ACC, ylabel="Test acc. CNN")
                
                _plot(axes[ij_x(1,idx)], data_speech_worst_case, typ=LOSS, ylabel="Loss Speech")
                _plot(axes[ij_x(3,idx)], data_ecg_worst_case, typ=LOSS, ylabel="Loss ECG")
                _plot(axes[ij_x(5,idx)], data_cnn_worst_case, typ=LOSS, ylabel="Loss CNN")
            else:
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=ACC, title=(r"$N_{\textnormal{steps}}=$ %s" % str(n)))
                _plot(axes[ij_x(2,idx)], data_ecg_worst_case, typ=ACC)
                _plot(axes[ij_x(4,idx)], data_cnn_worst_case, typ=ACC)
                
                _plot(axes[ij_x(1,idx)], data_speech_worst_case, typ=LOSS)
                _plot(axes[ij_x(3,idx)], data_ecg_worst_case, typ=LOSS)
                _plot(axes[ij_x(5,idx)], data_cnn_worst_case, typ=LOSS)

            print("---------------------------")
            print_worst_case_test(data_ecg_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC)

            print("---------------------------")
            print_worst_case_test(data_speech_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC)

            print("---------------------------")
            print_worst_case_test(data_cnn_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC)

            print("---------------------------")
            print_worst_case_test(data_ecg_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS)

            print("---------------------------")
            print_worst_case_test(data_speech_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS)

            print("---------------------------")
            print_worst_case_test(data_cnn_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS)

        plt.savefig("Resources/Figures/figure_worst_case.png", dpi=1200)
        plt.savefig("Resources/Figures/figure_worst_case.pdf", dpi=1200)
        plt.show()