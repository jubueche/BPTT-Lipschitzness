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

        attack_sizes = [0.0,0.005,0.01,0.05,0.1,0.2,0.3,0.5]
        seeds = [0]
        beta = 0.125
        dropout = 0.3

        grid = [model for model in mismatch_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_worst_case = configure(grid, {"mode":"direct"})
        grid_worst_case = split(grid_worst_case, "attack_size", attack_sizes)
        
        if(config.FLAGS.jax_enable_x64):
            for g in grid_worst_case:
                for p in g["theta"]:
                    g["theta"][p] = g["theta"][p].astype(jnp.float64)

        grid_worst_case = run(grid_worst_case, min_whole_attacked_test_acc, n_threads=1, store_key="min_acc_test_set_acc")(5, "{*}", "{data_dir}", 10, "{attack_size}", 0.0, 0.001, 0.0)
        
        def _get_data_acc(architecture, beta, identifier, grid):
            robust_data = onp.array(query(grid, identifier, where={"beta_robustness":beta, "attack_size_mismatch":0.2, "dropout_prob":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_dropout_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.3, "architecture":architecture})).reshape((len(seeds),-1))
            return vanilla_data, vanilla_dropout_data, robust_data

        data_ecg_worst_case = _get_data_acc("ecg_lsnn", beta, "min_acc_test_set_acc", grid_worst_case)
        data_speech_worst_case = _get_data_acc("speech_lsnn", beta, "min_acc_test_set_acc", grid_worst_case)
        data_cnn_worst_case = _get_data_acc("cnn", beta, "min_acc_test_set_acc", grid_worst_case)

        def print_worst_case_test(data, attack_sizes, beta, dropout):
            print("%s \t\t %s \t %s \t %s" % ("Attack size","Test acc. ($\\beta=0$)",f"Test acc. (dropout = {dropout})",f"Test acc. ($\\beta={beta}$)"))
            for idx,attack_size in enumerate(attack_sizes):
                dn = 100*onp.ravel(data[0])[idx]
                dnd = 100*onp.ravel(data[1])[idx]
                dr = 100*onp.ravel(data[2])[idx]
                print("%.3f \t\t\t %.2f \t\t\t %.2f \t\t\t\t %.2f" % (attack_size,dn,dnd,dr))

        fig = plt.figure(figsize=(14, 3), constrained_layout=False)
        axes = get_axes_worst_case(fig, N_rows=1, N_cols=3, attack_sizes=attack_sizes)

        def _plot(ax, data, labels=None, title=None):
            colors = ["#4c84e6","#fc033d","#03fc35"]
            for idx in range(len(data)):
                d = onp.ravel(data[idx])
                if(labels is None):
                    ax.plot(range(len(d)), d, color=colors[idx])
                else:
                    ax.plot(range(len(d)), d, color=colors[idx], label=labels[idx])
            if(title is not None):
                ax.set_title(title)
            if(labels is not None):
                ax.legend()
        
        _plot(axes[0], data_speech_worst_case, title="Speech")
        _plot(axes[1], data_ecg_worst_case, title="ECG")
        _plot(axes[2], data_cnn_worst_case, title="CNN", labels=["Normal","Dropout","Robust"])
        plt.show()

        print("---------------------------")
        print_worst_case_test(data_ecg_worst_case, attack_sizes, beta, dropout)

        print("---------------------------")
        print_worst_case_test(data_speech_worst_case, attack_sizes, beta, dropout)

        print("---------------------------")
        print_worst_case_test(data_cnn_worst_case, attack_sizes, beta, dropout)

        
