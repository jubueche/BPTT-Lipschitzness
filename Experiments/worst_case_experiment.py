from jax import config
# config.update("jax_enable_x64", True)
config.update('jax_disable_jit', False)

from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class worst_case_experiment:

    @staticmethod
    def train_grid():
        seeds = [0]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0}) 
        ecg1 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch":0.1})
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "dropout_prob": 0.3})
        ecg3 = configure(ecg, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.1,0.01", "n_epochs":"20,10"})
        ecg4 = configure(ecg, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        ecg5 = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg = ecg0 + ecg1 + ecg2 + ecg3 + ecg4 + ecg5

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = configure(speech, {"beta_robustness": 0.25, "attack_size_mismatch":0.1})
        speech2 = configure(speech, {"beta_robustness": 0.0, "dropout_prob":0.3})
        speech3 = configure(speech, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"40,10"})
        speech4 = configure(speech, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        speech5 = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        speech = speech0 + speech1 + speech2 + speech3 + speech4 + speech5

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch":0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"10,5"})
        cnn_grid4 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"10,2", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        cnn_grid5 = configure(cnn_grid, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3 + cnn_grid4 + cnn_grid5

        return ecg + speech + cnn_grid

    @staticmethod
    def visualize():

        ACC = 0
        LOSS = 1

        attack_sizes = [0.0,0.005,0.01,0.05,0.1,0.2,0.3,0.5]
        n_attack_steps = [10,15,40]
        seeds = [0]
        beta = 0.25
        dropout = 0.3
        attack_size_mismatch_speech = 0.1
        attack_size_mismatch_ecg = 0.1
        attack_size_mismatch_cnn = 0.1

        grid = [model for model in worst_case_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid_worst_case = configure(grid, {"mode":"direct"})
        grid_worst_case = split(grid_worst_case, "attack_size", attack_sizes)
        grid_worst_case = split(grid_worst_case, "n_attack_steps", n_attack_steps)
        
        if(config.FLAGS.jax_enable_x64):
            for g in grid_worst_case:
                for p in g["theta"]:
                    g["theta"][p] = g["theta"][p].astype(jnp.float64)

        grid_worst_case = run(grid_worst_case, min_whole_attacked_test_acc, n_threads=1, store_key="min_acc_test_set_acc")(1, "{*}", "{data_dir}", "{n_attack_steps}", "{attack_size}", 0.0, 0.001, 0.0)

        def _get_data_acc(architecture, beta, attack_size_mismatch, identifier, grid, n_attack_steps):
            robust_data = query(grid, identifier, where={"beta_robustness":beta, "attack_size_mismatch":attack_size_mismatch, "dropout_prob":0.0, "optimizer":"adam", "architecture":architecture, "n_attack_steps":n_attack_steps})
            robust_data_acc = onp.array([el[0] for el in robust_data])
            robust_data_loss = onp.array([el[1] for el in robust_data])

            vanilla_data = query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.0, "awp":False, "optimizer":"adam", "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_data_acc = onp.array([el[0] for el in vanilla_data])
            vanilla_data_loss = onp.array([el[1] for el in vanilla_data])

            vanilla_dropout_data = query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.3, "optimizer":"adam", "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_dropout_data_acc = onp.array([el[0] for el in vanilla_dropout_data])
            vanilla_dropout_data_loss = onp.array([el[1] for el in vanilla_dropout_data])

            vanilla_esgd_data = query(grid, identifier, where={"beta_robustness":0.0, "optimizer":"esgd", "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_esgd_data_acc = onp.array([el[0] for el in vanilla_esgd_data])
            vanilla_esgd_data_loss = onp.array([el[1] for el in vanilla_esgd_data])

            vanilla_abcd_data = query(grid, identifier, where={"beta_robustness":0.0, "optimizer":"abcd", "architecture":architecture, "n_attack_steps":n_attack_steps})
            vanilla_abcd_data_acc = onp.array([el[0] for el in vanilla_abcd_data])
            vanilla_abcd_data_loss = onp.array([el[1] for el in vanilla_abcd_data])

            return (vanilla_data_acc,vanilla_data_loss), (vanilla_dropout_data_acc,vanilla_dropout_data_loss), (robust_data_acc,robust_data_loss), (vanilla_esgd_data_acc,vanilla_esgd_data_loss), (vanilla_abcd_data_acc,vanilla_abcd_data_loss)

        def print_worst_case_test(data, attack_sizes, beta, dropout, n_attack_steps, typ, arch):
            print("\\begin{table}[!htb]\n\\begin{tabular}{llll}")
            if(typ == ACC):
                print("%s \t %s \t %s \t %s \t %s \t %s" % ("Attack size          & ","Test acc. ($\\beta=0$) & ",f"Test acc. (dropout = {dropout}) & ",f"Test acc. ($\\beta={beta}$) &", "Test acc. (ESGD)     & ", "Test acc. (ABCD)    \\\\"))
            else:
                print("%s \t %s \t %s \t %s \t %s \t %s" % ("Attack size          & ","Loss ($\\beta=0$)       & ",f"Loss (dropout = {dropout}) & ",f"Loss ($\\beta={beta}$) &", "Loss (ESGD)          & ", "Loss (ABCD)          \\\\"))
            for idx,attack_size in enumerate(attack_sizes):
                m = 1
                if(typ == ACC):
                    m = 100 # - Percentage
                dn = 100*onp.ravel(data[0][typ])[idx]
                dnd = 100*onp.ravel(data[1][typ])[idx]
                dr = 100*onp.ravel(data[2][typ])[idx]
                desgd = 100*onp.ravel(data[3][typ])[idx]
                dabcd = 100*onp.ravel(data[4][typ])[idx]
                print("%.3f & \t\t\t %.2f & \t\t\t %.2f & \t\t\t %.2f \t\t\t\t %.2f \t\t\t\t %.2f \\\\" % (attack_size,dn,dnd,dr,desgd,dabcd))
            print("\\end{tabular}")
            typ_string = "Loss"
            if(typ == ACC):
                typ_string = "Acc."
            print("\\caption{Architecture",arch," Type",typ_string," N",str(n_attack_steps),"}")
            print("\\end{table}")

        def _plot(ax, data, typ=ACC, labels=None, ylabel=None, title=None):
            colors = ["#4c84e6","#fc033d","#03fc35","#77fc03","#f803fc"]
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

        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        axes = get_axes_worst_case(fig, N_rows=3, N_cols=3, attack_sizes=attack_sizes)
        def ij_x(i,j):
            return i*3+j

        for idx,n in enumerate(n_attack_steps):

            data_ecg_worst_case = _get_data_acc("ecg_lsnn", beta, attack_size_mismatch_ecg, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_speech_worst_case = _get_data_acc("speech_lsnn", beta, attack_size_mismatch_speech, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_cnn_worst_case = _get_data_acc("cnn", beta, attack_size_mismatch_cnn, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)

            if(idx == 0):
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=ACC, ylabel="Test acc. Speech", labels=["Normal","Dropout","Robust","ESGD","ABCD"], title=(r"$N_{\textnormal{steps}}=$ %s" % str(n)))
                _plot(axes[ij_x(1,idx)], data_ecg_worst_case, typ=ACC, ylabel="Test acc. ECG")
                _plot(axes[ij_x(2,idx)], data_cnn_worst_case, typ=ACC, ylabel="Test acc. CNN")
                
            else:
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=ACC, title=(r"$N_{\textnormal{steps}}=$ %s" % str(n)))
                _plot(axes[ij_x(1,idx)], data_ecg_worst_case, typ=ACC)
                _plot(axes[ij_x(2,idx)], data_cnn_worst_case, typ=ACC)

            print("\n")
            print_worst_case_test(data_ecg_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC, arch="ECG")
            print("\n")
            print_worst_case_test(data_speech_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC, arch="Speech")
            print("\n")
            print_worst_case_test(data_cnn_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=ACC, arch="CNN")

        plt.savefig("Resources/Figures/figure_worst_case_test_acc.pdf", dpi=1200)
        plt.show()

        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
        axes = get_axes_worst_case(fig, N_rows=3, N_cols=3, attack_sizes=attack_sizes)

        for idx,n in enumerate(n_attack_steps):

            data_ecg_worst_case = _get_data_acc("ecg_lsnn", beta, attack_size_mismatch_ecg, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_speech_worst_case = _get_data_acc("speech_lsnn", beta, attack_size_mismatch_speech, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)
            data_cnn_worst_case = _get_data_acc("cnn", beta, attack_size_mismatch_cnn, "min_acc_test_set_acc", grid_worst_case, n_attack_steps=n)

            if(idx == 0):
                
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=LOSS, ylabel="Loss Speech")
                _plot(axes[ij_x(1,idx)], data_ecg_worst_case, typ=LOSS, ylabel="Loss ECG")
                _plot(axes[ij_x(2,idx)], data_cnn_worst_case, typ=LOSS, ylabel="Loss CNN")
            else:
                _plot(axes[ij_x(0,idx)], data_speech_worst_case, typ=LOSS)
                _plot(axes[ij_x(1,idx)], data_ecg_worst_case, typ=LOSS)
                _plot(axes[ij_x(2,idx)], data_cnn_worst_case, typ=LOSS)

            print("\n")
            print_worst_case_test(data_ecg_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS, arch="ECG")
            print("\n")
            print_worst_case_test(data_speech_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS, arch="Speech")
            print("\n")
            print_worst_case_test(data_cnn_worst_case, attack_sizes, beta, dropout, n_attack_steps=n, typ=LOSS, arch="CNN")

        plt.savefig("Resources/Figures/figure_worst_case_KL.pdf", dpi=1200)
        plt.show()