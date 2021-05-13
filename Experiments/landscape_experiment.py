from architectures import speech_lsnn, ecg_lsnn, cnn
from datajuicer import run, split, configure, query
from experiment_utils import *
import numpy as onp
import matplotlib as mpl

class landscape_experiment:

    @staticmethod
    def train_grid():
        betas = [0.25,0.5]
        seeds = [0]

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = split(speech, "beta_robustness", betas)
        speech1 = configure(speech1, {"attack_size_mismatch": 0.1})
        speech2 = configure(speech, {"beta_robustness": 0.0, "dropout_prob":0.3})
        speech3 = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        grid_speech = speech0 + speech1 + speech2 + speech3

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = split(ecg, "beta_robustness", betas)
        ecg1 = configure(ecg1, {"attack_size_mismatch": 0.1})
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "dropout_prob": 0.3})
        ecg3 = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        grid_ecg = ecg0 + ecg1 + ecg2 + ecg3

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = split(cnn_grid, "beta_robustness", betas)
        cnn_grid1 = configure(cnn_grid1, {"attack_size_mismatch": 0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry", "learning_rate":"0.0001,0.00001"})
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3

        final_grid = grid_ecg + grid_speech + cnn_grid
        final_grid = split(final_grid, "seed", seeds)

        return final_grid

    @staticmethod
    def visualize():
        seeds = [0]
        betas = [0.0,0.25,0.5]
        colors = ["#4c84e6","#fc033d","#03fc35","#f803fc","#eba434","#42e6f5","#f542e0"]
        grid = [model for model in landscape_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})

        num_steps = 100
        std = 0.2
        alpha_val = 0.2
        from_ = -2.0
        to_ = 2.0
        n_repeat = 15

        grid = run(grid, get_landscape_sweep, n_threads=1, run_mode="normal", store_key="landscape")("{*}", num_steps, "{data_dir}", std, from_, to_, n_repeat)

        def get_data(arch):
            data_dict = {}
            for beta in betas:
                data_tmp = query(grid, "landscape", where={"beta_robustness":beta, "dropout_prob":0.0, "architecture":arch})
                data_dict[beta] = data_tmp
            data_dict["Dropout"] = query(grid, "landscape", where={"beta_robustness":0.0, "dropout_prob":0.3, "architecture":arch})
            data_dict["AWP"] = query(grid, "landscape", where={"beta_robustness":0.0, "awp":True, "boundary_loss":"madry", "architecture":arch})
            return data_dict

        keys = betas + ["Dropout","AWP"]

        data_speech = get_data(arch="speech_lsnn")
        data_ecg = get_data(arch="ecg_lsnn")
        data_cnn = get_data(arch="cnn")

        fig = plt.figure(figsize=(12,3), constrained_layout=True)
        gridspec = fig.add_gridspec(1, 3, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        axes = [fig.add_subplot(gridspec[0,j]) for j in range(3)]
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[0].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_xlabel(r"$\alpha$")
        axes[1].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)
        axes[2].spines['top'].set_visible(False)
        axes[2].set_xlabel(r"$\alpha$")

        def ma(x, N, fill=True):
            return onp.concatenate([x for x in [ [None]*(N // 2 + N % 2)*fill, onp.convolve(x, onp.ones((N,))/N, mode='valid'), [None]*(N // 2 -1)*fill, ] if len(x)]) 

        def moving_average(x, N):
            result = onp.zeros_like(x)
            if x.ndim > 1:
                over = x.shape[0]
                for i in range(over):
                    result[i] = ma(x[i], N)
            else:
                result = ma(x, N)
            return result

        def get_ma(data,N=5):
            sum_data_over_seed = onp.array(data[0])
            for d in data[1:]:
                sum_data_over_seed += onp.array(d)
            # - Mean over seeds
            mean_data_over_seed = 1/len(data) * sum_data_over_seed
            smoothed_mean_data_over_seed = moving_average(mean_data_over_seed, N=N)
            return onp.mean(smoothed_mean_data_over_seed, axis=0)

        for beta_idx,beta in enumerate(keys):
            data_beta_speech = data_speech[beta]
            data_beta_ecg = data_ecg[beta]
            data_beta_cnn = data_cnn[beta]
            
            smoothed_mean_speech_over_seed = get_ma(data_beta_speech)
            smoothed_mean_ecg_over_seed = get_ma(data_beta_ecg)
            smoothed_mean_cnn_over_seed = get_ma(data_beta_cnn)

            label = None
            for idx_d,d in enumerate(data_beta_speech):
                if idx_d == 0:
                    if beta == 0.0:
                        label = "Normal"
                    elif beta == "Dropout":
                        label = "Dropout"
                    elif beta == "AWP":
                        label = "AWP"
                    else:
                        label = r"$\beta_{\textnormal{robust}}=$" + ("%s" % str(beta))               
                # axes[0].plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes[0].plot(onp.linspace(from_,to_,len(smoothed_mean_speech_over_seed)), smoothed_mean_speech_over_seed, c=colors[beta_idx], alpha=1.0, label=label)
            
            # for idx_d,d in enumerate(data_beta_ecg):
            #     axes[1].plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes[1].plot(onp.linspace(from_,to_,len(smoothed_mean_ecg_over_seed)), smoothed_mean_ecg_over_seed, c=colors[beta_idx], alpha=1.0)

            # for idx_d,d in enumerate(data_beta_cnn):
            #     axes[2].plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes[2].plot(onp.linspace(from_,to_,len(smoothed_mean_cnn_over_seed)), smoothed_mean_cnn_over_seed, c=colors[beta_idx], alpha=1.0)

        axes[1].set_title("ECG")
        axes[0].set_title("Speech")
        axes[2].set_title("CNN")
        axes[0].legend(fontsize=6)
        plt.savefig("Resources/Figures/landscape.pdf", dpi=1200)
        plt.plot()