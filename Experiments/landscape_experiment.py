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
        grid_speech_ = speech_lsnn.make()
        grid_speech = configure([grid_speech_], dictionary={"attack_size_mismatch":0.1})
        grid_speech = split(grid_speech, "beta_robustness", betas)
        grid_speech += configure([grid_speech_], dictionary={"dropout_prob":0.0, "beta_robustness":0.0})
        grid_speech += configure([grid_speech_], dictionary={"dropout_prob":0.3, "beta_robustness":0.0})
        grid_speech += configure([grid_speech_], dictionary={"beta_robustness":0.0, "awp":True, "boundary_loss":"madry"})
        grid_speech = split(grid_speech, "seed", seeds)

        grid_ecg_ = ecg_lsnn.make()
        grid_ecg = configure([grid_ecg_], dictionary={"attack_size_mismatch":0.1})
        grid_ecg = split(grid_ecg, "beta_robustness", betas)
        grid_ecg += configure([grid_ecg_], dictionary={"dropout_prob":0.0, "beta_robustness":0.0})
        grid_ecg += configure([grid_ecg_], dictionary={"dropout_prob":0.3, "beta_robustness":0.0})
        grid_ecg += configure([grid_ecg_], dictionary={"beta_robustness":0.0, "awp":True, "boundary_loss":"madry"})
        grid_ecg = split(grid_ecg, "seed", seeds)

        grid_cnn_ = cnn.make()
        grid_cnn = configure([grid_cnn_], dictionary={"attack_size_mismatch":0.1})
        grid_cnn = split(grid_cnn, "beta_robustness", betas)
        grid_cnn += configure([grid_cnn_], dictionary={"dropout_prob":0.0, "beta_robustness":0.0})
        grid_cnn += configure([grid_cnn_], dictionary={"dropout_prob":0.3, "beta_robustness":0.0})
        grid_cnn += configure([grid_cnn_], dictionary={"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry", "learning_rate":"0.0001,0.00001"})
        grid_cnn = split(grid_cnn, "seed", seeds)

        return grid_speech + grid_ecg + grid_cnn

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

        grid = run(grid, get_landscape_sweep, n_threads=1, store_key="landscape")("{*}", num_steps, "{data_dir}", std, from_, to_, n_repeat)

        def get_data(arch):
            data_dict = {}
            for beta in betas:
                data_tmp = query(grid, "landscape", where={"beta_robustness":beta, "dropout_prob":0.0, "architecture":arch})
                data_dict[beta] = data_tmp
            data_dict["Dropout"] = query(grid, "landscape", where={"beta_robustness":0.0, "dropout_prob":0.3, "architecture":arch})
            data_dict["AWP"] = query(grid, "landscape", where={"beta_robustness":0.0, "awp":True, "boundary_loss":"madry"})
            return data_dict

        keys = betas + ["Dropout","AWP"]

        data_speech = get_data(arch="speech_lsnn")
        data_ecg = get_data(arch="ecg_lsnn")
        data_cnn = get_data(arch="cnn")

        fig = plt.figure(figsize=(10,4), constrained_layout=False)
        axes_speech = plt.subplot(1,3,1) # - Speech
        axes_speech.set_xlabel(r"$\alpha$")
        axes_speech.set_ylabel("Cross-entropy loss")
        axes_speech.spines['right'].set_visible(False)
        axes_speech.spines['top'].set_visible(False)

        axes_ecg = plt.subplot(1,3,2)
        axes_ecg.spines['right'].set_visible(False)
        axes_ecg.spines['top'].set_visible(False)

        axes_cnn = plt.subplot(1,3,3)
        axes_cnn.spines['right'].set_visible(False)
        axes_cnn.spines['top'].set_visible(False)

        def ma(x, N, fill=True): return onp.concatenate([x for x in [ [None]*(N // 2 + N % 2)*fill, onp.convolve(x, onp.ones((N,))/N, mode='valid'), [None]*(N // 2 -1)*fill, ] if len(x)]) 

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
                # axes_speech.plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes_speech.plot(onp.linspace(from_,to_,len(smoothed_mean_speech_over_seed)), smoothed_mean_speech_over_seed, c=colors[beta_idx], alpha=1.0, label=label)
            
            # for idx_d,d in enumerate(data_beta_ecg):
            #     axes_ecg.plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes_ecg.plot(onp.linspace(from_,to_,len(smoothed_mean_ecg_over_seed)), smoothed_mean_ecg_over_seed, c=colors[beta_idx], alpha=1.0)

            # for idx_d,d in enumerate(data_beta_cnn):
            #     axes_cnn.plot(onp.linspace(from_,to_,num_steps), d.T, c=colors[beta_idx], alpha=alpha_val)
            axes_cnn.plot(onp.linspace(from_,to_,len(smoothed_mean_cnn_over_seed)), smoothed_mean_cnn_over_seed, c=colors[beta_idx], alpha=1.0)

        axes_ecg.set_title("ECG")
        axes_speech.set_title("Speech")
        axes_cnn.set_title("CNN")
        axes_speech.legend(fontsize=5)
        plt.savefig("Resources/Figures/landscape.pdf", dpi=1200)
        plt.plot()
