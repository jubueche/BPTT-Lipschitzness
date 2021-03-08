from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class mismatch_experiment:
    
    @staticmethod
    def train_grid():
        seeds = [0]

        ecg = ecg_lsnn.make()
        ecg = configure([ecg], dictionary={"attack_size_constant": 0.0, "initial_std_constant":0.0, "initial_std_mismatch":0.001, "seed":0})
        ecg0 = configure(ecg, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "n_epochs":"80,20"})
        ecg1 = configure(ecg, {"beta_robustness": 0.125, "attack_size_mismatch":0.2, "n_epochs":"80,20"})
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "n_epochs":"80,20", "dropout_prob": 0.3})
        ecg = ecg0 + ecg1 + ecg2

        speech = speech_lsnn.make()
        speech = configure([speech], dictionary={"attack_size_constant": 0.0, "initial_std_constant":0.0, "n_epochs":"100,50", "learning_rate":"0.001,0.0001", "initial_std_mismatch":0.001, "seed":0, "wanted_words":"yes,no,up,down,left,right"})
        speech0 = configure(speech, {"beta_robustness": 0.0, "attack_size_mismatch":0.3})
        speech1 = configure(speech, {"beta_robustness": 0.125, "attack_size_mismatch":0.3})
        speech2 = configure(speech, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "dropout_prob":0.3})
        speech = speech0 + speech1 + speech2

        cnn_grid = cnn.make()
        cnn_grid = split(cnn_grid, "attack_size_constant", [0.0])
        cnn_grid = split(cnn_grid, "attack_size_mismatch", [1.0])
        cnn_grid = split(cnn_grid, "initial_std_constant", [0.0])
        cnn_grid = split(cnn_grid, "initial_std_mismatch", [0.001])
        cnn_grid = split(cnn_grid, "beta_robustness", [0.0, 1.0])
        cnn_grid = split(cnn_grid, "seed", seeds)

        return speech
        # return ecg + speech + cnn_grid

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.2,0.3,0.5,0.7,0.9]
        ecg_mm_levels = [0.0, 0.2,0.3,0.5,0.7,0.9]
        cnn_mm_levels = [0.0, 0.5,0.7,0.9,1.1,1.5]
        seeds = [0]

        # - Per general column
        N_cols = 10 # - 10
        N_rows = 24 # - Will be determined by subplot that has the most rows

        fig = plt.figure(figsize=(14, 5), constrained_layout=False)
        hs = 5
        gridspecs = [fig.add_gridspec(N_rows, N_cols, left=0.05, right=0.31, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.35, right=0.61, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.65, right=0.98, hspace=hs)]

        axes_speech = get_axes_main_figure(fig, gridspecs[0], N_cols, N_rows, "speech", mismatch_levels=speech_mm_levels[1:], btm_ylims=[0.0,1.0])
        axes_ecg = get_axes_main_figure(fig, gridspecs[1], N_cols, N_rows, "ecg", mismatch_levels=ecg_mm_levels[1:], btm_ylims=[0.0,1.0])
        axes_cnn = get_axes_main_figure(fig, gridspecs[2], N_cols, N_rows, "cnn", mismatch_levels=cnn_mm_levels[1:], btm_ylims=[0.0,1.0])

        grid = [model for model in mismatch_experiment.train_grid() if model["seed"] in seeds] 

        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid = split(grid, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid = split(grid, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid = configure(grid, {"n_iterations":100})
        grid = configure(grid, {"n_iterations":1}, where={"mm_level":0.0})

        grid = configure(grid, {"mode":"direct"})
        
        grid = run(grid, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        def unravel(arr):
            mm_lvls = arr.shape[1]
            res = [[] for _ in range(mm_lvls)]
            for seed in range(len(seeds)):
                for i in range(mm_lvls):
                    res[i].extend(list(arr[seed,i]))
            return res

        def get_data_acc(architecture, beta):
            robust_data = onp.array(query(grid, "mismatch_list", where={"beta_robustness":beta, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, "mismatch_list", where={"beta_robustness":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            return list(zip(unravel(vanilla_data), unravel(robust_data)))

        data_speech_lsnn = get_data_acc("speech_lsnn", 0.125)
        # data_ecg_lsnn = get_data_acc("ecg_lsnn", 0.125)
        # data_cnn = get_data_acc("cnn", 1.0)

        plot_mm_distributions(axes_speech["btm"], data=data_speech_lsnn)
        # plot_mm_distributions(axes_ecg["btm"], data=data_ecg_lsnn)
        # plot_mm_distributions(axes_cnn["btm"], data=data_cnn)

        # - Get the sample data for speech
        X_speech, y_speech = get_data("speech")
        X_ecg, y_ecg = get_data("ecg")
        X_cnn, y_cnn = get_data("cnn")

        plot_images(axes_cnn["top"], X_cnn, y_cnn)
        plot_spectograms(axes_speech["top"], X_speech, y_speech)
        plot_ecg(axes_ecg["top"], X_ecg, y_ecg)

        axes_speech["btm"][0].set_ylabel("Accuracy")
        axes_speech["btm"][2].text(x = -0.5, y = -0.2, s="Mismatch level")

        plt.savefig("Resources/Figures/figure_main.png", dpi=1200)
        plt.savefig("Resources/Figures/figure_main.pdf", dpi=1200)
        plt.show()

        def print_experiment_info(data, mismatch_levels, beta):
            print("%s \t\t %s \t %s \t %s \t %s" % ("Mismatch level","Test acc. ($\\beta=0$)",f"Test acc. ($\\beta={beta}$)","$\Delta$ Acc.","P-Value"))
            for idx,mm in enumerate(mismatch_levels):
                dn = 100*onp.array(data[idx][0])
                dr = 100*onp.array(data[idx][1])
                mn = onp.mean(dn)
                mr = onp.mean(dr)
                sn = onp.std(dn)
                sr = onp.std(dr)
                d = mr-mn
                p = stats.mannwhitneyu(data[idx][0], data[idx][1])[1]
                print("%.2f \t\t\t %.2f$\pm$%.2f \t %.2f$\pm$%.2f \t\t %.2f \t\t %.3E" % (mm,mn,sn,mr,sr,d,p))

        beta_speech = onp.unique(query(grid, "beta_robustness", where={"architecture": "speech_lsnn"}))[1]
        print_experiment_info(data_speech_lsnn, speech_mm_levels, beta_speech)

        # beta_ecg = onp.unique(query(grid, "beta_robustness", where={"architecture": "ecg_lsnn"}))[1]
        # print("---------------------------")
        # print_experiment_info(data_ecg_lsnn, ecg_mm_levels, beta_ecg)

        # beta_cnn = onp.unique(query(grid, "beta_robustness", where={"architecture": "cnn"}))[1]
        # print("---------------------------")
        # print_experiment_info(data_cnn, cnn_mm_levels, beta_cnn)
