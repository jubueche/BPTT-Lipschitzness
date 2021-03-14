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
        ecg = configure([ecg], dictionary={"initial_std_mismatch":0.001, "seed":0})
        ecg0 = configure(ecg, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "n_epochs":"150,50"})
        
        ecg1 = configure(ecg, {"beta_robustness": 0.125, "n_epochs":"150,50"})
        ecg1 = split(ecg1, "attack_size_mismatch", [0.2,0.3])
        
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "n_epochs":"150,50", "dropout_prob": 0.3})
        
        ecg3 = configure(ecg, {"beta_robustness": 0.125, "n_epochs":"150,50", "dropout_prob": 0.3})
        ecg3 = split(ecg3, "attack_size_mismatch", [0.2,0.3])
        ecg = ecg0 + ecg1 + ecg2 + ecg3

        speech = speech_lsnn.make()
        speech = configure([speech], dictionary={"initial_std_mismatch":0.001, "seed":0})
        speech0 = configure(speech, {"beta_robustness": 0.0, "attack_size_mismatch":0.3})

        speech1 = configure(speech, {"beta_robustness": 0.125})
        speech1 = split(speech1, "attack_size_mismatch", [0.2,0.3])
        
        speech2 = configure(speech, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "dropout_prob":0.3})

        speech3 = configure(speech, {"beta_robustness": 0.125, "dropout_prob": 0.3})
        speech3 = split(speech3, "attack_size_mismatch", [0.2,0.3])
        speech = speech0 + speech1 + speech2 + speech3

        cnn_grid = cnn.make()
        cnn_grid = configure([cnn_grid], dictionary={"initial_std_mismatch":0.001, "seed":0})
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0, "attack_size_mismatch":0.3})

        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.125})
        cnn_grid1 = split(cnn_grid1, "attack_size_mismatch", [0.2,0.3])

        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "attack_size_mismatch":0.3, "dropout_prob":0.3})

        cnn_grid3 = configure(cnn_grid, {"beta_robustness": 0.125, "dropout_prob": 0.3})
        cnn_grid3 = split(cnn_grid3, "attack_size_mismatch", [0.2,0.3])        
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3

        return ecg
        # return cnn_grid
        # return speech

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.2,0.3,0.5,0.7,0.9]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0, 0.5,0.7,0.9,1.1,1.5]

        ecg_attack_sizes = [0.0,0.005,0.01,0.05,0.1,0.2,0.3,0.5]
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

        grid_mm = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid_mm = split(grid_mm, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid_mm = split(grid_mm, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid_mm = configure(grid_mm, {"n_iterations":100})
        grid_mm = configure(grid_mm, {"n_iterations":1}, where={"mm_level":0.0})
        grid_mm = configure(grid_mm, {"mode":"direct"})

        grid_worst_case = configure(grid, {"mode":"direct"})
        grid_worst_case = split(grid_worst_case, "attack_size", ecg_attack_sizes)
        
        grid_worst_case = run(grid_worst_case, min_whole_attacked_test_acc, n_threads=1, store_key="min_acc_test_set_acc")(5, "{*}", "{data_dir}", 10, "{attack_size}", 0.0, 0.001, 0.0)
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        def unravel(arr):
            mm_lvls = arr.shape[1]
            res = [[] for _ in range(mm_lvls)]
            for seed in range(len(seeds)):
                for i in range(mm_lvls):
                    res[i].extend(list(arr[seed,i]))
            return res

        def _get_data_acc(architecture, beta, identifier, grid):
            robust_data = onp.array(query(grid, identifier, where={"beta_robustness":beta, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            return vanilla_data, robust_data

        def get_data_acc(architecture, beta, identifier, grid):
            vanilla_data, robust_data = _get_data_acc(architecture, beta, identifier, grid)
            return list(zip(unravel(vanilla_data), unravel(robust_data)))

        # data_speech_lsnn = get_data_acc("speech_lsnn", 0.125, "mismatch_list", grid_mm)
        data_ecg_lsnn = get_data_acc("ecg_lsnn", 0.125, "mismatch_list", grid_mm)
        # data_cnn = get_data_acc("cnn", 1.0, "mismatch_list", grid_mm)

        data_ecg_worst_case = _get_data_acc("ecg_lsnn", 0.125, "min_acc_test_set_acc", grid_worst_case)

        # plot_mm_distributions(axes_speech["btm"], data=data_speech_lsnn)
        plot_mm_distributions(axes_ecg["btm"], data=data_ecg_lsnn)
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

        def print_worst_case_test(data, attack_sizes, beta):
            print("%s \t\t %s \t %s" % ("Attack size","Test acc. ($\\beta=0$)",f"Test acc. ($\\beta={beta}$)"))
            for idx,attack_size in enumerate(attack_sizes):
                dn = 100*onp.ravel(data[0])[idx]
                dr = 100*onp.ravel(data[1])[idx]
                print("%.3f \t\t\t %.2f \t\t\t %.2f" % (attack_size,dn,dr))


        # beta_speech = onp.unique(query(grid_mm, "beta_robustness", where={"architecture": "speech_lsnn"}))[1]
        # print_experiment_info(data_speech_lsnn, speech_mm_levels, beta_speech)

        beta_ecg = onp.unique(query(grid_mm, "beta_robustness", where={"architecture": "ecg_lsnn"}))[1]
        print("---------------------------")
        print_experiment_info(data_ecg_lsnn, ecg_mm_levels, beta_ecg)

        # beta_cnn = onp.unique(query(grid_mm, "beta_robustness", where={"architecture": "cnn"}))[1]
        # print("---------------------------")
        # print_experiment_info(data_cnn, cnn_mm_levels, beta_cnn)

        print("---------------------------")
        print_worst_case_test(data_ecg_worst_case, ecg_attack_sizes, beta_ecg)

        
