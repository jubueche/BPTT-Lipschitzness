from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import dj, split, configure, query, djm
from experiment_utils import *
from matplotlib.lines import Line2D

class mismatch_experiment:
    
    @staticmethod
    def train_grid():
        ecg = ecg_lsnn.make()
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "beta_robustness", [0.0, 1.0])
        ecg = split(ecg, "seed", [0,1,2,3,4,5,6,7,8,9])

        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0, 1.0])
        speech = split(speech, "seed", [0,1,2,3,4,5,6,7,8,9])

        cnn_grid = cnn.make()
        cnn_grid = split(cnn_grid, "attack_size_constant", [0.0])
        cnn_grid = split(cnn_grid, "attack_size_mismatch", [1.0])
        cnn_grid = split(cnn_grid, "initial_std_constant", [0.0])
        cnn_grid = split(cnn_grid, "initial_std_mismatch", [0.001])
        cnn_grid = split(cnn_grid, "beta_robustness", [0.0, 1.0])
        cnn_grid = split(cnn_grid, "seed", [0,1,2,3,4,5,6,7,8,9])

        return ecg + speech + cnn_grid

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.5,0.7,0.9,1.1,1.5]
        ecg_mm_levels = [0.0, 0.2,0.3,0.5,0.7,0.9]
        cnn_mm_levels = [0.0, 0.5,0.7,0.9,1.1,1.5]
        seeds = [0,1,3,4,5,7]

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
        grid = djm(grid, "train", run_mode="load")("{*}")

        grid = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})
        grid = split(grid, "mm_level", ecg_mm_levels , where={"architecture":"ecg_lsnn"})
        grid = split(grid, "mm_level", cnn_mm_levels , where={"architecture":"cnn"})

        grid = configure(grid, {"n_iterations":50})
        grid = configure(grid, {"n_iterations":1}, where={"mm_level":0.0})

        grid = configure(grid, {"mode":"direct"})
        
        grid = djm(grid, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        def unravel(arr):
            mm_lvls = arr.shape[1]
            res = [[] for _ in range(mm_lvls)]
            for seed in range(len(seeds)):
                for i in range(mm_lvls):
                    res[i].extend(list(arr[seed,i]))
            return res

        def get_data_acc(architecture):
            robust_data = onp.array(query(grid, "mismatch_list", where={"beta_robustness":1.0, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, "mismatch_list", where={"beta_robustness":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            return list(zip(unravel(vanilla_data), unravel(robust_data)))

        plot_mm_distributions(axes_speech["btm"], data=get_data_acc("speech_lsnn"))
        plot_mm_distributions(axes_ecg["btm"], data=get_data_acc("ecg_lsnn"))
        plot_mm_distributions(axes_cnn["btm"], data=get_data_acc("cnn"))

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