from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import dj, split, configure
from experiment_utils import test_accuracy_after_mismatch_attack

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
        def setup_grid(grid, mm_levels):
            mm0 = configure(grid,{"mm_level":0.0, "n_test_iterations":1})
            
            mm = split(grid, "mm_level", mm_levels)
            mm = configure(mm, {"n_test_iterations":1})
            
            combined = configure(mm0 + mm, {"test_accuracy_function": test_accuracy_after_mismatch_attack, "test_accuracy_dependencies":["mm_level", "{architecture}_session_id"]})
            return run("test_accuracy", combined)
        
        grid = mismatch_experiment.train_grid()

        trained_models = dj(grid, "train", run_mode="load")("{*}")
        trained_models = split(trained_models, "mm_level", [0.0,0.1,0.2,0.3,0.5,0.7], where= {"architecture":"ecg_lsnn"})



        # ecg = run(load, grids["ecg"], "ecg_lsnn")
        # ecg = run(ecg_lsnn.loader, ecg)
        # ecg = setup_grid(ecg, [0.1,0.2,0.3,0.5,0.7])

        # speech = get("speech_lsnn",grids["speech_lsnn"],make_hyperparameters_speech(),checker=speech_lsnn_checker)
        # speech = map(speech_lsnn_loader, speech)
        # speech = setup_grid(speech,[0.5,0.7,0.9,1.1,1.5])

        # cnn = get("cnn", grids.cnn_mm_exp, make_hyperparameters_cnn(),checker=cnn_checker)
        # cnn = map(cnn_loader, cnn)
        # cnn = setup_grid(cnn, [0.5,0.7,0.9,1.1,1.5])

        # ecg[0][""]
        



        # # - Main order is speech, ECG and CNN
        # mm_levels = get_mm_levels()
        # # - Per general column
        # N_cols = 2*len(mm_levels[0]) # - 10
        # N_rows = 24 # - Will be determined by subplot that has the most rows

        # fig = plt.figure(figsize=(14, 5), constrained_layout=False)
        # hs = 5
        # gridspecs = [fig.add_gridspec(N_rows, N_cols, left=0.05, right=0.31, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.35, right=0.61, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.65, right=0.98, hspace=hs)]

        # axes_speech = get_axes_main_figure(fig, gridspecs[0], N_cols, N_rows, "speech", mismatch_levels=mm_levels[0], btm_ylims=[0.0,0.95])
        # axes_ecg = get_axes_main_figure(fig, gridspecs[1], N_cols, N_rows, "ecg", mismatch_levels=mm_levels[1], btm_ylims=[0.0,0.95])
        # axes_cnn = get_axes_main_figure(fig, gridspecs[2], N_cols, N_rows, "cnn", mismatch_levels=mm_levels[2], btm_ylims=[0.0,0.95])

        # # - Fill the plots with data
        # plot_mm_distributions(axes_speech["btm"], data=[(get_mismatch_data(mm/2.5),get_mismatch_data(mm/2)) for mm in mm_levels[0][::-1]])
        # plot_mm_distributions(axes_ecg["btm"], data=[(get_mismatch_data(mm/1.5),get_mismatch_data(mm/1)) for mm in mm_levels[1][::-1]])
        # plot_mm_distributions(axes_cnn["btm"], data=[(get_mismatch_data(mm/2.5),get_mismatch_data(mm/2)) for mm in mm_levels[2][::-1]])

        # # - Get the sample data for speech
        # X_speech, y_speech = get_data("speech")
        # X_ecg, y_ecg = get_data("ecg")
        # X_cnn, y_cnn = get_data("cnn")

        # plot_images(axes_cnn["top"], X_cnn, y_cnn)
        # plot_spectograms(axes_speech["top"], X_speech, y_speech)
        # plot_ecg(axes_ecg["top"], X_ecg, y_ecg)

        # axes_speech["btm"][0].set_ylabel("Accuracy")
        # axes_speech["btm"][2].text(x = -0.5, y = -0.2, s="Mismatch level")

        # plt.savefig("Resources/Figures/figure_main.png", dpi=1200)
        # plt.show()