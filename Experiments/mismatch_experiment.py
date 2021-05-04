from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class mismatch_experiment:
    
    @staticmethod
    def train_grid():
        seeds = [0]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "dropout_prob": 0.3})
        ecg3 = configure(ecg, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.1,0.01", "n_epochs":"20,10"})
        ecg4 = configure(ecg, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        ecg5 = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg = ecg0 + ecg1 + ecg2 + ecg3 + ecg4 + ecg5

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = configure(speech, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        speech2 = configure(speech, {"beta_robustness": 0.0, "dropout_prob":0.3})
        speech3 = configure(speech, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"40,10"})
        speech4 = configure(speech, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        speech5 = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        speech = speech0 + speech1 + speech2 + speech3 + speech4 + speech5

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"10,5"})
        cnn_grid4 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"10,2", "learning_rate":"0.001,0.0001", "abcd_etaA":0.001})
        cnn_grid5 = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry", "learning_rate":"0.0001,0.00001"})
        cnn_grid = cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3 + cnn_grid4 + cnn_grid5

        return ecg + speech + cnn_grid

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.2,0.3,0.5,0.7,0.9]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0,0.2,0.3,0.5,0.7,0.9]
        seeds = [0]
        dropout = 0.3
        beta_ecg = 0.25
        beta_speech = 0.25
        beta_cnn = 0.25
        attack_size_mismatch_speech = 0.1
        attack_size_mismatch_ecg = 0.1
        attack_size_mismatch_cnn = 0.1
        awp_gamma = 0.1

        # - Per general column
        N_cols = 10 # - 10
        N_rows = 24 # - Will be determined by subplot that has the most rows

        fig = plt.figure(figsize=(12, 5), constrained_layout=False)
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
        grid_mm = run(grid_mm, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        def unravel(arr):
            mm_lvls = arr.shape[1]
            res = [[] for _ in range(mm_lvls)]
            for seed in range(len(seeds)):
                for i in range(mm_lvls):
                    res[i].extend(list(arr[seed,i]))
            return res

        def _get_data_acc(architecture, beta,attack_size_mismatch, awp_gamma, identifier, grid):
            robust_data = onp.array(query(grid, identifier, where={"beta_robustness":beta, "attack_size_mismatch":attack_size_mismatch, "dropout_prob":0.0, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.0, "awp":False, "optimizer":"adam", "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_dropout_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "dropout_prob":0.3, "architecture":architecture})).reshape((len(seeds),-1))
            abcd_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "optimizer":"abcd", "architecture":architecture})).reshape((len(seeds),-1))
            esgd_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "optimizer":"esgd", "architecture":architecture})).reshape((len(seeds),-1))
            awp_data = onp.array(query(grid, identifier, where={"beta_robustness":0.0, "awp":True, "awp_gamma":awp_gamma, "boundary_loss":"madry", "architecture":architecture})).reshape((len(seeds),-1))
            return vanilla_data, vanilla_dropout_data ,robust_data, abcd_data, esgd_data, awp_data

        def get_data_acc(architecture, beta, attack_size_mismatch,identifier, grid):
            vanilla_data, vanilla_dropout_data, robust_data, abcd_data, esgd_data, awp_data = _get_data_acc(architecture, beta,attack_size_mismatch, awp_gamma, identifier, grid)
            return list(zip(unravel(vanilla_data), unravel(vanilla_dropout_data), unravel(robust_data), unravel(abcd_data), unravel(esgd_data), unravel(awp_data)))

        val_acc_speech = [max(a[0]) for a in _get_data_acc("speech_lsnn", beta_speech, attack_size_mismatch_speech, awp_gamma, "validation_accuracy", grid)]
        val_acc_ecg = [max(a[0]) for a in _get_data_acc("ecg_lsnn", beta_ecg, attack_size_mismatch_ecg, awp_gamma, "validation_accuracy", grid)]
        val_acc_cnn = [max(a[0]) for a in _get_data_acc("cnn", beta_cnn, attack_size_mismatch_cnn, awp_gamma, "validation_accuracy", grid)]

        data_speech_lsnn = get_data_acc("speech_lsnn", beta_speech, attack_size_mismatch_speech, "mismatch_list", grid_mm)
        data_ecg_lsnn = get_data_acc("ecg_lsnn", beta_ecg, attack_size_mismatch_ecg, "mismatch_list", grid_mm)
        data_cnn = get_data_acc("cnn", beta_cnn, attack_size_mismatch_cnn, "mismatch_list", grid_mm)

        plot_mm_distributions(axes_speech["btm"], data=data_speech_lsnn, labels=["Normal","Robust"])
        plot_mm_distributions(axes_ecg["btm"], data=data_ecg_lsnn, labels=["Normal","Robust"])
        plot_mm_distributions(axes_cnn["btm"], data=data_cnn, labels=["Normal","Robust"],legend=True)

        # - Get the sample data for speech
        X_speech, y_speech = get_data("speech")
        X_ecg, y_ecg = get_data("ecg")
        X_cnn, y_cnn = get_data("cnn")

        plot_images(axes_cnn["top"], X_cnn, y_cnn)
        plot_spectograms(axes_speech["top"], X_speech, y_speech)
        plot_ecg(axes_ecg["top"], X_ecg, y_ecg)

        axes_speech["btm"][0].set_ylabel("Accuracy")
        axes_speech["btm"][2].text(x = -0.5, y = -0.2, s="Mismatch level")

        plt.savefig("Resources/Figures/figure_main.pdf", dpi=1200)
        plt.show()

        def print_experiment_info(data, mismatch_levels, beta, dropout):
            print("\\begin{table}[!htb]\n\\begin{tabular}{lllll}")
            print("%s \t\t %s \t %s \t %s \t %s \t %s \t %s" % ("Mismatch level","Test acc. ($\\beta=0$)",f"Test acc. (dropout = {dropout})",f"Test acc. ($\\beta={beta}$)","Test acc. ABCD", "Test acc. ESGD", "Test acc. AWP"))
            for idx,mm in enumerate(mismatch_levels):
                dn = 100*onp.array(data[idx][0])
                dnd = 100*onp.array(data[idx][1])
                dr = 100*onp.array(data[idx][2])
                dabcd = 100*onp.array(data[idx][3])
                desgd = 100*onp.array(data[idx][4])
                dawp = 100*onp.array(data[idx][5])
                mn = onp.mean(dn)
                mnd = onp.mean(dnd)
                mr = onp.mean(dr)
                mabcd = onp.mean(dabcd)
                mesgd = onp.mean(desgd)
                mawp = onp.mean(dawp)
                sn = onp.std(dn)
                snd = onp.std(dnd)
                sr = onp.std(dr)
                sabcd = onp.std(dabcd)
                sesgd = onp.std(desgd)
                sawp = onp.std(dawp)
                print("%.2f \t\t\t %.2f$\pm$%.2f \t %.2f$\pm$%.2f \t\t %.2f$\pm$%.2f \t\t %.2f$\pm$%.2f \t %.2f$\pm$%.2f \t %.2f$\pm$%.2f" % (mm,mn,sn,mnd,snd,mr,sr,mabcd,sabcd,mesgd,sesgd,mawp,sawp))
            print("\\end{table} \n")

        def print_val_acc(data):
            print("%s \t %s \t %s \t %s \t %s \t %s" % ("Val acc. normal",f"Val acc. dropout",f"Val acc. robust","Val acc. ABCD", "Val acc. ESGD", "Val acc. AWP"))
            dn = 100*data[0]
            dnd = 100*data[1]
            dr = 100*data[2]
            dabcd = 100*data[3]
            desgd = 100*data[4]
            dawp = 100*data[5]
            print("%.2f \t\t\t %.2f \t\t\t %.2f \t\t\t %.2f \t\t %.2f \t\t %.2f" % (dn,dnd,dr,dabcd,desgd,dawp))

        print_val_acc(val_acc_speech)
        print_val_acc(val_acc_ecg)
        print_val_acc(val_acc_cnn)

        print_experiment_info(data_speech_lsnn, speech_mm_levels, beta_speech, dropout)
        print_experiment_info(data_ecg_lsnn, ecg_mm_levels, beta_ecg, dropout)
        print_experiment_info(data_cnn, cnn_mm_levels, beta_cnn, dropout)

        
