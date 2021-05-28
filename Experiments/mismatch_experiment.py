from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run, reduce_keys
from datajuicer.visualizers import *
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import numpy as np
import seaborn as sns

class mismatch_experiment:
    
    @staticmethod
    def train_grid():
        seeds = [0,1]

        ecg = [ecg_lsnn.make()]
        ecg0 = configure(ecg, {"beta_robustness": 0.0})
        ecg1 = configure(ecg, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        ecg2 = configure(ecg, {"beta_robustness": 0.0, "dropout_prob": 0.3})
        ecg3 = []#configure(ecg, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.1,0.01", "n_epochs":"20,10"})
        ecg4 = configure(ecg, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        ecg5 = []#configure(ecg, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001"})
        ecg6 = configure(ecg, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        ecg7 = configure(ecg, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        ecg = ecg0 + ecg1 + ecg2 + ecg3 + ecg4 + ecg5 + ecg6 + ecg7

        speech = [speech_lsnn.make()]
        speech0 = configure(speech, {"beta_robustness": 0.0})
        speech1 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1})
        speech2 = configure(speech, {"beta_robustness": 0.0, "dropout_prob":0.3})
        speech3 = [] # configure(speech, {"beta_robustness": 0.0, "optimizer": "esgd", "learning_rate":"0.001,0.0001", "n_epochs":"40,10"})
        speech4 = configure(speech, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        speech5 = [] #configure(speech, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"40,10", "learning_rate":"0.001,0.0001"})
        speech6 = configure(speech, {"beta_robustness": 0.0, "awp":True, "boundary_loss":"madry", "awp_gamma":0.1})
        # speech7 = configure(speech, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        speech8 = configure(speech, {"beta_robustness": 0.5, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        speech = speech0 + speech1 + speech2 + speech3 + speech4  + speech5 + speech6 + speech8

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = [] #configure(cnn_grid, {"beta_robustness": 0.0, "optimizer": "esgd", "n_epochs":"10,5"})
        cnn_grid4 = configure(cnn_grid, {"beta_robustness": 0.0, "noisy_forward_std":0.3})
        cnn_grid5 = [] #configure(cnn_grid, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"10,2"})
        cnn_grid6 = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry"})
        cnn_grid7 = configure(cnn_grid, {"beta_robustness": 0.1, "attack_size_mismatch": 0.1, "noisy_forward_std":0.3})
        cnn_grid = cnn_grid0 + cnn_grid2 + cnn_grid3 + cnn_grid4 + cnn_grid5 + cnn_grid6 + cnn_grid7

        final_grid = ecg + speech + cnn_grid
        final_grid = split(final_grid, "seed", seeds) + cnn_grid1

        return final_grid

    @staticmethod
    def visualize():

        speech_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        ecg_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        cnn_mm_levels = [0.0,0.1,0.2,0.3,0.5,0.7]
        seeds = [0,1]

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

        @visualizer(dim=4)
        def violin(table, axes_dict):
            def get_val(*args): 
                l = table.get_val(*args)
                if l is None:
                    return None
                return sum(l,[])
            shape= table.shape()
            for i0 in range(shape[0]):
                axes = axes_dict[table.get_label(axis=0, index=i0)]
                legend=False
                if i0 == shape[0]-1:
                    legend=True
                colors = ["#4c84e6","#000000"]

                offset = 0
                for i1 in range(shape[2])[1:]:
                    a_idx = i1-1
                    el = [np.array(get_val(i0, i2, i1, 0)).reshape((-1,)) for i2 in range(shape[1])]
                    el = [e for e in el if all(e != [None])]

                    # - Compute the P-value between them here
                    p_value_mw = stats.mannwhitneyu(el[0], el[1])[1]
                    print(f"P-Value {p_value_mw} Architecture{table.get_label(0,i0)} MM Level {table.get_label(2,i1)}")

                    x = []
                    y = []
                    hue = []
                    x = onp.hstack([x, [0] * (len(el[0]) + len(el[1]))])
                    y = onp.hstack([y, el[0]])
                    y = onp.hstack([y, el[1]])
                    hue = onp.hstack([hue, [0] * len(el[0])])
                    hue = onp.hstack([hue, [1] * len(el[1])])
                    sns.violinplot(ax = axes[a_idx],
                        x = x,
                        y = y,
                        split = True,
                        hue = hue,
                        inner = 'quartile', cut=0,
                        scale = "width", palette = colors, saturation=1.0, linewidth=0.5)
                    axes[a_idx].set_xticks([])

                    for l in axes[a_idx].lines[3:6]:
                        l.set_color('white')

                    if(not (legend and i1==shape[2]-1)):
                        axes[a_idx].get_legend().remove()
                
                if(legend):
                    a = axes[a_idx]
                    lines = [Line2D([0,0],[0,0], color=c, lw=3.) for c in colors]
                    labels = ["Standard","Ours"]
                    a.legend(lines, labels, frameon=False, loc=3, prop={'size': 7})

        label_dict = {
            "beta_robustness": "Beta",
            "optimizer": "Optimizer",
            "mismatch_list_mean": "Mean Acc.",
            "mismatch_list_std":"Std.",
            "mismatch_list_min":"Min.",
            "dropout_prob":"Dropout",
            "mm_level": "Mismatch",
            "cnn" : "CNN",
            "speech_lsnn": "Speech LSNN",
            "ecg_lsnn": "ECG LSNN",
            "awp": "AWP",
            "AWP = True":"AWP",
            "Beta = 0.25":"Beta 0.25",
            "Beta = 0.5":"Beta 0.5",
            "Beta = 0.2, Forward": "Forward Noise + Beta 0.2",
            "Beta = 0.3, Forward": "Forward Noise + Beta 0.3",
            "Beta = 0.5, Forward": "Forward Noise + Beta 0.5",
            "Beta 0.25, Forward": "Forward Noise + Beta 0.25",
            "Beta 0.5, Forward": "Forward Noise + Beta 0.5",
            "Beta = 0.1, Forward": "Forward Noise + Beta 0.1",
            "noisy_forward_std = 0.3": "Forward Noise",
            "Optimizer = abcd":"ABCD",
            "Optimizer = esgd":"ESGD"
        }
        order_violin = {
            "Method": ["Standard", "Forward + Beta"]
        }

        order = {
            "architecture": ["speech_lsnn", "ecg_lsnn", "cnn"],
            "Method": ["Forward Noise + Beta 0.1", "Forward Noise + Beta 0.5", "Beta 0.25", "Beta 0.5", "Standard", "Forward Noise"]
        }

        grid_plot = [g for g in grid_mm if g["optimizer"]=="adam" and not g["awp"] and g["dropout_prob"]==0.0 and ((g["beta_robustness"]==0.0 and g["noisy_forward_std"]==0) or (g["beta_robustness"]!=0.0 and g["noisy_forward_std"]!=0.0))]
        independent_keys = ["architecture", Table.Deviation_Var(default={"beta_robustness":0.0, "noisy_forward_std":0.0},label="Method"), "mm_level"]
        dependent_keys = ["mismatch_list"]
        axes_dict = {"Speech LSNN":axes_speech["btm"], "ECG LSNN":axes_ecg["btm"], "CNN":axes_cnn["btm"]}
        violin(grid_plot, independent_keys=independent_keys,dependent_keys=dependent_keys,label_dict=label_dict, axes_dict=axes_dict, order=order_violin)

        # - Get the sample data for speech
        X_speech, y_speech = get_data("speech")
        X_ecg, y_ecg = get_data("ecg")
        X_cnn, y_cnn = get_data("cnn")

        plot_images(axes_cnn["top"], X_cnn, y_cnn)
        plot_spectograms(axes_speech["top"], X_speech, y_speech)
        plot_ecg(axes_ecg["top"], X_ecg, y_ecg)

        axes_speech["btm"][0].set_ylabel("Test acc.")
        axes_ecg["btm"][2].text(x = -0.5, y = -0.2, s=r"Mismatch level ($\zeta$)")
        axes_cnn["btm"][2].text(x = -0.5, y = -0.2, s=r"Mismatch level ($\zeta$)")
        axes_speech["btm"][2].text(x = -0.5, y = -0.2, s=r"Mismatch level ($\zeta$)")

        plt.savefig("Resources/Figures/figure_main.pdf", dpi=1200)
        plt.show()

        group_by = ["architecture", "awp", "beta_robustness", "dropout_prob", "optimizer", "noisy_forward_std", "mm_level"]
        for g in grid_mm:
            g["mismatch_list"] = list(100 * np.array(g["mismatch_list"])) 
        reduced = reduce_keys(grid_mm, "mismatch_list", reduction={"mean": lambda l: float(np.mean(l)), "std": lambda l: float(np.std(l)), "min": lambda l: float(np.min(l))}, group_by=group_by)

        independent_keys = ["architecture",Table.Deviation_Var({"beta_robustness":0.0, "awp":False, "dropout_prob":0.0, "optimizer":"adam", "noisy_forward_std":0.0}, label="Method"),  "mm_level"]
        dependent_keys = ["mismatch_list_mean", "mismatch_list_std","mismatch_list_min"]
        

        print(latex(reduced, independent_keys, dependent_keys, label_dict, order, bold_order=[max,min,max]))

        reduced2 = reduce_keys(grid, "validation_accuracy", reduction=lambda a: float(100 * np.mean([np.max(aa) for aa in a])), group_by=group_by[:-1])
        print(latex(reduced2, independent_keys[:-1], dependent_keys=["validation_accuracy"], label_dict=label_dict, order=order))