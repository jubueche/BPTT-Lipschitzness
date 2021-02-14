from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class sgd_experiment:
    
    @staticmethod
    def train_grid():
        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0])
        speech = split(speech, "eval_step_interval", [1000])
        speech = split(speech, "learning_rate", ["0.01,0.001"])
        speech = split(speech, "optimizer", ["sgd"])
        speech1 = split(speech, "batch_size", [8])
        speech2 = split(speech, "batch_size", [128])
        speech1 = split(speech1, "n_epochs",["200,56"])
        speech2 = split(speech2, "n_epochs", ["200,88"])
        speech = speech1 + speech2
        speech = split(speech, "seed", [0,1,2,3,4,5,6,7,8,9])
        return speech

    @staticmethod
    def visualize():
        speech_mm_levels = [0.0,0.5,0.7,0.9,1.1,1.5]
        seeds = [0,1,2,3,4]
        grid = [model for model in sgd_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")

        grid = split(grid, "mm_level", speech_mm_levels, where={"architecture":"speech_lsnn"})

        grid = configure(grid, {"n_iterations":50})
        grid = configure(grid, {"n_iterations":1}, where={"mm_level":0.0})
        grid = configure(grid, {"mode":"direct"})
        grid = configure(grid, {"data_dir":"/cluster/scratch/jubueche/speech_dataset"})
        
        grid = run(grid, get_mismatch_list, n_threads=10, store_key="mismatch_list")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}", 100)

        def unravel(arr):
            mm_lvls = arr.shape[1]
            res = [[] for _ in range(mm_lvls)]
            for seed in range(len(seeds)):
                for i in range(mm_lvls):
                    res[i].extend(list(arr[seed,i]))
            return res

        def get_data_acc(architecture):
            robust_data = onp.array(query(grid, "mismatch_list", where={"batch_size":8, "architecture":architecture})).reshape((len(seeds),-1))
            vanilla_data = onp.array(query(grid, "mismatch_list", where={"batch_size":128, "architecture":architecture})).reshape((len(seeds),-1))
            return list(zip(unravel(vanilla_data), unravel(robust_data)))

        data_speech_lsnn = get_data_acc("speech_lsnn")
        fig = plt.figure(figsize=(7, 5), constrained_layout=False)
        gridspec = fig.add_gridspec(1,10)
        r = 5
        axes = [fig.add_subplot(gridspec[0,int(i*2):int((i+1)*2)]) for i in range(r)]
        for i,ax in enumerate(axes):
            ax.spines['left'].set_visible(ax.is_first_col())
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xlabel(speech_mm_levels[i+1])
            ax.set_xticks([])
            ax.set_ylim([0.0,1.0])
            if(i>0): ax.set_yticks([])
        plot_mm_distributions(axes, data=data_speech_lsnn)
        plt.savefig("Resources/Figures/figure_sgd.png", dpi=1200)
        plt.savefig("Resources/Figures/figure_sgd.pdf", dpi=1200)
        plt.show()
        plt.show()


