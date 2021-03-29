from architectures import ecg_lsnn
from datajuicer import configure, run, query
import matplotlib.pyplot as plt
from functools import reduce

class saturation_experiment:
    @staticmethod
    def train_grid():
        grid = [ecg_lsnn.make()]
        grid = configure(grid, {"eval_step_interval":25})
        return grid
    
    @staticmethod
    def visualize():
        running_max = lambda l: reduce(lambda acc, ele: acc + [max(acc[-1],ele)], l, [l[0]])
        grid = saturation_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        
        va = grid[0]["validation_accuracy"]
        varm = running_max(va)

        mm = [sum(l)/len(l) for l in grid[0]["mm_val_robustness"] for _ in (0,1)]
        mmrm = running_max(mm)

        plt.plot(va[120:-1], label="Validation Accuracy")
        plt.plot(varm[120:-1])
        plt.plot(mm[120:-1], label="MM attacked Valid. Acc")
        plt.plot(mmrm[120:-1])
        plt.legend()
        plt.savefig("Resources/Figures/saturation_figure.png", dpi=1200)
        plt.savefig("Resources/Figures/saturation_figure.pdf", dpi=1200)


        print("done")