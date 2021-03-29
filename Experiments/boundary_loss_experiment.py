from architectures import ecg_lsnn
from datajuicer import *
import matplotlib.pyplot as plt

class boundary_loss_experiment:
    @staticmethod
    def train_grid():
        grid = [ecg_lsnn.make()]
        grid = split(grid, "boundary_loss", ["kl", "reverse_kl", "js", "l2"])
        return grid
        
    @staticmethod
    def visualize():
        grid = boundary_loss_experiment.train_grid()
        grid = run(grid, func="train", run_mode="load", store_key="*")("{*}")

        fig, axs = plt.subplots(2)

        kl_valid_acc = query(grid, "validation_accuracy", {"boundary_loss":"kl"})[0][0:70]
        reverse_kl_valid_acc = query(grid, "validation_accuracy", {"boundary_loss":"reverse_kl"})[0][0:70]
        js_valid_acc = query(grid, "validation_accuracy", {"boundary_loss":"js"})[0][0:70]
        l2_valid_acc = query(grid, "validation_accuracy", {"boundary_loss":"l2"})[0][0:70]
        
        print(len(kl_valid_acc))
        print(len(reverse_kl_valid_acc))
        print(len(js_valid_acc))
        print(len(l2_valid_acc))

        axs[0].plot(kl_valid_acc, label= "kl")
        axs[0].plot(reverse_kl_valid_acc, label= "reverse kl")
        axs[0].plot(js_valid_acc, label= "js")
        axs[0].plot(l2_valid_acc, label= "l2")
        axs[0].legend()

        average = lambda l: [sum(x)/len(x) for x in l]

        kl_mm = average(query(grid, "mm_val_robustness", {"boundary_loss":"kl"})[0][0:35])
        reverse_kl_mm = average(query(grid, "mm_val_robustness", {"boundary_loss":"reverse_kl"})[0][0:35])
        js_mm = average(query(grid, "mm_val_robustness", {"boundary_loss":"js"})[0][0:35])
        l2_mm = average(query(grid, "mm_val_robustness", {"boundary_loss":"l2"})[0][0:35])
        

        print(len(kl_mm))
        print(len(reverse_kl_mm))
        print(len(js_mm))
        print(len(l2_mm))

        axs[1].plot(kl_mm, label= "kl")
        axs[1].plot(reverse_kl_mm, label= "reverse kl")
        axs[1].plot(js_mm, label= "js")
        axs[1].plot(l2_mm, label= "l2")
        axs[1].legend()

        plt.savefig("Resources/Figures/boundary_figure.png", dpi=1200)

        print("end")
        