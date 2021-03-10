from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class treat_as_constant_experiment:

    @staticmethod
    def train_grid():
        speech = speech_lsnn.make()
        speech = configure([speech], dictionary={"attack_size_constant":0.0,"attack_size_mismatch":0.3,"initial_std_constant":0.0, "initial_std_mismatch":0.001, "beta_robustness":0.125, "seed":0})
        speech = split(speech, "treat_as_constant", [True,False])
        speech = split(speech, "boundary_loss", ["kl","reverse_kl","l2"])
        return speech

    @staticmethod
    def visualize():
        grid = treat_as_constant_experiment.train_grid()
        grid = run(grid, "train", n_threads=8, run_mode="load", store_key="*")("{*}")

        treat_as_constant = [True, False]
        boundary_loss = ["kl", "reverse_kl", "l2"]
        b_loss_names = ["KL", "Rev. KL", "L2"]

        plt.figure(figsize=(10,5))
        c = 0
        for mode in treat_as_constant:
            for idx,loss_type in enumerate(boundary_loss):
                acc = query(grid, "validation_accuracy", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                attacked_acc = query(grid, "attacked_validation_accuracies", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                c += 1
                plt.plot(acc, label=f"Cons. {mode} BL {b_loss_names[idx]}", color=f"C{c}")
                plt.plot(attacked_acc, linestyle="--", color=f"C{c}")

        plt.legend()
        plt.show()