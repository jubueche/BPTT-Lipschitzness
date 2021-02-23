from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class treat_as_constant_experiment:

    @staticmethod
    def train_grid():
        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "seed", [0])
        speech_beta0 = split(speech, "beta_robustness", [0.0])
        speech_beta1 = split(speech, "beta_robustness", [1.0])
        speech_beta1 = split(speech_beta1, "treat_as_constant", ["False","True"])
        speech_beta1 = split(speech_beta1, "boundary_loss", ["kl","reverse_kl","l2"])
        speech = speech_beta1
        
        return speech

    @staticmethod
    def visualize():
        seeds = [0]
        grid = [model for model in treat_as_constant_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", n_threads=8, run_mode="load", store_key="*")("{*}")

        treat_as_constant = ["True", "False"]
        boundary_loss = ["kl", "reverse_kl", "l2"]
        b_loss_names = ["KL", "Rev. KL", "L2"]

        plt.figure(figsize=(10,5))
        c = 0

        for mode in treat_as_constant:
            for idx,loss_type in enumerate(boundary_loss):
                try:
                    training_acc = query(grid, "training_accuracy", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                    attacked_training_acc = query(grid, "attacked_training_accuracy", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                except:
                    training_acc = query(grid, "training_accuracies", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                    attacked_training_acc = query(grid, "attacked_training_accuracies", where={"treat_as_constant": mode, "boundary_loss": loss_type})[0]
                c += 1
                plt.plot(training_acc, label=f"Cons. {mode} BL {b_loss_names[idx]}", color=f"C{c}")
                plt.plot(attacked_training_acc, linestyle="--", color=f"C{c}")

        plt.legend()
        plt.show()