from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class regularization_comparison_experiment:
    
    @staticmethod
    def train_grid():
        
        seeds = [0]
        top = speech_lsnn.make()
        top = split(top, "attack_size_constant", [0.0])
        top = split(top, "attack_size_mismatch", [2.0])
        top = split(top, "initial_std_constant", [0.0])
        top = split(top, "initial_std_mismatch", [0.001])
        top = split(top, "beta_robustness", [0.0])
        
        # - SGD models with large and small bs-lr ratio
        sgd = split(top, "eval_step_interval", [1000])
        sgd = split(sgd, "optimizer", ["sgd"])
        sgd1 = split(sgd, "batch_size", [8])
        sgd1 = split(sgd1, "learning_rate", ["0.01,0.001"])
        sgd2 = split(sgd, "batch_size", [128])
        sgd2 = split(sgd2, "learning_rate", ["0.1,0.01"])
        sgd1 = split(sgd1, "n_epochs",["200,56"])
        sgd2 = split(sgd2, "n_epochs", ["200,88"])
        sgd = sgd1 + sgd2
        sgd = split(sgd, "seed", seeds)

        # - Normal
        normal = split(top, "seed", seeds)

        # - Dropout
        dropout = split(top, "dropout_prob", [0.3,0.5,0.7])
        dropout = split(dropout, "n_epochs",["200,56"])
        dropout = split(dropout, "seed", seeds)

        # - L2 decay
        l2 = split(top, "l2_weight_decay", [10.0,1.0,0.1])
        l2 = split(l2, "n_epochs",["200,56"])
        l2 = split(l2, "l2_weight_decay_params", ["['W_rec']"])
        l2 = split(l2, "seed", seeds)

        # - L1 decay
        l1 = split(top, "l1_weight_decay", [0.01,0.001,0.0001])
        l1 = split(l1, "n_epochs",["200,56"])
        l1 = split(l1, "l1_weight_decay_params", ["['W_rec']"])
        l1 = split(l1, "seed", seeds)

        # - Entropy SGD
        esgd = split(top, "optimizer", ["esgd"])
        esgd = split(esgd, "n_epochs",["200,56"])
        esgd = split(esgd, "learning_rate", ["0.1,0.01"])

        # - Ours
        ours = split(top, "beta_robustness", [1.0])
        ours = split(ours, "seed", seeds)

        return sgd + normal + dropout + l2 + l1 + esgd + ours

    @staticmethod
    def visualize():
        seeds = [0]
        grid = [model for model in regularization_comparison_experiment.train_grid() if model["seed"] in seeds]
        grid = run(grid, "train", run_mode="normal", store_key="*")("{*}")