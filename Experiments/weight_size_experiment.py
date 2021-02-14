from architectures import speech_lsnn
from datajuicer import run, split, configure, query
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class weight_size_experiment:
    
    @staticmethod
    def train_grid():
        
        constant = configure(speech_lsnn.make(), {"attack_size_mismatch":0.0})

        constant = split(constant, "attack_size_constant", [0.01, 0.05, 0.1])
        
        mismatch = configure(speech_lsnn.make(), {"attack_size_constant":0.0})

        mismatch = split(mismatch, "attack_size_mismatch", [0.2, 1.0, 2.0])

        return constant + mismatch

    @staticmethod
    def visualize():
        #make a histogram
        pass