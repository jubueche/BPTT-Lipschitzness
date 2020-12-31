from architectures import speech_lsnn, ecg_lsnn, cnn
from datajuicer import *


class example_experiment:
    @staticmethod
    def train_grid():
        grid = cnn.make()
        grid = split(grid, "seed", [0,1,2,3])
        return grid
        
    @staticmethod
    def visualize():
        grid = example_experiment.train_grid()
        grid = run(grid, func="train", run_mode="load", store_key="*")("{*}")


        print("end")
        


