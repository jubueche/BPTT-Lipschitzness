from architectures import ecg_lsnn
from datajuicer import configure, run

class saturation_experiment:
    @staticmethod
    def train_grid():
        grid = [ecg_lsnn.make()]
        grid = configure(grid, {"eval_step_interval":25})
        return grid
    
    @staticmethod
    def visualize():
        grid = saturation_experiment.train_grid()
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")