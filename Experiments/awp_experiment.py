from architectures import ecg_lsnn
from datajuicer import split


class awp_experiment:
    @staticmethod
    def train_grid(): 
        grid = ecg_lsnn.make()
        grid["awp"]=True
        grid["boundary_loss"]="madry"
        grid = split(grid, "awp_gamma", [0.01, 0.1, 0.25, 0.5, 1.0])
        return grid