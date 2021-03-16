from architectures import ecg_lsnn
from datajuicer import configure

class saturation_experiment:
    def train():
        grid = [ecg_lsnn.make()]
        grid = configure(grid, {"eval_step_interval":25})
        return grid