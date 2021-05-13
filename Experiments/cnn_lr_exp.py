
from architectures import cnn
from datajuicer import configure

class cnn_lr_exp:
    
    @staticmethod
    def train_grid():
        seeds = [0]

        cnn_grid = [cnn.make()]
        cnn_grid0 = configure(cnn_grid, {"beta_robustness": 0.0})
        cnn_grid1 = configure(cnn_grid, {"beta_robustness": 0.25, "attack_size_mismatch": 0.1})
        cnn_grid2 = configure(cnn_grid, {"beta_robustness": 0.0, "dropout_prob":0.3})
        cnn_grid3 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer": "esgd", "n_epochs":"10,5"})
        cnn_grid4 = configure(cnn_grid, {"beta_robustness": 0.0, "optimizer":"abcd", "abcd_L":2, "n_epochs":"10,2", "abcd_etaA":0.001})
        cnn_grid5 = configure(cnn_grid, {"beta_robustness":0.0, "awp":True, "awp_gamma":0.1, "boundary_loss":"madry"})
        return configure(cnn_grid0 + cnn_grid1 + cnn_grid2 + cnn_grid3 + cnn_grid4 + cnn_grid5,{"learning_rate":"0.0001,0.00001"})

