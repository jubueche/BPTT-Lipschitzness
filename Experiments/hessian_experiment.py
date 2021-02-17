from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats
import matplotlib.pyplot as plt
from Hessian import density as density_lib

class hessian_experiment:
    
    @staticmethod
    def train_grid():
        seeds = [0,1,2,3,4,5,6,7,8,9]

        ecg = ecg_lsnn.make()
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "beta_robustness", [0.0, 1.0])
        ecg = split(ecg, "seed", seeds)

        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0, 1.0])
        speech = split(speech, "seed", seeds)

        speech_constant = speech_lsnn.make()
        speech_constant = split(speech_constant, "attack_size_constant", [0.01])
        speech_constant = split(speech_constant, "attack_size_mismatch", [0.0])
        speech_constant = split(speech_constant, "initial_std_constant", [0.001])
        speech_constant = split(speech_constant, "initial_std_mismatch", [0.0])
        speech_constant = split(speech_constant, "beta_robustness", [0.0, 1.0])
        speech_constant = split(speech_constant, "seed", seeds)

        cnn_grid = cnn.make()
        cnn_grid = split(cnn_grid, "attack_size_constant", [0.0])
        cnn_grid = split(cnn_grid, "attack_size_mismatch", [1.0])
        cnn_grid = split(cnn_grid, "initial_std_constant", [0.0])
        cnn_grid = split(cnn_grid, "initial_std_mismatch", [0.001])
        cnn_grid = split(cnn_grid, "beta_robustness", [0.0, 1.0])
        cnn_grid = split(cnn_grid, "seed", seeds)

        # return ecg + speech + cnn_grid
        return ecg + speech + speech_constant

    @staticmethod
    def visualize():
        seeds = [0,2]
        sigmas_ecg = [(3e3,1e2),(1e4,1e6)] 

        fig = plt.figure(figsize=(14, 5), constrained_layout=False)
        axes_top, axes_btm = get_axes_hessian(fig, ["Speech LSNN", "CNN", "ECG LSNN"])

        grid = [model for model in hessian_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = split(grid, "mm_level", [0.0])
        grid = configure(grid, {"n_iterations":1})
        grid = configure(grid, {"mode":"direct"})

        grid = run(grid, compute_hessian, n_threads=1, store_key="hessian")("{*}", "{data_dir}", 100)
        grid = run(grid, get_mismatch_list, n_threads=1, store_key="test_acc")("{n_iterations}", "{*}", "{mm_level}", "{data_dir}")

        def get_hess(architecture, constant=0.0):
            robust_hess = query(grid, "hessian", where={"beta_robustness":1.0, "architecture":architecture, "attack_size_constant":constant})
            vanilla_hess = query(grid, "hessian", where={"beta_robustness":0.0, "architecture":architecture})
            return vanilla_hess, robust_hess

        def get_test_acc(architecture, constant=0.0):
            robust_test_acc = query(grid, "test_acc", where={"beta_robustness":1.0, "architecture":architecture, "attack_size_constant":constant})
            vanilla_test_acc = query(grid, "test_acc", where={"beta_robustness":0.0, "architecture":architecture})
            return vanilla_test_acc, robust_test_acc

        def plot_arch(data,data_test_acc, i, sigma=1e-2, thresh=1e-10):
            d_norm = data[0]
            d_rob = data[1]
            ta_norm = data_test_acc[0]
            ta_rob = data_test_acc[1]
            for idx in range(len(d_rob)):
                normal = d_norm[idx]
                robust = d_rob[idx]
                normal_test_acc = ta_norm[idx]
                robust_test_acc = ta_rob[idx]
                if(i == 2):
                    sigma_norm = sigmas_ecg[idx][0]
                    sigma_rob = sigmas_ecg[idx][1]
                else:
                    sigma_norm = sigma
                    sigma_rob = sigma
                density_normal, grids_normal = density_lib.tridiag_to_density([normal[0]], grid_len=10000, sigma_squared=sigma_norm)
                density_rob, grids_rob = density_lib.tridiag_to_density([robust[0]], grid_len=10000, sigma_squared=sigma_rob)
                density_normal = list(map(lambda x : 0 if(x < thresh) else x,density_normal))
                density_rob = list(map(lambda x : 0 if(x < thresh) else x,density_rob))
                axes_top[i].semilogy(grids_normal, density_normal, alpha=0.3, label=("%.3f" % (normal_test_acc[0]-100/1146)))
                axes_btm[i].semilogy(grids_normal, density_rob, alpha=0.3, label=("%.3f" % (robust_test_acc[0]-100/1146)))

        data_speech_lsnn = get_hess("speech_lsnn", constant=0.0)
        data_speech_lsnn_test_acc = get_test_acc("speech_lsnn")
        plot_arch(data_speech_lsnn, data_speech_lsnn_test_acc, 0)
        data_ecg_lsnn = get_hess("ecg_lsnn", constant=0.0)
        data_ecg_lsnn_test_acc = get_test_acc("ecg_lsnn")
        plot_arch(data_ecg_lsnn, data_ecg_lsnn_test_acc, 2, sigma=2e3, thresh=1e-50)
        for a1,a2 in zip(axes_top,axes_btm):
            a1.legend(); a2.legend()

        plt.show()