from architectures import ecg_lsnn, speech_lsnn, cnn
from datajuicer import run, split, configure, query, run
from experiment_utils import *
from matplotlib.lines import Line2D
from scipy import stats

class quantization_experiment:
    
    @staticmethod
    def train_grid():
        seed = [0,1,2,3,4,5,6,7,8,9]
        bits = [1,2,3,4,5,6,7,-1]
        ecg = ecg_lsnn.make()
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "beta_robustness", [0.0, 1.0])
        ecg = split(ecg, "seed", seed)
        ecg = split(ecg, "bits", bits)

        speech = speech_lsnn.make()
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0, 1.0])
        speech = split(speech, "seed", seed)
        speech = split(speech, "bits", bits)

        cnn_grid = cnn.make()
        cnn_grid = split(cnn_grid, "attack_size_constant", [0.0])
        cnn_grid = split(cnn_grid, "attack_size_mismatch", [1.0])
        cnn_grid = split(cnn_grid, "initial_std_constant", [0.0])
        cnn_grid = split(cnn_grid, "initial_std_mismatch", [0.001])
        cnn_grid = split(cnn_grid, "beta_robustness", [0.0, 1.0])
        cnn_grid = split(cnn_grid, "seed", seed)
        cnn_grid = split(cnn_grid, "bits", bits)

        # return ecg + speech + cnn_grid
        return speech

    @staticmethod
    def visualize():
        seeds = [0,1,2,3,4,5,6,7,8,9]
        bits = [1,2,3,4,5,6,7,-1]
        betas = [1.0,1.0,1.0]
        bits_labels = ["1","2","3","4","5","6","Full"]
        architectures = ["speech_lsnn", "cnn", "ecg_lsnn"]
        
        grid = [model for model in quantization_experiment.train_grid() if model["seed"] in seeds] 
        grid = run(grid, "train", run_mode="load", store_key="*")("{*}")
        grid = configure(grid, {"mode":"direct"})
        
        grid = run(grid, get_quantized_acc, n_threads=1, store_key="test_acc")("{bits}", "{*}", "{data_dir}")

        print("\t\t Speech LSNN \t\t\t\t CNN \t\t\t\t\t ECG LSNN")
        for i,b in enumerate(bits):
            print(f"{bits_labels[i]}", end=" ")
            for j,a in enumerate(architectures):
                grids_beta_0 = query(grid, "test_acc", where={"bits":b, "beta_robustness":0, "architecture":a})
                grids_beta_1 = query(grid, "test_acc", where={"bits":b, "beta_robustness":betas[j], "architecture":a})
                m0 = onp.mean(grids_beta_0)*100
                m1 = onp.mean(grids_beta_1)*100
                s0 = onp.std(grids_beta_0)*100
                s1 = onp.std(grids_beta_1)*100
                print("\t\t %.2f$\pm$%.2f/%.2f$\pm$%.2f" % (m0,s0,m1,s1), end=" ")
            print("\n")
