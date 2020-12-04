from architectures import *
from run_utils import *
from experiment_utils import test_accuracy_after_mismatch_attack

class mismatch_experiment:
    
    @staticmethod
    def get_grids(mode):
        grids = {}
        
        ecg = ecg_lsnn.make(mode)
        ecg = split(ecg, "attack_size_constant", [0.0])
        ecg = split(ecg, "attack_size_mismatch", [2.0])
        ecg = split(ecg, "initial_std_constant", [0.0])
        ecg = split(ecg, "initial_std_mismatch", [0.001])
        ecg = split(ecg, "beta_robustness", [0.0, 1.0])
        ecg = split(ecg, "seed", [0,1,2,3,4,5,6,7,8,9])
        grids["ecg"] = ecg

        speech = speech_lsnn.make(mode)
        speech = split(speech, "attack_size_constant", [0.0])
        speech = split(speech, "attack_size_mismatch", [2.0])
        speech = split(speech, "initial_std_constant", [0.0])
        speech = split(speech, "initial_std_mismatch", [0.001])
        speech = split(speech, "beta_robustness", [0.0, 1.0])
        speech = split(speech, "seed", [0,1,2,3,4,5,6,7,8,9])
        grids["speech"] = speech

        cnn_grid = cnn.make(mode)
        cnn_grid = split(cnn_grid, "attack_size_constant", [0.0])
        cnn_grid = split(cnn_grid, "attack_size_mismatch", [1.0])
        cnn_grid = split(cnn_grid, "initial_std_constant", [0.0])
        cnn_grid = split(cnn_grid, "initial_std_mismatch", [0.001])
        cnn_grid = split(cnn_grid, "beta_robustness", [0.0, 1.0])
        cnn_grid = split(cnn_grid, "seed", [0,1,2,3,4,5,6,7,8,9])
        grids["cnn"] = cnn_grid

        return grids

    @staticmethod
    def visualize():
        def setup_grid(grid, mm_levels):
            mm0 = configure(grid,{"mm_level":0.0, "n_test_iterations":1})
            
            mm = split(grid, "mm_level", mm_levels)
            mm = configure(mm, {"n_test_iterations":1})
            
            combined = configure(mm0 + mm, {"test_accuracy_function": test_accuracy_after_mismatch_attack, "test_accuracy_dependencies":["mm_level", "{architecture}_session_id"]})
            return run("test_accuracy", combined)
        
        # grids = mismatch_experiment.get_grids()
        # ecg = run(load, grids["ecg"], "ecg_lsnn")
        # ecg = run(ecg_lsnn.loader, ecg)
        # ecg = setup_grid(ecg, [0.1,0.2,0.3,0.5,0.7])

        # speech = get("speech_lsnn",grids["speech_lsnn"],make_hyperparameters_speech(),checker=speech_lsnn_checker)
        # speech = map(speech_lsnn_loader, speech)
        # speech = setup_grid(speech,[0.5,0.7,0.9,1.1,1.5])

        # cnn = get("cnn", grids.cnn_mm_exp, make_hyperparameters_cnn(),checker=cnn_checker)
        # cnn = map(cnn_loader, cnn)
        # cnn = setup_grid(cnn, [0.5,0.7,0.9,1.1,1.5])

        # ecg[0][""]