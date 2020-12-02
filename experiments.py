import os
from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from RNN_Jax import RNN
import ujson as json
import jax.numpy as jnp
import numpy as onp
import jax.random as jax_random
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
import math

from run_utils import get, run, run_command, query, split, copy_key
from defaults import make_defaults_speech, make_defaults_ecg, make_defaults_cnn, make_hyperparameters_speech, make_hyperparameters_ecg, make_hyperparameters_cnn


def speech_lsnn_loader(model):
    session_id = model["speech_lsnn_session_id"]
    training_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Resources/TrainingResults/{session_id}.train"),'r'))
    
    for key in training_data:
        model[key] = training_data[key]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, f"Resources/Models/{session_id}_model.json")
    rnn, theta = RNN.load(model_path)
    model["rnn"] = rnn
    model["theta"] = theta
    model["audio_processor"] =  input_data.AudioProcessor(
        model["data_url"], model["data_dir"],
        model["silence_percentage"], model["unknown_percentage"],
        model["wanted_words"].split(','), model["validation_percentage"],
        model["testing_percentage"],
        model["n_thr_spikes"], model["in_repeat"], model["seed"]
    )
    
def speech_lsnn_checker(model):
    try:
        speech_lsnn_loader(model)
    except:
        return False
    epochs_list = list(map(int, model["n_epochs"].split(',')))
    n_steps = sum([math.ceil(epochs * model["audio_processor"].set_size("training")/model["batch_size"]) for epochs in epochs_list])
    return len(model["training_accuracy"]) >= int(n_steps/2)

def ecg_lsnn_loader(model):
    session_id = model["ecg_lsnn_session_id"]
    training_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Resources/TrainingResults/{session_id}.train"),'r'))
    
    for key in training_data:
        model[key] = training_data[key]
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, f"Resources/Models/{session_id}_model.json")
    rnn, theta = RNN.load(model_path)
    model["rnn"] = rnn
    model["theta"] = theta
    model["ecg_data_loader"] = ECGDataLoader(path=model["data_dir"], batch_size=model["batch_size"])

def ecg_lsnn_checker(model):
    try:
        ecg_lsnn_loader(model)
    except:
        return False
    return len(model["training_accuracy"]) > 10

def cnn_loader(model):
    session_id = model["cnn_session_id"]
    training_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Resources/TrainingResults/{session_id}.train"),'r'))
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, f"Resources/Models/{session_id}_model.json")
    cnn, theta = CNN.load(model_path)
    model["theta"] = theta
    model["cnn"] = cnn
    model["cnn_data_loader"] = CNNDataLoader(model["batch_size"],model["data_dir"])

def cnn_checker(model):
    try:
        cnn_loader(model)
    except:
        return False
    epochs_list = list(map(int, model["n_epochs"].split(',')))
    n_steps = sum([math.ceil(epochs * model["cnn_data_loader"].N_train/model["batch_size"]) for epochs in epochs_list])
    return len(model["training_accuracy"]) >= int(n_steps/2)

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

def test_accuracy_after_mismatch_attack(model):
    theta = model["theta"]
    attacked_theta = {}
    for key in theta:
        attacked_theta[key] = theta[key] * (1 + model["mm_std"] * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    model["attacked_theta"]=attacked_theta

    if model["architecture"] = "speech":
        set_size = model["audio_processor"].set_size('testing')
    
    if model["architecture"] = "ecg":
        pass
    
    total_accuracy = 0.0
    rnn = model["rnn"]

    audio_processor = model["audio_processor"]
    for i in range(0, set_size, model["batch_size"]):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(model["batch_size"], i, rnn.model_settings ,0.0, 0.0, 0.0, 'testing'))
        X = validation_fingerprints.numpy()
        y = validation_ground_truth.numpy()
        logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **attacked_theta)
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * model["batch_size"]) / set_size
    
    return onp.float64(total_accuracy)


def run_models(leonhard=False):
    #combine all grids in make_grids()
    pass

def mismatch_experiment():
    def setup_grid(grid, mm_levels):
        mm0 = split(grid, "mm_level", [0.0])
        mm0 = split(mm0, "n_test_iterations", [1])
        mm = split(grid, "mm_level", mm_levels)
        mm = split(mm, "n_test_iterations", [50])
        return map(test_accuracy_after_mismatch_attack, mm0 + mm)
    
    grids = make_grids()
    ecg = get("ecg_lsnn", grids.ecg_mm_exp, make_hyperparameters_ecg(),checker=ecg_lsnn_checker)
    ecg = map(ecg_lsnn_loader, ecg)
    ecg = setup_grid(ecg, [0.1,0.2,0.3,0.5,0.7])

    speech = get("speech_lsnn",grids.speech_mm_exp,make_hyperparameters_speech(),checker=speech_lsnn_checker)
    speech = map(speech_lsnn_loader, speech)
    speech = setup_grid(speech,[0.5,0.7,0.9,1.1,1.5])

    cnn = get("cnn", grids.cnn_mm_exp, make_hyperparameters_cnn(),checker=cnn_checker)
    cnn = map(cnn_loader, cnn)
    cnn = setup_grid(cnn, [0.5,0.7,0.9,1.1,1.5])

    ecg[0][""]



    

def make_grids():
    class Namespace:
        pass
    grids = Namespace()
    
    ecg = make_defaults_ecg()
    ecg = split(ecg, "attack_size_constant", [0.0])
    ecg = split(ecg, "attack_size_mismatch", [2.0])
    ecg = split(ecg, "initial_std_constant", [0.0])
    ecg = split(ecg, "initial_std_mismatch", [0.001])
    ecg = split(ecg, "beta_robustness", [0.0, 1.0])
    ecg = split(ecg, "seed", [0,1,2,3,4,5,6,7,8,9])
    ecg = split(ecg, "architecture", ["ecg"])
    ecg = split(ecg, "code_file", ["main_ecg_lsnn.py"])
    grids.ecg_mm_exp = ecg

    speech = make_defaults_speech()
    speech = split(speech, "attack_size_constant", [0.0])
    speech = split(speech, "attack_size_mismatch", [2.0])
    speech = split(speech, "initial_std_constant", [0.0])
    speech = split(speech, "initial_std_mismatch", [0.001])
    speech = split(speech, "beta_robustness", [0.0, 1.0])
    speech = split(speech, "seed", [0,1,2,3,4,5,6,7,8,9])
    speech = split(speech, "architecture", ["speech"])
    speech = split(speech, "code_file", ["main_speech_lsnn.py"])
    grids.speech_mm_exp = speech

    cnn = make_defaults_cnn()
    cnn = split(cnn, "attack_size_constant", [0.0])
    cnn = split(cnn, "attack_size_mismatch", [1.0])
    cnn = split(cnn, "initial_std_constant", [0.0])
    cnn = split(cnn, "initial_std_mismatch", [0.001])
    cnn = split(cnn, "beta_robustness", [0.0, 1.0])
    cnn = split(cnn, "seed", [0,1,2,3,4,5,6,7,8,9])
    cnn = split(cnn, "architecture", ["cnn"])
    cnn = split(cnn, "code_file", ["main_CNN_jax.py"])
    grids.cnn_mm_exp = cnn

    return grids