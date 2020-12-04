from run_utils import Architecture
from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from RNN_Jax import RNN
import ujson as json
import jax.numpy as jnp
import numpy as onp
import math
import os

def standard_defaults():
    return {
        "clip_duration_ms":1000,
        "window_size_ms":30.0,
        "window_stride_ms":10.0, 
        "feature_bin_count":40 ,
        "sample_rate":16000 ,
        "in_repeat":1 ,
        "preprocess":"mfcc",
        "data_url":"https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "silence_percentage":10 ,
        "unknown_percentage":10 ,
        "validation_percentage":10 ,
        "testing_percentage":10 ,
        "n_thr_spikes":-1 ,
        "background_volume":0.1 ,
        "background_frequency":0.8 ,
        "time_shift_ms":100.0,
        "dt":1.,
        "tau":20. ,
        "beta":2. ,
        "tau_adaptation":98. ,
        "thr":0.01 ,
        "thr_min":0.005 ,
        "refr":2 ,
        "dampening_factor":0.3 ,
        "dropout_prob":0.0,
        "reg":0.001,

        "learning_rate":"0.001,0.0001",
        "wanted_words":"yes,no",
        "n_epochs":"32,8",
        "n_attack_steps": 10,
        "beta_robustness": 1.0,
        "seed":0,
        "n_hidden":256,
        "n_layer":1,
        "batch_size":100
        }

def standard_help():
    return {
        "batch_size": "The batch size of the model."
        }

def standard_launch_settings(mode):
    if mode == "direct":
        return {
            "launch":"python {code_file} {make_args}"
        }
    elif mode == "bsub":
        return {
            "launch": "bsub -o ../logs/{{architecture}_session_id} -W 24:00 -n 16 -R \"rusage[mem=4096]\" \"python3 {code_file} {make_args}\""
        }
    raise Exception("Invalid Mode")

class speech_lsnn(Architecture):

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        d["eval_step_interval"]=200
        d["attack_size_constant"]=0.01
        d["attack_size_mismatch"]=0.0
        d["initial_std_constant"]=0.0001
        d["initial_std_mismatch"]=0.0
        return d
    

    @staticmethod
    def environment_parameters(mode="direct"):
        if mode=="direct":
            return {
                "data_dir": "TensorCommands/speech_dataset/"
            }
        elif mode=="bsub":
            return {
                "data_dir": "$SCRATCH/speech_dataset"
            }
        raise Exception("Invalid Mode")

    @staticmethod
    def launch_settings(mode):
        d = standard_launch_settings(mode)
        d["code_file"] = "main_speech_lsnn.py"
        return d

    @staticmethod
    def help():
        return standard_help()

    @staticmethod
    def checker(model):
        try:
            speech_lsnn.loader(model)
        except:
            return False
        epochs_list = list(map(int, model["n_epochs"].split(',')))
        n_steps = sum([math.ceil(epochs * model["audio_processor"].set_size("training")/model["batch_size"]) for epochs in epochs_list])
        return len(model["training_accuracy"]) >= int(n_steps/2)

    @staticmethod
    def loader(model):
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
        
    

class ecg_lsnn(Architecture):
    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        d["eval_step_interval"]=200
        d["attack_size_constant"]=0.0
        d["attack_size_mismatch"]=2.0
        d["initial_std_constant"]=0.0
        d["initial_std_mismatch"]=0.0001
        d["initial_std_mismatch"]=0.0
        return d
    

    @staticmethod
    def environment_parameters(mode="direct"):
        if mode=="direct":
            return {
                "data_dir": "ECG/ecg_recordings/"
            }
        elif mode=="bsub":
            return {
                "data_dir": "$SCRATCH/ecg_recordings/"
            }
        print(mode)
        raise Exception("Invalid Mode")

    @staticmethod
    def launch_settings(mode):
        d = standard_launch_settings(mode)
        d["code_file"] = "main_ecg_lsnn.py"
        return d

    @staticmethod
    def help():
        return standard_help()


    @staticmethod
    def checker(model):
        try:
            ecg_lsnn.loader(model)
        except:
            return False
        return len(model["training_accuracy"]) > 10

    
    @staticmethod
    def loader(model):
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

class cnn(Architecture):
    @staticmethod 
    def default_hyperparameters():
        d = standard_defaults()
        d["eval_step_interval"]=1000
        d["attack_size_constant"]=0.0
        d["attack_size_mismatch"]=1.0
        d["initial_std_constant"]=0.0
        d["initial_std_mismatch"]=0.0001
        d["Kernels"]="[[64,1,4,4],[64,64,4,4]]"
        d["Dense"]="[[1600,256],[256,64],[6410]]"
        d["initial_std_mismatch"]=0.0
        return d

    @staticmethod
    def environment_parameters(mode="direct"):
        if mode=="direct":
            return {
                "data_dir": "CNN/fashion_mnist/"
            }
        elif mode=="bsub":
            return {
                "data_dir": "$SCRATCH/fashion_mnist"
            }
        raise Exception("Invalid Mode")
    
    @staticmethod
    def launch_settings(mode):
        d = standard_launch_settings(mode)
        d['code_file'] = "main_CNN_jax.py"
        return d
    
    @staticmethod
    def help():
        return standard_help()

    @staticmethod
    def checker(model):
        try:
            cnn.loader(model)
        except:
            return False
        epochs_list = list(map(int, model["n_epochs"].split(',')))
        n_steps = sum([math.ceil(epochs * model["cnn_data_loader"].N_train/model["batch_size"]) for epochs in epochs_list])
        return len(model["training_accuracy"]) >= int(n_steps/2)

    @staticmethod
    def loader(model):
        session_id = model["cnn_session_id"]
        training_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Resources/TrainingResults/{session_id}.train"),'r'))
        for key in training_data:
            model[key] = training_data[key]
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{session_id}_model.json")
        cnn, theta = CNN.load(model_path)
        model["theta"] = theta
        model["cnn"] = cnn
        model["cnn_data_loader"] = CNNDataLoader(model["batch_size"],model["data_dir"])

