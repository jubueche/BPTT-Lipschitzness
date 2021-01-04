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
import os.path
from datajuicer import cachable, get, format_template
import argparse
import random

def standard_defaults():
    return {
        "dropout_prob":0.0,
        "reg":0.001,
        "learning_rate":"0.001,0.0001",
        "n_epochs":"32,8",
        "n_attack_steps": 10,
        "beta_robustness": 1.0,
        "seed":0,
        "n_hidden":256,
        "n_layer":1,
        "batch_size":100
        }

def help():
    return {
        "batch_size": "The batch size of the model."
        }

launch_settings = {
    "direct":"python {code_file} {args}",
    "bsub":"bsub -o ../logs/{session_id} -W 24:00 -n 16 -R \"rusage[mem=4096]\" \"python3 {code_file} {args}\""
}

def mk_runner(architecture, env_vars):
    @cachable(
        dependencies=["model:"+key for key in architecture.default_hyperparameters().keys()], 
        saver = None,
        loader = architecture.loader,
        checker=architecture.checker,
        table_name=architecture.__name__
    )
    def runner(model):
        try:
            mode = get(model, "mode")
        except:
            mode = "direct"
        model["mode"] = mode
        model["args"] = " ".join([f"-{key}={get(model, key)}" for key in list(architecture.default_hyperparameters().keys())+env_vars + ["session_id"]])
        command = format_template(model,launch_settings[mode])
        os.system(command)
        return None

    return runner

def _get_flags(default_dict, help_dict):
    parser = argparse.ArgumentParser()
    for key, value in default_dict.items():
        parser.add_argument("-" + key,type=type(value),default=value,help=help_dict.get(key,""))
    parser.add_argument("-session_id", type=int, default = 0)
    
    flags = parser.parse_args()
    if flags.session_id==0:
        flags.session_id = random.randint(1000000000, 9999999999)
    return flags

def log(session_id, key, value, save_dir = None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/TrainingResults/")
    file = os.path.join(save_dir, str(session_id) + ".json")
    exists = os.path.isfile(file)
    directory = os.path.dirname(file)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if exists:
        data = open(file).read()
        try:
            d = json.loads(data)
        except:
            d = {}
    else:
        d = {}
    with open(file,'w+') as f:
        if key in d:
            d[key] += [value]
        else:
            d[key]=[value]
        json.dump(d,f)

class speech_lsnn:
    @staticmethod
    def make():
        d = speech_lsnn.default_hyperparameters()
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "TensorCommands/speech_dataset/"
            elif mode=="bsub":
                return "$SCRATCH/speech_dataset"
            raise Exception("Invalid Mode")
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_speech_lsnn.py"
        d["architecture"] = "speech_lsnn"
        d["train"] = mk_runner(speech_lsnn, ["data_dir"])
        return d
        

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        d["dt"]=1.
        d["tau"]=20. 
        d["beta"]=2. 
        d["tau_adaptation"]=98.
        d["thr"]=0.01
        d["thr_min"]=0.005
        d["refr"]=2
        d["dampening_factor"]=0.3
        d["sample_rate"]=16000 
        d["eval_step_interval"]=200
        d["clip_duration_ms"]=1000
        d["window_size_ms"]=30.0
        d["window_stride_ms"]=10.0 
        d["feature_bin_count"]=40 
        d["sample_rate"]=16000 
        d["in_repeat"]=1 
        d["preprocess"]="mfcc"
        d["data_url"]="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        d["silence_percentage"]=10.0 
        d["unknown_percentage"]=10.0 
        d["validation_percentage"]=10.0 
        d["testing_percentage"]=10.0 
        d["n_thr_spikes"]=-1 
        d["background_volume"]=0.1 
        d["background_frequency"]=0.8 
        d["time_shift_ms"]=100.0
        d["wanted_words"] = "yes,no"
        d["attack_size_constant"]=0.0
        d["initial_std_constant"]=0.0
        d["attack_size_mismatch"] = 2.0
        d["initial_std_mismatch"]=0.001
        d["n_epochs"] = "64,16"
        d["optimizer"] = "adam"
        return d
    
    @staticmethod
    def get_flags():
        default_dict = {**speech_lsnn.default_hyperparameters(), **{"data_dir":"TensorCommands/speech_dataset/"}}
        return _get_flags(default_dict, help())

    @staticmethod
    def checker(sid, table, cache_dir):
        try:
            data = speech_lsnn.loader(sid, table, cache_dir)
        except Exception as er:
            print(er)
            return False
        if "training_accuracies" in data:
            ta = data["training_accuracies"]
        elif "training_accuracy" in data:
            ta = data["training_accuracy"]
        else:
            return False
            
        return True # len(ta) >= 50

    @staticmethod
    def loader(sid, table, cache_dir):
        data = json.load(open(os.path.join("Resources/TrainingResults",f"{sid}.json"),'r'))
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.json")
        rnn, theta = RNN.load(model_path)
        data["network"] = rnn
        data["theta"] = theta
        data["speech_lsnn_session_id"] = sid
        # data["audio_processor"] =  input_data.AudioProcessor(
        #     data["data_url"], model["data_dir"],
        #     data["silence_percentage"], data["unknown_percentage"],
        #     data["wanted_words"].split(','), data["validation_percentage"],
        #     data["testing_percentage"],
        #     data["n_thr_spikes"], data["in_repeat"], data["seed"]
        # )
        return data

class ecg_lsnn:

    @staticmethod
    def default_hyperparameters():
        d = standard_defaults()
        d["dt"]=1.
        d["tau"]=20. 
        d["beta"]=2. 
        d["tau_adaptation"]=98.
        d["thr"]=0.01
        d["thr_min"]=0.005
        d["refr"]=2
        d["dampening_factor"]=0.3
        d["sample_rate"]=16000 
        d["eval_step_interval"]=400
        d["attack_size_constant"]=0.0
        d["initial_std_constant"]=0.0
        d["attack_size_mismatch"]=2.0
        d["initial_std_mismatch"]=0.001
        d["clip_duration_ms"]=1000
        d["window_size_ms"]=30.0
        d["window_stride_ms"]=10.0 
        d["preprocess"]="mfcc"
        d["feature_bin_count"]=40 
        d["in_repeat"]=1 
        d["n_thr_spikes"]=-1 
        return d
    
    @staticmethod
    def make():
        d = ecg_lsnn.default_hyperparameters()
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "ECG/ecg_recordings/"
            elif mode=="bsub":
                return "$SCRATCH/ecg_recordings/"
            raise Exception("Invalid Mode")
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_ecg_lsnn.py"
        d["architecture"] = "ecg_lsnn"
        d["train"] = mk_runner(ecg_lsnn, ["data_dir"])
        return d

    @staticmethod
    def get_flags():
        default_dict = {**ecg_lsnn.default_hyperparameters(), **{"data_dir":"ECG/ecg_recordings/"}}
        return _get_flags(default_dict, help())

    @staticmethod
    def checker(sid, table, cache_dir):
        try:
            data = ecg_lsnn.loader(sid, table, cache_dir)
        except Exception as er:
            print("Checker error",er)
            return False
        if "training_accuracies" in data:
            ta = data["training_accuracies"]
        elif "training_accuracy" in data:
            ta = data["training_accuracy"]
        else:
            return False

        return len(ta) > 50

    
    @staticmethod
    def loader(sid, table, cache_dir):
        data = json.load(open(os.path.join("Resources/TrainingResults",f"{sid}.json"),'r'))
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.json")
        rnn, theta = RNN.load(model_path)
        data["network"] = rnn
        data["theta"] = theta
        data["ecg_lsnn_session_id"] = sid
        #data["ecg_data_loader"] = ECGDataLoader(path=model["data_dir"], batch_size=data["batch_size"])
        return data

class cnn:
    @staticmethod 
    def default_hyperparameters():
        d = standard_defaults()
        d["eval_step_interval"]=1000
        d["attack_size_constant"]=0.0
        d["attack_size_mismatch"]=1.0
        d["initial_std_constant"]=0.0
        d["initial_std_mismatch"]=0.001
        d["Kernels"]="[[64,1,4,4],[64,64,4,4]]"
        d["Dense"]="[[1600,256],[256,64],[64,10]]"
        return d

    @staticmethod
    def make():
        d = cnn.default_hyperparameters()
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "CNN/fashion_mnist/"
            elif mode=="bsub":
                return "$SCRATCH/fashion_mnist"
            raise Exception("Invalid Mode")
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_CNN_jax.py"
        d["architecture"] = "cnn"
        d["train"] = mk_runner(cnn, ["data_dir"])
        return d

    @staticmethod
    def get_flags():
        default_dict = {**cnn.default_hyperparameters(), **{"data_dir":"CNN/fashion_mnist/"}}
        return _get_flags(default_dict, help())

    @staticmethod
    def checker(sid, table, cache_dir):
        try:
            data = cnn.loader(sid, table, cache_dir)
        except Exception as er:
            print("error", er)
            return False
        
        if "training_accuracies" in data:
            ta = data["training_accuracies"]
        elif "training_accuracy" in data:
            ta = data["training_accuracy"]
        else:
            return False
            
        return len(ta) >= 50

    @staticmethod
    def loader(sid, table, cache_dir):
        data = json.load(open(os.path.join("Resources/TrainingResults",f"{sid}.json"),'r'))
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.json")
        cnnmodel, theta = CNN.load(model_path)
        data["theta"] = theta
        data["network"] = cnnmodel
        data["cnn_session_id"] = sid
        #data["cnn_data_loader"] = CNNDataLoader(data["batch_size"],model["data_dir"])
        return data

