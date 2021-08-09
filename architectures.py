from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from RNN_Jax import RNN
from MLP_Jax import MLP
import ujson as json
import jax.numpy as jnp
import numpy as onp
import math
import os
import os.path
from datajuicer import cachable, get, format_template
import argparse
import random
import re

def standard_defaults():
    return {
        "dropout_prob":0.0,
        "l2_weight_decay":0.0,
        "l2_weight_decay_params":"[]",
        "l1_weight_decay":0.0,
        "l1_weight_decay_params":"[]",
        "contractive":0.0,
        "contractive_params":"[]",
        "reg":0.001,
        "learning_rate":"0.001,0.0001",
        "n_attack_steps": 10,
        "beta_robustness": 0.125, #TODO
        "seed":0,
        "n_hidden":256,
        "n_layer":1,
        "batch_size":100,
        "boundary_loss":"kl",
        "treat_as_constant":False,
        "abcd_etaA":0.1,
        "abcd_L":2,
        "p_norm":"inf",
        "hessian_robustness":False,
        "awp":False,
        "noisy_forward_std":0.0,
        "warmup":0,
        "awp_gamma":0.1
        }

def help():
    return {
        "batch_size": "The batch size of the model."
        }

launch_settings = {
    "direct":"mkdir -p Resources/Logs; python {code_file} {args} 2>&1 | tee Resources/Logs/{session_id}.log",
    "bsub":"mkdir -p Resources/Logs; bsub -o Resources/Logs/{session_id}.log -W 24:00 -n 16 -R \"rusage[mem=4096]\" \"python3 {code_file} {args}\""
}


def _dict_to_bash(dic):
    def _format(key, value):
        if type(value) is bool:
            if value==True:
                return f"-{key}"
            else:
                return ""
        else: return f"-{key}={value}"
    
    return  " ".join([_format(key, val) for key, val in dic.items()])


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
        def _format(key, value):
            if type(value) is bool:
                if value==True:
                    return f"-{key}"
                else:
                    return ""
            else: return f"-{key}={value}"

        model["args"] = _dict_to_bash({key:get(model,key) for key in list(architecture.default_hyperparameters().keys())+env_vars + ["session_id"]})
        
        command = format_template(model,launch_settings[mode])
        print(command)
        os.system(command)
        return None

    return runner

def _get_flags(default_dict, help_dict, arg_dict=None):
    parser = argparse.ArgumentParser()
    for key, value in default_dict.items():
        if type(value) is bool:
            parser.add_argument("-"+key, action="store_true",help=help_dict.get(key,""))
        else:
            parser.add_argument("-" + key,type=type(value),default=value,help=help_dict.get(key,""))
    parser.add_argument("-session_id", type=int, default = 0)
    if arg_dict is None:
        flags = parser.parse_args()
    else:
        dic = {key: get(arg_dict,key) for key in arg_dict if key in default_dict or key=="session_id"}
        string = _dict_to_bash(dic)
        flags = parser.parse_args(str.split(string))
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
        d = json.loads(data)
    else:
        d = {}
    with open(file,'w+') as f:
        if key in d:
            d[key] += [value]
        else:
            d[key]=[value]
        out = re.sub('(?<!")Inf(?!")','"Inf"', json.dumps(d))
        out = re.sub('(?<!")NaN(?!")','"NaN"', out)
        f.write(out)

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
        d["eval_step_interval"]=400
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
        d["wanted_words"] ="yes,no,up,down,left,right"
        d["attack_size_constant"]=0.0
        d["initial_std_constant"]=0.0
        d["attack_size_mismatch"] = 0.2
        d["initial_std_mismatch"]=0.001
        d["n_epochs"] = "80,20" # ~ 18h
        d["optimizer"]="adam"
        return d
    
    @staticmethod
    def get_flags(dic=None):
        default_dict = {**speech_lsnn.default_hyperparameters(), **{"data_dir":"TensorCommands/speech_dataset/"}}
        FLAGS = _get_flags(default_dict, help(), dic)
        def _next_power_of_two(x):
            return 1 if x == 0 else 2**(int(x) - 1).bit_length()

        FLAGS.l2_weight_decay_params = str(FLAGS.l2_weight_decay_params[1:-1]).split(",")
        FLAGS.l1_weight_decay_params = str(FLAGS.l1_weight_decay_params[1:-1]).split(",")
        FLAGS.contractive_params = str(FLAGS.contractive_params[1:-1]).split(",")
        if(FLAGS.l1_weight_decay_params == ['']):
            FLAGS.l1_weight_decay_params = []
        if(FLAGS.l2_weight_decay_params == ['']):
            FLAGS.l2_weight_decay_params = []
        if(FLAGS.contractive_params == ['']):
            FLAGS.contractive_params = []

        FLAGS.desired_samples = int(FLAGS.sample_rate * FLAGS.clip_duration_ms / 1000)
        FLAGS.window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
        FLAGS.window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
        FLAGS.length_minus_window = (FLAGS.desired_samples - FLAGS.window_size_samples)
        if FLAGS.length_minus_window < 0:
            spectrogram_length = 0
        else:
            FLAGS.spectrogram_length = 1 + int(FLAGS.length_minus_window / FLAGS.window_stride_samples)
        if FLAGS.preprocess == 'average':
            fft_bin_count = 1 + (_next_power_of_two(FLAGS.window_size_samples) / 2)
            FLAGS.average_window_width = int(math.floor(fft_bin_count / FLAGS.feature_bin_count))
            FLAGS.fingerprint_width = int(math.ceil(fft_bin_count / FLAGS.average_window_width))
        elif FLAGS.preprocess in ['mfcc', 'fbank']:
            FLAGS.average_window_width = -1
            FLAGS.fingerprint_width = FLAGS.feature_bin_count
        elif FLAGS.preprocess == 'micro':
            FLAGS.average_window_width = -1
            FLAGS.fingerprint_width = FLAGS.feature_bin_count
        else:
            raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                            ' "average", or "micro")' % (FLAGS.preprocess))
        FLAGS.fingerprint_size = FLAGS.fingerprint_width * FLAGS.spectrogram_length
        FLAGS.architecture = "speech_lsnn"
        return FLAGS

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
        return data


class mnist_mlp:
    @staticmethod
    def make():
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "MNIST/mnist_dataset/"
            elif mode=="bsub":
                return "$SCRATCH/mnist_dataset"
            raise Exception("Invalid Mode")
        d = mnist_mlp.default_hyperparameters()
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
        d["code_file"] = "main_mnist_mlp.py"
        d["architecture"] = "mnist_mlp"
        d["train"] = mk_runner(mnist_mlp, ["data_dir"])
        return d
        

    @staticmethod
    def default_hyperparameters():
        d = {}
        d["step_size"]=0.001
        d["batch_size"]=128
        d["weight_increase"]=0.0
        d["n_epochs"] = "20"
        d["n_iters"]=10
        d["eps_attack"]=0.2
        return d
    
    @staticmethod
    def get_flags(dic=None):
        default_dict = {**mnist_mlp.default_hyperparameters(), **{"data_dir":"MNIST/mnist_dataset/"}}
        return _get_flags(default_dict, help(), dic)

    @staticmethod
    def checker(sid, table, cache_dir):
        try:
            data = mnist_mlp.loader(sid, table, cache_dir)
        except Exception as er:
            print(er)
            return False
        return True

    @staticmethod
    def loader(sid, table, cache_dir):
        data = json.load(open(os.path.join("Resources/TrainingResults",f"{sid}.json"),'r'))
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, f"Resources/Models/{sid}_model.json")
        mlp, params = MLP.load(model_path)
        data["network"] = mlp
        data["theta"] = params
        data["mnist_mlp_session_id"] = sid
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
        d["attack_size_mismatch"]=0.2
        d["initial_std_mismatch"]=0.001
        d["clip_duration_ms"]=1000
        d["window_size_ms"]=30.0
        d["window_stride_ms"]=10.0 
        d["preprocess"]="mfcc"
        d["feature_bin_count"]=40 
        d["in_repeat"]=1 
        d["n_thr_spikes"]=-1
        d["n_epochs"] = "150,50" # ~ 17 h
        d["optimizer"]="adam"
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
    def get_flags(dic=None):
        default_dict = {**ecg_lsnn.default_hyperparameters(), **{"data_dir":"ECG/ecg_recordings/"}}
        FLAGS = _get_flags(default_dict, help(), dic)

        def _next_power_of_two(x):
            return 1 if x == 0 else 2**(int(x) - 1).bit_length()

        FLAGS.l2_weight_decay_params = str(FLAGS.l2_weight_decay_params[1:-1]).split(",")
        FLAGS.l1_weight_decay_params = str(FLAGS.l1_weight_decay_params[1:-1]).split(",")
        FLAGS.contractive_params = str(FLAGS.contractive_params[1:-1]).split(",")
        if(FLAGS.l1_weight_decay_params == ['']):
            FLAGS.l1_weight_decay_params = []
        if(FLAGS.l2_weight_decay_params == ['']):
            FLAGS.l2_weight_decay_params = []
        if(FLAGS.contractive_params == ['']):
            FLAGS.contractive_params = []

        FLAGS.desired_samples = int(FLAGS.sample_rate * FLAGS.clip_duration_ms / 1000)
        FLAGS.window_size_samples = int(FLAGS.sample_rate * FLAGS.window_size_ms / 1000)
        FLAGS.window_stride_samples = int(FLAGS.sample_rate * FLAGS.window_stride_ms / 1000)
        FLAGS.length_minus_window = (FLAGS.desired_samples - FLAGS.window_size_samples)
        if FLAGS.length_minus_window < 0:
            spectrogram_length = 0
        else:
            FLAGS.spectrogram_length = 1 + int(FLAGS.length_minus_window / FLAGS.window_stride_samples)
        if FLAGS.preprocess == 'average':
            fft_bin_count = 1 + (_next_power_of_two(FLAGS.window_size_samples) / 2)
            FLAGS.average_window_width = int(math.floor(fft_bin_count / FLAGS.feature_bin_count))
            FLAGS.fingerprint_width = int(math.ceil(fft_bin_count / FLAGS.average_window_width))
        elif FLAGS.preprocess in ['mfcc', 'fbank']:
            FLAGS.average_window_width = -1
            FLAGS.fingerprint_width = FLAGS.feature_bin_count
        elif FLAGS.preprocess == 'micro':
            FLAGS.average_window_width = -1
            FLAGS.fingerprint_width = FLAGS.feature_bin_count
        else:
            raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                            ' "average", or "micro")' % (FLAGS.preprocess))
        FLAGS.fingerprint_size = FLAGS.fingerprint_width * FLAGS.spectrogram_length

        FLAGS.architecture = "ecg_lsnn"

        return FLAGS

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
        return data

class cnn:
    @staticmethod 
    def default_hyperparameters():
        d = standard_defaults()
        d["eval_step_interval"]=1000
        d["attack_size_constant"]=0.0
        d["attack_size_mismatch"]=0.2
        d["initial_std_constant"]=0.0
        d["initial_std_mismatch"]=0.001
        d["n_epochs"] = "35,5" # ~ 18 h
        d["Kernels"]="[[64,1,4,4],[64,64,4,4]]"
        d["Dense"]="[[1600,256],[256,64],[64,10]]"
        d["optimizer"]="adam"
        d["learning_rate"] = "0.0001,0.00001"
        d["dataset"] = "fashion"
        return d

    @staticmethod
    def make_cifar():
        d = cnn.make()
        d["dataset"] = "cifar"
        d["Kernels"]="[[64,3,4,4],[64,64,4,4]]"
        d["Dense"]="[[2304,256],[256,64],[64,10]]"
        def mk_data_dir(mode="direct"):
            if mode=="direct":
                return "CIFAR10/"
            elif mode=="bsub":
                return "$SCRATCH/CIFAR10"
            raise Exception("Invalid Mode")
        d["mk_data_dir"] = mk_data_dir
        d["data_dir"] = "{mk_data_dir({mode})}"
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
    def get_flags(dic=None):
        default_dict = {**cnn.default_hyperparameters(), **{"data_dir":"CNN/fashion_mnist/"}}
        FLAGS = _get_flags(default_dict, help(), dic)
        FLAGS.Kernels = json.loads(FLAGS.Kernels)
        FLAGS.Dense = json.loads(FLAGS.Dense)

        FLAGS.l2_weight_decay_params = str(FLAGS.l2_weight_decay_params[1:-1]).split(",")
        FLAGS.l1_weight_decay_params = str(FLAGS.l1_weight_decay_params[1:-1]).split(",")
        FLAGS.contractive_params = str(FLAGS.contractive_params[1:-1]).split(",")
        if(FLAGS.l1_weight_decay_params == ['']):
            FLAGS.l1_weight_decay_params = []
        if(FLAGS.l2_weight_decay_params == ['']):
            FLAGS.l2_weight_decay_params = []
        if(FLAGS.contractive_params == ['']):
            FLAGS.contractive_params = []
        
        FLAGS.architecture = "cnn"
        return FLAGS

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
        return data
