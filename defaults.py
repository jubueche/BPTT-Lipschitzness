import argparse


###only return hyperparameters in make_defaults functions. Any environment parameters like data_dir should only appear in get_flags
def make_defaults():
    d = {}
    d["clip_duration_ms"]=1000 
    d["window_size_ms"]=30.0 
    d["window_stride_ms"]=10.0 
    d["feature_bin_count"]=40 
    d["sample_rate"]=16000 
    d["in_repeat"]=1 
    d["preprocess"]="mfcc"
    d["data_url"]="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    d["silence_percentage"]=10 
    d["unknown_percentage"]=10 
    d["validation_percentage"]=10 
    d["testing_percentage"]=10 
    d["n_thr_spikes"]=-1 
    d["background_volume"]=0.1 
    d["background_frequency"]=0.8 
    d["time_shift_ms"]=100.0
    d["dt"]=1.
    d["tau"]=20. 
    d["beta"]=2. 
    d["tau_adaptation"]=98. 
    d["thr"]=0.01 
    d["thr_min"]=0.005 
    d["refr"]=2 
    d["dampening_factor"]=0.3 
    d["dropout_prob"]=0.0
    d["reg"]=0.001

    d["learning_rate"]="0.0010.0001"
    d["wanted_words"]="yesno"
    d["n_epochs"]="328"
    d["n_attack_steps"]= 10
    d["beta_robustness"]= 1.0
    d["seed"]=0
    d["n_hidden"]=256
    d["n_layer"]=1
    d["batch_size"]=100
    return d

def make_help():
    d= {}
    d["batch_size"] = "The batch size of the model."
    return d

def make_defaults_lsnn():
    d = make_defaults()
    d["eval_step_interval"]=200
    d["attack_size_constant"]=0.01
    d["attack_size_mismatch"]=0.0
    d["initial_std_constant"]=0.0001
    d["initial_std_mismatch"]=0.0
    return d

def make_hyperparameters_lsnn():
    return make_defaults_lsnn().keys()

def get_flags_lsnn():
    help=make_help()
    for key, value in make_defaults_lsnn().items():
        parser = argparse.ArgumentParser()
        parser.add_argument(key,type=type(value),default=value,help=help.get(key,default=""))
    parser.add_argument("data_dir",type=str,default='TensorCommands/speech_dataset/',help='Directory of speech signals')
    return parser.parse_args()

def make_defaults_ecg():
    d = make_defaults()
    d["eval_step_interval"]=200
    d["attack_size_constant"]=0.0
    d["attack_size_mismatch"]=2.0
    d["initial_std_constant"]=0.0
    d["initial_std_mismatch"]=0.0001
    d["initial_std_mismatch"]=0.0
    return d

def make_hyperparameters_ecg():
    return make_defaults_ecg().keys()

def get_flags_ecg():
    help=make_help()
    for key, value in make_defaults_ecg().items():
        parser = argparse.ArgumentParser()
        parser.add_argument(key,type=type(value),default=value,help=help.get(key,default=""))
    parser.add_argument("data_dir",type=str,default='ECG/ecg_recordings/',help='Directory of ECG signals')
    return parser.parse_args()

def make_defaults_cnn():
    d = make_defaults()
    d["eval_step_interval"]=1000
    d["attack_size_constant"]=0.0
    d["attack_size_mismatch"]=1.0
    d["initial_std_constant"]=0.0
    d["initial_std_mismatch"]=0.0001
    d["Kernels"]="[[64,1,4,4],[64,64,4,4]]"
    d["Dense"]="[[1600,256],[256,64],[6410]]"
    d["initial_std_mismatch"]=0.0
    return d

def make_hyperparameters_cnn():
    return make_defaults_cnn().keys()

def get_flags_cnn():
    help=make_help()
    for key, value in make_defaults_cnn().items():
        parser = argparse.ArgumentParser()
        parser.add_argument(key,type=type(value),default=value,help=help.get(key,default=""))
    parser.add_argument("data_dir",t=str,default='CNN/fashion_mnist/',help='Directory of fashion mnist images')
    return parser.parse_args()