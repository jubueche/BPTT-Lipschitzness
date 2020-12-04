# from run_utils import *


# g = {"this":"is", "a":"test", "dictionary":24, "hi":"how"}

# g= split(g,"dictionary", [1,2,3,4,5,6])

# def concatenate(model):
#     print("executing")
#     print(model)
#     model["concatenated"] = model["this"]+ model["a"]+str(model["dictionary"])

# g = configure(g, {"concat_function":concatenate, "concat_checker":check(load_file,'found file'),"concat_dependencies": ["this","a","dictionary"]})

# g = run("concat", g)
# print(["concatenated" in d for d in g])

# run(save, g, "concat")

# g = configure(g, {"echo_command": "echo $$concatenated$$"})
# run("echo", g)

# import inspect

# [obj for _, obj in inspect.getmembers(sys.modules[__name__] if inspect.isclass())

from run_utils import format_template, run

bla = {'clip_duration_ms': 1000, 'window_size_ms': 30.0, 'window_stride_ms': 10.0, 'feature_bin_count': 40, 'sample_rate': 16000, 'in_repeat': 1, 'preprocess': 'mfcc', 'data_url': 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz', 'silence_percentage': 10, 'unknown_percentage': 10, 'validation_percentage': 10, 'testing_percentage': 10, 'n_thr_spikes': -1, 'background_volume': 0.1, 'background_frequency': 0.8, 'time_shift_ms': 100.0, 'dt': 1.0, 'tau': 20.0, 'beta': 2.0, 'tau_adaptation': 98.0, 'thr': 0.01, 'thr_min': 0.005, 'refr': 2, 'dampening_factor': 0.3, 'dropout_prob': 0.0, 'reg': 0.001, 'learning_rate': '0.001,0.0001', 'wanted_words': 'yes,no', 'n_epochs': '32,8', 'n_attack_steps': 10, 'beta_robustness': 1.0, 'seed': 0, 'n_hidden': 256, 'n_layer': 1, 'batch_size': 100, 'eval_step_interval': 200, 'attack_size_constant': 0.0, 'attack_size_mismatch': 2.0, 'initial_std_constant': 0.0, 'initial_std_mismatch': 0.0, 'data_dir': 'ECG/ecg_recordings/', 'launch': 'python {code_file} {make_args}', 'code_file': 'main_ecg_lsnn.py'}

my_dict = {'clip_duration_ms': 1000, 
            'window_size_ms': 30.0,
            'window_stride_ms': 10.0, 
            'feature_bin_count': 40, 
            'sample_rate': 16000, 
            'in_repeat': 1, 
            'preprocess': 'mfcc',
             'data_url': 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
             'silence_percentage': 10, 'unknown_percentage': 10, 'validation_percentage': 10, 'testing_percentage': 10, 'n_thr_spikes': -1, 'background_volume': 0.1, 'background_frequency': 0.8, 'time_shift_ms': 100.0, 'dt': 1.0, 'tau': 20.0, 'beta': 2.0, 'tau_adaptation': 98.0, 'thr': 0.01, 'thr_min': 0.005, 'refr': 2, 'dampening_factor': 0.3, 'dropout_prob': 0.0, 'reg': 0.001, 'learning_rate': '0.001,0.0001', 'wanted_words': 'yes,no', 'n_epochs': '32,8', 'n_attack_steps': 10, 'beta_robustness': 1.0, 'seed': 9, 'n_hidden': 256, 'n_layer': 1, 'batch_size': 100, 'eval_step_interval': 200, 'attack_size_constant': 0.0, 'attack_size_mismatch': 2.0, 'initial_std_constant': 0.0, 'initial_std_mismatch': 0.001, 'data_dir': 'ECG/ecg_recordings/', 
             'launch': 'python {code_file} {make_args}', 'code_file': 'main_ecg_lsnn.py', 'make_args': lambda model: " ".join(["-" + key + " = " + str(model[key]) for key in ret]),
             }

run(format_template,my_dict,"{launch}")