from GraphExecution import utils
import copy
from threading import Thread
import os
from itertools import zip_longest
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9])
parser.add_argument(
    '--force',
    action='store_true',
    default=False,
    help='Retrains all models even if their files are present.')
ARGS = parser.parse_args()

LEONHARD = True

defaultparams = {}
# defaultparams["batch_size"] = 100
defaultparams["batch_size"] = 5
# defaultparams["eval_step_interval"] = 100
defaultparams["eval_step_interval"] = 2
defaultparams["model_architecture"] = "lsnn"
#defaultparams["n_hidden"] = 256
defaultparams["n_hidden"] = 16
defaultparams["wanted_words"] = 'yes,no'
defaultparams["use_epsilon_ball"] = True
defaultparams["epsilon_lipschitzness"] = 0.01
defaultparams["num_steps_lipschitzness"] = 10
defaultparams["beta_lipschitzness"] = 1.0
# defaultparams["how_many_training_steps"] = "15000,3000"
defaultparams["how_many_training_steps"] = "2,2"

if LEONHARD:
    defaultparams["data_dir"]="$SCRATCH/speech_dataset"

def grid(params):
    def flatten_lists(ll):
        flatten = lambda t: [item for sublist in t for item in sublist]
        
        if type(ll)==list and len(ll)>0 and type(ll[0])==list:
            return flatten(ll)
        return ll
    if type(params)==dict:
        for key in params:
            if type(params[key])==list:
                ret = []
                for value in params[key]:
                    p =  copy.copy(params)
                    p[key] = value
                    ret += [grid(p)]
                return flatten_lists(ret)
        return params
    if type(params)==list:
        return [grid(p) for p in params]


def find_model(params):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "Jax/Resources/")
    name_end = '_{}_h{}_b{}_s{}'.format(params["model_architecture"], params["n_hidden"], float(params["beta_lipschitzness"]), params["seed"])
    for file in os.listdir(model_path):
        if name_end in file:
            return os.path.join(name_end, file)
    return None

def estimate_memory(params):
    return 1024
    #return 4096

def estimate_time(params):
    return "00:30"
    # return "24:00"

def estimate_cores(params):
    return 1
    # return 16

def run_model(params, force=False):
    model_file = find_model(params)
    if not model_file is None and not force:
        print("Found Model, skipping")
        return
    else:
        print("Training model {}".format(params))
        if LEONHARD:
            logfilename = '{}_{}_h{}_b{}_s{}'.format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), params["model_architecture"], params["n_hidden"], float(params["beta_lipschitzness"]), params["seed"])
            command = "bsub -o ../logs/blabla.log -W " + str(estimate_time(params)) + " -n " + str(estimate_cores(params)) + " -R \"rusage[mem=" + str(estimate_memory(params)) + "]\" \"python3 Jax/main_jax.py "
        else:
            command = "python Jax/main_jax.py "
        for key in params:
            if type(params[key]) == bool:
                if params[key]==True:
                    command+= "--" + key + " "
            else:
                command += "--" + key + "=" + str(params[key]) + " "
        if LEONHARD:
            command += '\"'
        os.system(command)
    
def run_models(pparams, force = False, parallelness = 10):
    if LEONHARD:
        for params in grid(pparams):
            run_model(params,force)
        return
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(*args, fillvalue=fillvalue)
    threads = []
    for params in grid(pparams):
        t = Thread(target=run_model, args=(params, force))
        threads.append(t)
    for ts in grouper(threads,parallelness):
        for t in ts:
            t.start()
        for t in ts:
            t.join()

def get_model(params):
    pass

def get_models(pparams):
    pass

def experiment_a():
    pass
    #TODO run experiment

def experiment_b():
    pass
    #TODO run experiment

def experiment_c():
    pass
    #TODO run experiment

def experiment_d():
    pass
    #TODO run experiment

def experiment_e():
    pass
    #TODO run experiment

pparams = copy.copy(defaultparams)
pparams["seed"] = ARGS.seeds
pparams["beta_lipschitzness"] = [0.0,0.001*defaultparams["beta_lipschitzness"],0.01*defaultparams["beta_lipschitzness"],0.1*defaultparams["beta_lipschitzness"],1.0*defaultparams["beta_lipschitzness"],10.0*defaultparams["beta_lipschitzness"]]
pparams["n_hidden"] = [defaultparams["n_hidden"]*(2**i) for i in [0,1,2,3,4]]
run_models(pparams, ARGS.force)

experiment_a()
experiment_b()
experiment_c()
experiment_d()
experiment_e()