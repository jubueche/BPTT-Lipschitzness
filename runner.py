from GraphExecution import utils
import copy
from threading import Thread
import os
from itertools import zip_longest

defaultparams = {}
defaultparams["batch_size"] = 1
defaultparams["eval_step_interval"] = 5
defaultparams["model_architecture"] = "lsnn"
defaultparams["n_hidden"] = 10
defaultparams["wanted_words"] = 'yes,no'
defaultparams["use_epsilon_ball"] = True
defaultparams["epsilon_lipschitzness"] = 0.01
defaultparams["num_steps_lipschitzness"] = 10
defaultparams["beta_lipschitzness"] = 1.0
defaultparams["how_many_training_steps"] = "5,5"


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
    name_end = '_{}_h{}_b{}_s{}'.format(params["model_architecture"], params["n_hidden"], params["beta_lipschitzness"], params["seed"])
    for file in os.listdir(model_path):
        if name_end in file:
            return os.path.join(name_end, file)
    return None

def run_model(params, force=False):
    model_file = find_model(params)
    if not model_file is None and not force:
        print("Found Model, skipping")
        return
    else:
        print("Training model {}".format(params))
        command = "python Jax/main_jax.py "
        for key in params:
            if type(params[key]) == bool:
                if params[key]==True:
                    command+= "--" + key + " "
            else:
                command += "--" + key + "=" + str(params[key]) + " "
        os.system(command)
    
def run_models(pparams, force = False, parallelness = 10):
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
    
def experiment_a(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [defaultparams["beta_lipschitzness"],0]
    run_models(pparams, force)
    #TODO run experiment

def experiment_b(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [defaultparams["beta_lipschitzness"],0]
    run_models(pparams, force)
    #TODO run experiment

def experiment_c(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [0,0.001,0.01,0.1,1]
    run_models(pparams, force)
    #TODO run experiment

def experiment_d(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [0,0.001,0.01,0.1,1]
    pparams["n_hidden"] = [defaultparams["n_hidden"]*(2**i) for i in [0,1,2,3,4]]
    run_models(pparams, force)
    #TODO run experiment

def experiment_e(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [0,0.001,0.01,0.1,1]
    run_models(pparams, force)
    #TODO run experiment

experiment_a()
experiment_b()
experiment_c()
experiment_d()
experiment_e()