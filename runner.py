from GraphExecution import utils
import copy
import thread
DEFAULT_N_HIDDEN = 200 #???
DEFAULT_BETA_LIPSCHITZNESS = 10 #???
BATCH_SIZE = 100 #???

default = ["--batch_size=100","--model_architecture=lsnn","--n_hidden=200", "--wanted_words=yes,no", "--use_epsilon_ball", "--epsilon_lipschitzness=0.01", "--num_steps_lipschitzness=10","--beta_lipschitzness=10.0"]
parser = utils.get_parser()
defaultparams = vars(parser.parse_args(default))

import os

def grid(params):
    for key in params:
        if type(params[key])==list:
            return [grid(copy.copy(params).update({key:value})) for value in params[key]]
    return [params]

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
            command += "--" + key + "=" + params[key]
        os.system(command)
    
def run_models(pparams, force = False):
    try:
        for params in grid(pparams):
            thread.start_new_thread(run_model, (params, force))
    except:
        print("unable to run models")

def get_model(params):
    pass

def get_models(pparams):
    pass
    
def experiment_a(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] += [0]
    run_models(pparams, force)
    #TODO run experiment

def experiment_b(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] += [0]
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
    pparams["n_hidden"] = [defaultparams["n_hidden"]*(2**i) for i in range [0,1,2,3,4]]
    run_models(pparams, force)
    #TODO run experiment

def experiment_e(force=False):
    pparams = copy.copy(defaultparams)
    pparams["seed"] = [0,1,2,3,4,5,6,7,8,9]
    pparams["beta_lipschitzness"] = [0,0.001,0.01,0.1,1]
    run_models(pparams, force)
    #TODO run experiment

if __name__ == "main":
    experiment_a()
    experiment_b()
    experiment_c()
    experiment_d()
    experiment_e()