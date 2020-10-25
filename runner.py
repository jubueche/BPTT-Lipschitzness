from GraphExecution import utils
import copy
from threading import Thread
import os
from itertools import zip_longest
import argparse
from datetime import datetime
import numpy as onp
from Jax.RNN_Jax import RNN
import input_data_eager as input_data
import ujson as json
from six.moves import xrange
import jax.numpy as jnp


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
defaultparams["batch_size"] = 100
# defaultparams["batch_size"] = 5
# defaultparams["eval_step_interval"] = 100
defaultparams["eval_step_interval"] = 2
defaultparams["model_architecture"] = "lsnn"
defaultparams["n_hidden"] = 256
# defaultparams["n_hidden"] = 16
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


def find_model(params, get_track = False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "Jax/Resources/")
    track_path = os.path.join(base_path, "Jax/Resources/Plotting/")
    name_end = '_{}_h{}_b{}_s{}'.format(params["model_architecture"], params["n_hidden"], float(params["beta_lipschitzness"]), params["seed"])
    search_path = model_path
    if(get_track):
        search_path = track_path
    for file in os.listdir(search_path):
        if name_end in file:
            return os.path.join(search_path, file)
    return None

def estimate_memory(params):
    if params["n_hidden"] >= 128:
        return 4096
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
            logfilename = '{}_{}_h{}_b{}_s{}.log'.format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), params["model_architecture"], params["n_hidden"], float(params["beta_lipschitzness"]), params["seed"])
            command = "bsub -o ../logs/"+ logfilename +" -W " + str(estimate_time(params)) + " -n " + str(estimate_cores(params)) + " -R \"rusage[mem=" + str(estimate_memory(params)) + "]\" \"python3 Jax/main_jax.py "
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

def gaussian_noise(theta, eps):
    """Apply Gaussian noise to theta with zero mean and eps standard deviation"""
    theta_star = {}
    for key in theta.keys():
        theta_star[key] = onp.random.randn(theta[key].shape[0],theta[key].shape[1]) * eps
    return theta_star

def apply_mismatch(theta, mm_std):
    """Apply mismatch noise to theta with given percentage"""
    theta_star = {}
    for key in theta.keys():
        theta_star[key] = theta[key] * (1 + mm_std * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    return theta_star


def get_model(params):
    model_fn = find_model(params)
    if(model_fn is None):
        raise Exception
    else:
        model = RNN.load(model_fn)
    track_fn = find_model(params, get_track=True)
    if(track_fn is None):
        raise Exception
    else:
        with open(track_fn, "r") as f:
            track_dict = json.load(f)
    return (model[0],model[1],track_dict)

def get_models(pparams):
    # - Models saves tuples of rnn objects and theta's which are returned by the RNN class
    models = []
    grid_params = grid(pparams)
    for params in grid_params:
        models.append(get_model(params))
    return models

# - This should move into another file
def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

# - This should move into another file
def get_audio_processor(model_params):
    audio_processor = input_data.AudioProcessor(
        model_params["data_url"], model_params["data_dir"],
        model_params["silence_percentage"], model_params["unknown_percentage"],
        model_params["wanted_words"].split(','), model_params["validation_percentage"],
        model_params["testing_percentage"], model_params, model_params["summaries_dir"],
        model_params["n_thr_spikes"], model_params["in_repeat"], model_params["seed"]
    )
    return audio_processor

def get_test_acc(audio_processor, rnn, theta_star):
    set_size = audio_processor.set_size('testing')
    total_accuracy = 0
    for i in xrange(0, set_size, rnn.model_settings["batch_size"]):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(rnn.model_settings["batch_size"], i, rnn.model_settings ,0.0, 0.0, 0.0, 'testing'))
        X = validation_fingerprints.numpy()
        y = validation_ground_truth.numpy()
        logits, _ = rnn.call(X, **theta_star)
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * rnn.model_settings["batch_size"]) / set_size
    return onp.float64(total_accuracy)
    
def experiment_a(pparams):
    mismatch_levels = [0.4, 0.5, 0.6, 0.7]
    num_iter = 10
    # - Get all the models that we need
    models = get_models(pparams)
    # - Get the audio processor
    audio_processor_settings = models[0][0].model_settings
    audio_processor_settings["data_dir"] = "tmp/speech_dataset"
    audio_processor = get_audio_processor(audio_processor_settings)
    # - We expect 2 * number of seeds many models
    assert len(models) == len(pparams["seed"]*2), "Number of models does not match expected"
    mm_dict = {'0.0': []}
    for mismatch_level in mismatch_levels:
        mm_dict[str(mismatch_level)] = []
    # - For each level of mismatch, except for 0.0, and each model, we have to re-draw from the Gaussian. We do that 10 times and record the testing accuracy.
    experiment_dict = {"normal": copy.copy(mm_dict), "robust": copy.copy(mm_dict)}
    for model in models:
        # - Unpack the model
        rnn, theta, track_dict = model
        network_type = "normal"
        if(rnn.model_settings["beta_lipschitzness"] > 0.0):
            network_type = "robust"
        # - Get the test accuracy for 0.0 mismatch (and we don't need to repeat this)
        test_acc = get_test_acc(audio_processor, rnn, theta)
        experiment_dict[network_type]['0.0'].append(test_acc)
        print(f"Mismatch level 0.0 Network type {network_type} testing accuracy {test_acc}")
        for mismatch_level in mismatch_levels:
            # - Attack the weights num_iter times and collect the test accuracy
            for _ in range(num_iter):
                theta_star = apply_mismatch(theta, mm_std=mismatch_level)
                # - Get the testing accuracy
                test_acc = get_test_acc(audio_processor, rnn, theta_star)
                print(f"Mismatch level {mismatch_level} Network type {network_type} testing accuracy {test_acc}")
                # - Append to the correct list in the track
                experiment_dict[network_type][str(mismatch_level)].append(test_acc)

    # - We are done. Return the experiment dict
    return experiment_dict

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
# run_models(pparams, ARGS.force)

# - Check if the folder "Experiments" exists. If not, create
if(not os.path.exists("Experiments/")):
    os.mkdir("Experiments/")

# - Use the default parameters (best parameters) for the mismatch experiment
experiment_a_params = copy.copy(defaultparams)
experiment_a_params["seed"] = ARGS.seeds
experiment_a_params["beta_lipschitzness"] = [0.0,defaultparams["beta_lipschitzness"]]
experiment_a_path = "Experiments/experiment_a.json"
# - Check if the path exists
if(os.path.exists(experiment_a_path)):
    print("File for experiment A already exists. Skipping...")
else:
    experiment_a_return_dict = experiment_a(experiment_a_params)
    # - Save the data for experiment a
    with open(experiment_a_path, "w") as f:
        json.dump(experiment_a_return_dict, f)
    print("Successfully completed Experiment A.")

experiment_b()
experiment_c()
experiment_d()
experiment_e()