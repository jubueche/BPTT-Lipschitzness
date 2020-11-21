from jax import config
config.FLAGS.jax_log_compiles=False
from GraphExecution import utils
from threading import Thread
from RNN_Jax import RNN
from CNN_Jax import CNN
import input_data_eager as input_data
import jax.numpy as jnp
from loss_jax import attack_network
import jax.random as jax_random
import copy
import os
import sys
from itertools import zip_longest
import argparse
from datetime import datetime
import numpy as onp
import ujson as json
from six.moves import xrange
import sqlite3
from random import randint
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import DataLoader
from jax import partial, jit
import matplotlib.pyplot as plt


# - Set numpy seed
onp.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4,5,6,7,8,9])
parser.add_argument(
    '--force',
    action='store_true',
    default=False,
    help='Retrains all models even if their files are present.')
parser.add_argument(
    '--db',
    type=str,
    default = "default"
)
ARGS = parser.parse_args()

LEONHARD = True

defaultparams = {}
defaultparams["batch_size"] = 100
defaultparams["eval_step_interval"] = 200
defaultparams["model_architecture"] = "lsnn"
defaultparams["n_hidden"] = 256
defaultparams["wanted_words"] = 'yes,no'
defaultparams["attack_epsilon"] = 0.01
defaultparams["beta_lipschitzness"] = 1.0
defaultparams["n_epochs"] = "64,16"
defaultparams["relative_initial_std"] = False
defaultparams["relative_epsilon"] = False
defaultparams["num_attack_steps"] = 10
defaultparams["db"] = ARGS.db

defaultparams_ecg = {}
defaultparams_ecg["batch_size"] = 100
defaultparams_ecg["model_architecture"] = "lsnn_ecg"
defaultparams_ecg["n_hidden"] = 256
defaultparams_ecg["attack_epsilon"] = 2.0
defaultparams_ecg["beta_lipschitzness"] = 1.0
defaultparams_ecg["n_epochs"] = "32,8"
defaultparams_ecg["relative_initial_std"] = False
defaultparams_ecg["relative_epsilon"] = False
defaultparams_ecg["num_attack_steps"] = 10
defaultparams_ecg["db"] = ARGS.db

defaultparams_cnn = {}
defaultparams_cnn["batch_size"] = 100
defaultparams_cnn["model_architecture"] = "cnn"
defaultparams_cnn["beta_lipschitzness"] = 1.0
defaultparams_cnn["n_epochs"] = "32,8"
defaultparams_cnn["num_attack_steps"] = 10
defaultparams_cnn["db"] = ARGS.db

# LEONHARD = False
# defaultparams["n_hidden"] = 16
# defaultparams["batch_size"] = 5
# defaultparams["eval_step_interval"] = 1
# defaultparams["n_epochs"] = "1"
# defaultparams["learning_rate"] = 0.001

if LEONHARD:
    defaultparams["data_dir"]="$SCRATCH/speech_dataset"
    defaultparams_ecg["data_dir"] = "$SCRATCH/ecg_recordings"
    defaultparams_cnn["data_dir"] = "$SCRATCH/fashion_mnist"

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
        return flatten_lists([grid(p) for p in params])


def find_model(params, get_track = False):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "Resources/")
    track_path = os.path.join(base_path, "Resources/Plotting/")
    def format_value(val):
        if type(val) == int:
            return str(val)
        if type(val) == float:
            return str(val)
        return "\"" + str(val) + "\""
        
    try:
        conn = sqlite3.connect("sessions_" + ARGS.db +".db")
        command = "SELECT session_id FROM sessions WHERE {0} ORDER BY start_time DESC LIMIT 1;".format(" AND ".join(map("=".join,zip(params.keys(),map(format_value,params.values())))))
        
        cur = conn.cursor()
        cur.execute(command)
        result = cur.fetchall()
    except sqlite3.Error as error:
        return None
    finally:
        if (conn):
            conn.close()

    if len(result)==0:
        return None
    session_id = result[0]
    if(type(session_id) is tuple):
        session_id = session_id[0]
    
    if(get_track):
        return os.path.join(track_path, str(session_id)+"_track.json")
    else:
        return os.path.join(model_path, str(session_id)+ "_model.json")
    

def estimate_memory(params):
    if params["n_hidden"] >= 128:
        return 4096
    return 1024

def estimate_time(params):
    return "24:00"

def estimate_cores(params):
    return 16

def run_model(params, force=False):
    model_file = find_model(params)
    if not model_file is None and not force:
        print("Found Model, skipping")
        return
    else:
        session_id = randint(1000000000, 9999999999)
        params["session_id"] = session_id
        if LEONHARD:
            script = "main_jax.py "
            if(params["model_architecture"] == "lsnn_ecg"):
                script = "main_ecg_classifier.py "
            elif(params["model_architecture"] == "cnn"):
                script = "main_CNN_jax.py "
            os.system("module load python_cpu/3.7.1")
            logfilename = str(session_id)+'.log'
            command = "bsub -o ../logs/"+ logfilename +" -W " + str(estimate_time(params)) + " -n " + str(estimate_cores(params)) + " -R \"rusage[mem=" + str(estimate_memory(params)) + "]\" \"python3 " + script
        else:
            command = "python main_jax.py "
        for key in params:
            if type(params[key]) == bool:
                if params[key]==True:
                    command+= "--" + key + " "
            else:
                command += "--" + key + "=" + str(params[key]) + " "
        if LEONHARD:
            command += '\"'
        print(command)
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
        theta_star[key] = theta[key] + onp.random.normal(loc=0.0,scale=eps, size=theta[key].shape)
    return theta_star

def get_surface_point(theta, eps):
    theta_star = {}
    for key in theta.keys():
        theta_star[key] = theta[key] + eps * onp.sign(onp.random.uniform(low=-1, high=1, size=theta[key].shape))
    return theta_star

def apply_mismatch(theta, mm_std):
    """Apply mismatch noise to theta with given percentage"""
    theta_star = {}
    for key in theta.keys():
        theta_star[key] = theta[key] * (1 + mm_std * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    return theta_star

def get_model(params, arch="rnn"):
    model_fn = find_model(params)
    if(model_fn is None):
        print("Cant find model with "); print(params) 
        sys.exit(0)
    else:
        if(arch=="cnn"):
            model = CNN.load(model_fn)
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
        models.append(get_model(params, params["model_architecture"]))
    return models

# - This should move into another file
def get_batched_accuracy(y, logits):
    if(y.ndim > 1):
        y = onp.argmax(y, axis=1)
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

class MyFLAGS(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def get_test_acc(audio_processor, rnn, theta_star, ATTACK):
    set_size = audio_processor.set_size('testing')
    total_accuracy = 0.0
    FLAGS = MyFLAGS(rnn.model_settings)
    if(ATTACK):
        attacked_total_accuracy = 0.0
    for i in xrange(0, set_size, FLAGS.batch_size):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(FLAGS.batch_size, i, rnn.model_settings ,0.0, 0.0, 0.0, 'testing'))
        X = validation_fingerprints.numpy()
        y = validation_ground_truth.numpy()
        logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **theta_star)
        if(ATTACK):
            _, logits_theta_star = attack_network(X, theta_star, logits, rnn, FLAGS, rnn._rng_key)
            rnn._rng_key, _ = jax_random.split(rnn._rng_key)
            attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
            attacked_total_accuracy += (attacked_batched_validation_acc * FLAGS.batch_size) / set_size
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * FLAGS.batch_size) / set_size
    if(ATTACK):
        return onp.float64(attacked_total_accuracy)
    else:
        return onp.float64(total_accuracy)

def get_test_acc_ecg(ecg_processor, rnn, theta_star):
    set_size = ecg_processor.N_test
    total_accuracy = 0.0
    bs = 200
    for _ in range(0, int(onp.ceil(set_size/bs))):
        X,y = ecg_processor.get_batch("test", batch_size=bs)
        logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **theta_star)
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * X.shape[0]) / set_size
    return onp.float64(total_accuracy)


def load_audio_processor(model, data_dir = "tmp/speech_dataset"):
    audio_processor_settings = model.model_settings
    audio_processor_settings["data_dir"] = data_dir
    audio_processor = get_audio_processor(audio_processor_settings)
    return audio_processor
    
def experiment_a(pparams, ATTACK = False):
    mismatch_levels = pparams[0].pop("mismatch_levels")
    num_iter = pparams[0].pop("num_iter")
    # - Get all the models that we need
    models = get_models(pparams)
    # - Get the audio processor
    audio_processor = load_audio_processor(models[0][0])
    mm_dict = {'0.0': []}
    for mismatch_level in mismatch_levels:
        mm_dict[str(mismatch_level)] = []
    # - For each level of mismatch, except for 0.0, and each model, we have to re-draw from the Gaussian. We do that 10 times and record the testing accuracy.
    experiment_dict = {"normal": copy.deepcopy(mm_dict), "robust": copy.deepcopy(mm_dict)}
    for model in models:
        # - Unpack the model
        rnn, theta, _ = model
        network_type = "normal"
        if(rnn.model_settings["beta_lipschitzness"] > 0.0):
            network_type = "robust"
        # - Get the test accuracy for 0.0 mismatch (and we don't need to repeat this)
        test_acc = get_test_acc(audio_processor, rnn, theta, False)
        experiment_dict[network_type]['0.0'].append(test_acc)
        print(f"Mismatch level 0.0 Network type {network_type} testing accuracy {test_acc}")
        for mismatch_level in mismatch_levels:
            if(ATTACK):
                rnn.model_settings["attack_epsilon"] = mismatch_level
            for _ in range(num_iter):
                theta_star = apply_mismatch(theta, mm_std=mismatch_level)
                # - Get the testing accuracy
                if(ATTACK):
                    test_acc = get_test_acc(audio_processor, rnn, theta, True)
                else:
                    test_acc = get_test_acc(audio_processor, rnn, theta_star, False)
                print(f"Mismatch level {mismatch_level} Network type {network_type} testing accuracy {test_acc}")
                # - Append to the correct list in the track
                experiment_dict[network_type][str(mismatch_level)].append(test_acc)

    # - We are done. Return the experiment dict
    return experiment_dict

def experiment_b(pparams):
    gaussian_eps = pparams.pop("gaussian_eps")
    gaussian_attack_eps = pparams.pop("attack_epsilon")
    n_iter = pparams.pop("num_iter")
    # - Get the models
    models = get_models(pparams)
    # - Get the audio processor
    audio_processor = load_audio_processor(models[0][0])
    # - Initialize the dictionary to hold the information
    model_grid = grid(pparams)
    experiment_dict = {"experiment_params": {"gaussian_eps": gaussian_eps, "gaussian_attack_eps": gaussian_attack_eps}}
    for i,model_params in enumerate(model_grid):
        experiment_dict[str(i)] = {"normal_test_acc" : [], "mismatch": [], "gaussian": [], "gaussian_attack": [], "gaussian_with_eps_attack": [], "ball_surface": [], "ball_surface_larger_eps": [],  "model_params": model_params}
    # - Iterate over each model
    for model_idx,model in enumerate(models):
        # - Unpack
        rnn, theta, _ = model
        assert (rnn.model_settings["attack_epsilon"] == gaussian_attack_eps), "Attack epsilon of RNN not the same as passed attack epsilon"
        # - Get the normal test accuracy
        test_acc_normal = get_test_acc(audio_processor, rnn, theta, False)
        # - Assign to the dictionary
        experiment_dict[str(model_idx)]["normal_test_acc"].append(test_acc_normal)
        curr_beta = rnn.model_settings["beta_lipschitzness"]
        print(f"beta {curr_beta} and normal test acc is {test_acc_normal}")
        # - For n_iter times, draw new variables and get the test accuracy under attack
        for _ in range(n_iter):
            theta_gaussian = gaussian_noise(theta, eps = gaussian_eps)
            theta_surface = get_surface_point(theta, eps=gaussian_attack_eps)
            theta_surface_larger_eps = get_surface_point(theta, eps=gaussian_eps)
            theta_gaussian_eps_attack = gaussian_noise(theta, eps = gaussian_attack_eps)
            test_acc_gaussian_with_eps_attack = get_test_acc(audio_processor, rnn, theta_gaussian_eps_attack, False)
            test_acc_gaussian = get_test_acc(audio_processor, rnn, theta_gaussian, False)
            test_acc_attack = get_test_acc(audio_processor, rnn, theta, True)
            test_acc_surface = get_test_acc(audio_processor, rnn, theta_surface, False)
            test_acc_surface_larger_eps = get_test_acc(audio_processor, rnn, theta_surface_larger_eps, False)
            print(f"beta {curr_beta} gaussian acc {test_acc_gaussian} gaussian with eps of attack acc {test_acc_gaussian_with_eps_attack} attack acc {test_acc_attack} surface attack {test_acc_surface} surface attack larger eps {test_acc_surface_larger_eps}")
            # - Save
            experiment_dict[str(model_idx)]["gaussian"].append(test_acc_gaussian)
            experiment_dict[str(model_idx)]["gaussian_attack"].append(test_acc_attack)
            experiment_dict[str(model_idx)]["gaussian_with_eps_attack"].append(test_acc_gaussian_with_eps_attack)
            experiment_dict[str(model_idx)]["ball_surface"].append(test_acc_surface)
            experiment_dict[str(model_idx)]["ball_surface_larger_eps"].append(test_acc_surface_larger_eps)
    return experiment_dict

def experiment_c(pparams):
    gaussian_attack_eps = pparams.pop("attack_epsilon")
    n_iter = pparams.pop("num_iter")
    # - Get the models
    models = get_models(pparams)
    # - Get the audio processor
    audio_processor = load_audio_processor(models[0][0])
    # - Initialize the dictionary to hold the information
    model_grid = grid(pparams)
    experiment_dict = {"experiment_params": {"gaussian_attack_eps": gaussian_attack_eps}}
    for i,model_params in enumerate(model_grid):
        experiment_dict[str(i)] = {"normal_test_acc" : [], "gaussian_attack": [], "model_params": model_params}
    # - Iterate over each model
    for model_idx,model in enumerate(models):
        # - Unpack
        rnn, theta, tracking_dict = model
        # - Save the tracking dict
        experiment_dict[str(model_idx)]["tracking_dict"] =  tracking_dict 
        # - Get the normal test accuracy
        test_acc_normal = get_test_acc(audio_processor, rnn, theta, False)
        # - Assign to the dictionary
        experiment_dict[str(model_idx)]["normal_test_acc"].append(test_acc_normal)
        curr_beta = rnn.model_settings["beta_lipschitzness"]
        print(f"beta {curr_beta} and normal test acc is {test_acc_normal}")
        # - For n_iter times, draw new variables and get the test accuracy under attack
        for _ in range(n_iter):
            test_acc_attack = get_test_acc(audio_processor, rnn, theta, True)
            print(f"beta {curr_beta} attack acc {test_acc_attack}")
            # - Save
            experiment_dict[str(model_idx)]["gaussian_attack"].append(test_acc_attack)
    return experiment_dict

def experiment_d():
    pass
    #TODO run experiment

def experiment_e(pparams):
    # - Get the models
    models = get_models(pparams)
    model_grid = grid(pparams)
    experiment_dict = {"betas":[]}
    for i,model_params in enumerate(model_grid):
        experiment_dict[str(i)] = {"model_params": model_params}
    # - Iterate over each model
    for model_idx,model in enumerate(models):
        # - Unpack
        rnn, theta, tracking_dict = model
        # - Save the tracking dict
        for key in theta.keys():
            theta[key] = theta[key].tolist()
        experiment_dict[str(model_idx)]["tracking_dict"] =  tracking_dict
        experiment_dict[str(model_idx)]["theta"] = theta
        curr_beta = rnn.model_settings["beta_lipschitzness"]
        experiment_dict["betas"].append(curr_beta)
    return experiment_dict

def experiment_f(pparams):
    mismatch_levels = pparams[0].pop("mismatch_levels")
    num_iter = pparams[0].pop("num_iter")
    # - Get all the models that we need
    models = get_models(pparams)
    FLAGS = MyFLAGS(models[0][0].model_settings)
    # - Get the ECG Dataloader
    ecg_processor = ECGDataLoader(path=FLAGS.data_dir, batch_size=FLAGS.batch_size)

    mm_dict = {'0.0': []}
    for mismatch_level in mismatch_levels:
        mm_dict[str(mismatch_level)] = []
    # - For each level of mismatch, except for 0.0, and each model, we have to re-draw from the Gaussian. We do that 10 times and record the testing accuracy.
    experiment_dict = {"normal": copy.deepcopy(mm_dict), "robust": copy.deepcopy(mm_dict)}
    # - Get the full sequence and the chopped up signals and labels
    X,y,ecg_seq = ecg_processor.get_sequence(N_per_class=10, path=FLAGS.data_dir)
    X = onp.array(X)
    # - Perform class. of the sequence under fixed mm level
    rob_rnn, theta_rob, _ = models[0]
    norm_rnn, theta_norm, _ = models[1]
    assert rob_rnn.model_settings["beta_lipschitzness"] > 0.0
    found_good_rob = False; found_good_norm = False
    mm_level_ecg_seq = 0.2
    # - This is just for illustrative purposes. We search for a case where the difference is truly visible.
    # - This is of course not reflective of actual performance.
    for _ in range(100):
        if(not found_good_rob):
            theta_rob_star = apply_mismatch(theta_rob, mm_level_ecg_seq)
            logits, _ = rob_rnn.call(X, jnp.ones(shape=(1,rob_rnn.units)), **theta_rob_star)
            preds = onp.argmax(logits, axis=1)
            acc = get_batched_accuracy(jnp.array(y),logits)
            print(f"Robust {acc}")
            if(acc>0.9):
                found_good_rob = True
                experiment_dict["rob_pred_seq"] = preds.tolist()
        if(not found_good_norm):
            theta_norm_star = apply_mismatch(theta_norm, mm_level_ecg_seq)
            logits, _ = norm_rnn.call(X, jnp.ones(shape=(1,norm_rnn.units)), **theta_norm_star)
            preds = onp.argmax(logits, axis=1)
            acc = get_batched_accuracy(jnp.array(y),logits)
            print(f"Normal {acc}")
            if(acc<0.7):
                found_good_norm = True
                experiment_dict["norm_pred_seq"] = preds.tolist()
        if(found_good_norm and found_good_rob):
            break
    experiment_dict["ecg_seq"] = ecg_seq.tolist()
    experiment_dict["ecg_seq_y"] = onp.array(y, dtype=onp.float64).tolist()

    for model in models:
        # - Unpack the model
        rnn, theta, _ = model
        network_type = "normal"
        if(rnn.model_settings["beta_lipschitzness"] > 0.0):
            network_type = "robust"
        # - Get the test accuracy for 0.0 mismatch (and we don't need to repeat this)
        test_acc = get_test_acc_ecg(ecg_processor, rnn, theta)
        experiment_dict[network_type]['0.0'].append(test_acc)
        print(f"Mismatch level 0.0 Network type {network_type} testing accuracy {test_acc}")
        for mismatch_level in mismatch_levels:
            for _ in range(num_iter):
                theta_star = apply_mismatch(theta, mm_std=mismatch_level)
                # - Get the testing accuracy
                test_acc = get_test_acc_ecg(ecg_processor, rnn, theta_star)
                print(f"Mismatch level {mismatch_level} Network type {network_type} testing accuracy {test_acc}")
                # - Append to the correct list in the track
                experiment_dict[network_type][str(mismatch_level)].append(test_acc)

    return experiment_dict

def experiment_g(pparams):
    mismatch_levels = pparams[0].pop("mismatch_levels")
    num_iter = pparams[0].pop("num_iter")
    # - Get all the models that we need
    models = get_models(pparams)
    FLAGS = MyFLAGS(models[0][0].model_settings)
    # - Get the CNN data loader
    data_loader = DataLoader(FLAGS.batch_size)
    n_per_class = 5
    nc = 4
    classes = {0 : "Shirt", 1 : "Trouser", 4: "Jacket", 5: "Shoe"}
    image_dict, X, y = data_loader.get_n_images(n_per_class, list(classes.keys()))
    # fig=plt.figure(figsize=(8, 8))
    # for i,idx in enumerate(classes.keys()):
    #     for j in range(n_per_class):
    #         fig.add_subplot(nc, n_per_class, 1 + i*n_per_class + j)
    #         plt.imshow(onp.squeeze(image_dict[str(idx)][j]))
    # plt.show()

    mm_dict = {'0.0': []}
    for mismatch_level in mismatch_levels:
        mm_dict[str(mismatch_level)] = []
    # - For each level of mismatch, except for 0.0, and each model, we have to re-draw from the Gaussian. We do that 10 times and record the testing accuracy.
    experiment_dict = {"normal": copy.deepcopy(mm_dict), "robust": copy.deepcopy(mm_dict)}

    rob_cnn, theta_rob, _ = models[0]
    norm_cnn, theta_norm, _ = models[1]
    assert rob_cnn.model_settings["beta_lipschitzness"] > 0.0
    found_good_rob = False; found_good_norm = False
    mm_level_im_seq = 0.3
    for _ in range(100):
        if(not found_good_rob):
            theta_rob_star = apply_mismatch(theta_rob, mm_level_im_seq)
            logits, _ = rob_cnn.call(X, [[1]], **theta_rob_star)
            preds = onp.argmax(logits, axis=1)
            acc = get_batched_accuracy(jnp.array(onp.argmax(y, axis=1)),logits)
            print(f"Robust {acc}")
            if(acc>0.9):
                found_good_rob = True
                experiment_dict["rob_pred_seq"] = preds.tolist()
        if(not found_good_norm):
            theta_norm_star = apply_mismatch(theta_norm, mm_level_im_seq)
            logits, _ = norm_cnn.call(X, [[1]], **theta_norm_star)
            preds = onp.argmax(logits, axis=1)
            acc = get_batched_accuracy(jnp.array(onp.argmax(y,axis=1)),logits)
            print(f"Normal {acc}")
            if(acc<0.7):
                found_good_norm = True
                experiment_dict["norm_pred_seq"] = preds.tolist()
        if(found_good_norm and found_good_rob):
            break
    if(not (found_good_rob and found_good_norm)):
        sys.exit(0)
    experiment_dict["image_seq"] = X.tolist()
    experiment_dict["image_seq_y"] = onp.array(onp.argmax(y,axis=1), dtype=onp.float64).tolist()
    experiment_dict["class_labels"] = classes

    for model in models:
        # - Unpack the model
        cnn, theta, _ = model
        network_type = "normal"
        if(cnn.model_settings["beta_lipschitzness"] > 0.0):
            network_type = "robust"
        # - Get the test accuracy for 0.0 mismatch (and we don't need to repeat this)
        test_acc = get_test_acc_ecg(data_loader, cnn, theta)
        experiment_dict[network_type]['0.0'].append(test_acc)
        print(f"Mismatch level 0.0 Network type {network_type} testing accuracy {test_acc}")
        for mismatch_level in mismatch_levels:
            for _ in range(num_iter):
                theta_star = apply_mismatch(theta, mm_std=mismatch_level)
                # - Get the testing accuracy
                test_acc = get_test_acc_ecg(data_loader, cnn, theta_star)
                print(f"Mismatch level {mismatch_level} Network type {network_type} testing accuracy {test_acc}")
                # - Append to the correct list in the track
                experiment_dict[network_type][str(mismatch_level)].append(test_acc)

    return experiment_dict


pparams = copy.copy(defaultparams)
pparams["seed"] = ARGS.seeds
pparams["beta_lipschitzness"] = [0.0,0.01*defaultparams["beta_lipschitzness"],0.1*defaultparams["beta_lipschitzness"],1.0*defaultparams["beta_lipschitzness"],2.0*defaultparams["beta_lipschitzness"],10.0*defaultparams["beta_lipschitzness"]]
pparams["n_hidden"] = 256
# if(LEONHARD):
    # run_models(pparams, ARGS.force)

###MISMATCH BALL MODELS
pparams = copy.copy(defaultparams)
pparams["seed"] = ARGS.seeds
pparams["beta_lipschitzness"] = 1.0
pparams["relative_initial_std"] = True
pparams["relative_epsilon"] = True
pparams["attack_epsilon"] = 2.0

pparams2 = copy.copy(defaultparams)
pparams2["seed"] = ARGS.seeds
pparams2["beta_lipschitzness"] = 0
pparams2["relative_initial_std"] = True
pparams2["relative_epsilon"] = True

# if(LEONHARD):
#     run_models([pparams,pparams2],ARGS.force)

pparams_ecg = copy.copy(defaultparams_ecg)
pparams_ecg["seed"] = ARGS.seeds
pparams_ecg["beta_lipschitzness"] = [0.0,1.0]
pparams_ecg["relative_initial_std"] = True
pparams_ecg["relative_epsilon"] = True
pparams_ecg["attack_epsilon"] = 2.0

# if(LEONHARD):
#     run_models(pparams_ecg,ARGS.force)

pparams_cnn = copy.copy(defaultparams_cnn)
pparams_cnn["seed"] = ARGS.seeds
pparams_cnn["n_hidden"] = 256
pparams_cnn["beta_lipschitzness"] = [0.0,1.0]
pparams_cnn["relative_initial_std"] = True
pparams_cnn["relative_epsilon"] = True
pparams_cnn["attack_epsilon"] = 1.0

if(LEONHARD):
    run_models(pparams_cnn,ARGS.force)

if(LEONHARD):
    # - Exit here before we run experiments on Leonhard login node
    sys.exit(0)

# - Check if the folder "Experiments" exists. If not, create
if(not os.path.exists("Experiments/")):
    os.mkdir("Experiments/")

experiment_a_params = copy.copy(defaultparams)
experiment_b_params = copy.copy(defaultparams)
experiment_c_params = copy.copy(defaultparams)
experiment_d_params = copy.copy(defaultparams)
experiment_e_params = copy.copy(defaultparams)
experiment_f_params = copy.copy(defaultparams_ecg)
experiment_g_params = copy.copy(defaultparams_cnn)
experiment_g_params2 = copy.copy(defaultparams_cnn)
experiment_a_params["seed"] = ARGS.seeds
experiment_b_params["seed"] = ARGS.seeds
experiment_c_params["seed"] = ARGS.seeds
experiment_d_params["seed"] = ARGS.seeds
experiment_e_params["seed"] = ARGS.seeds
experiment_f_params["seed"] = ARGS.seeds
experiment_g_params["seed"] = ARGS.seeds
experiment_g_params2["seed"] = ARGS.seeds

experiments = []
####################### A #######################

# - Use the default parameters (best parameters) for the mismatch experiment
experiment_a_params["beta_lipschitzness"] = [defaultparams["beta_lipschitzness"]]
experiment_a_params["relative_initial_std"] = True
experiment_a_params["relative_epsilon"] = True
experiment_a_params["attack_epsilon"] = 2.0
experiment_a_params["mismatch_levels"] = [0.5,0.7,0.9,1.1,1.5]
experiment_a_params["num_iter"] = 50
experiment_a_params_attack = copy.deepcopy(experiment_a_params)
experiment_a_params_attack["mismatch_levels"] = [0.1,0.2,0.5,2.0,7.0]
experiment_a_params_attack["num_iter"] = 10
experiment_a_params2 = copy.deepcopy(experiment_a_params)
experiment_a_params2.pop("attack_epsilon")
experiment_a_params2.pop("mismatch_levels")
experiment_a_params2.pop("num_iter")
experiment_a_params2["beta_lipschitzness"] = 0.0
experiment_a_path = "Experiments/experiment_a.json"
experiment_a_path_attack = "Experiments/experiment_a_attack.json"
experiments.append(copy.deepcopy(experiment_a_params))
experiments.append(copy.deepcopy(experiment_a_params_attack))

if(os.path.exists(experiment_a_path)):
    print("File for experiment A already exists. Skipping...")
else:
    experiment_a_return_dict = experiment_a([experiment_a_params,experiment_a_params2])
    # - Save the data for experiment a
    with open(experiment_a_path, "w") as f:
        json.dump(experiment_a_return_dict, f)
    print("Successfully completed Experiment A.")

if(os.path.exists(experiment_a_path_attack)):
    print("File for experiment A ATTACK already exists. Skipping...")
else:
    experiment_a_attack_return_dict = experiment_a([experiment_a_params_attack,experiment_a_params2], ATTACK=True)
    with open(experiment_a_path_attack, "w") as f:
        json.dump(experiment_a_attack_return_dict, f)
    print("Successfully completed Experiment A ATTACK.")

####################### B ####################### 


experiment_b_params["beta_lipschitzness"] = [0.0,0.001,0.01,0.1,1.0,10.0]
experiment_b_params["gaussian_eps"] = 0.1
experiment_b_params["num_iter"] = 10
experiment_b_path = "Experiments/experiment_b.json"
experiments.append(copy.deepcopy(experiment_b_params))
# - Check if the path exists
if(True or os.path.exists(experiment_b_path)):
    print("File for experiment B already exists. Skipping...")
else:
    experiment_b_return_dict = experiment_b(experiment_b_params)
    with open(experiment_b_path, "w") as f:
        json.dump(experiment_b_return_dict, f)
    print("Successfully completed Experiment B.")

####################### C ####################### 

experiment_c_params["beta_lipschitzness"] = [0.0,0.1,1.0,10.0]
experiment_c_params["num_iter"] = 10
experiment_c_path = "Experiments/experiment_c.json"
experiments.append(copy.deepcopy(experiment_c_params))
# - Check if the path exists
if(os.path.exists(experiment_c_path)):
    print("File for experiment C already exists. Skipping...")
else:
    experiment_c_return_dict = experiment_c(experiment_c_params)
    with open(experiment_c_path, "w") as f:
        json.dump(experiment_c_return_dict, f)
    print("Successfully completed Experiment C.")

####################### D #######################

####################### E #######################

experiment_e_params["beta_lipschitzness"] = [0.0,0.1,1.0,10.0]
experiment_e_path = "Experiments/experiment_e.json"
experiments.append(copy.deepcopy(experiment_e_params))
# - Check if the path exists
if(os.path.exists(experiment_e_path)):
    print("File for experiment E already exists. Skipping...")
else:
    experiment_e_return_dict = experiment_e(experiment_e_params)
    with open(experiment_e_path, "w") as f:
        json.dump(experiment_e_return_dict, f)
    print("Successfully completed Experiment E.")

####################### F #######################

experiment_f_path = "Experiments/experiment_f.json"
experiment_f_params["relative_initial_std"] = True
experiment_f_params["relative_epsilon"] = True
experiment_f_params["attack_epsilon"] = 2.0
experiment_f_params["beta_lipschitzness"] = 1.0
experiment_f_params["mismatch_levels"] = [0.1,0.2,0.3,0.5,0.7]
experiment_f_params["num_iter"] = 50
experiment_f_params_2 = copy.deepcopy(experiment_f_params)
experiment_f_params_2.pop("relative_initial_std")
experiment_f_params_2.pop("relative_epsilon")
experiment_f_params_2.pop("attack_epsilon")
experiment_f_params_2.pop("mismatch_levels")
experiment_f_params_2.pop("num_iter")
experiment_f_params_2["beta_lipschitzness"] = 0.0
experiment_f_params_2["n_epochs"] = "16,4"
if(os.path.exists(experiment_f_path)):
    print("File for experiment F already exists. Skipping...")
else:
    experiment_f_return_dict = experiment_f([experiment_f_params,experiment_f_params_2])
    with open(experiment_f_path, "w") as f:
        json.dump(experiment_f_return_dict, f)
    print("Successfully completed Experiment F.")


####################### G #######################
experiment_g_path = "Experiments/experiment_g.json"
experiment_g_params["relative_initial_std"] = True
experiment_g_params["relative_epsilon"] = True
experiment_g_params["attack_epsilon"] = 1.0
experiment_g_params["mismatch_levels"] = [0.5,0.7,0.9,1.1,1.5]
experiment_g_params["num_iter"] = 50
experiment_g_params2["beta_lipschitzness"] = 0.0
if(os.path.exists(experiment_g_path)):
    print("File for experiment G already exists. Skipping...")
else:
    # experiment_g_return_dict = experiment_g([experiment_g_params,experiment_g_params2])
    experiment_g_return_dict = experiment_g([experiment_g_params,experiment_g_params2])
    with open(experiment_g_path, "w") as f:
        json.dump(experiment_g_return_dict, f)
    print("Successfully completed Experiment G.")

# - Print experiment parameters in Latex table format
print("Figure \t Architecture \t Conv. Layers \t FC Layers \t $\epsilon_\\textnormal{attack}$ \t $\epsilon_\\textnormal{gaussian}$ \t $L$ \t $\\beta$ \t Rel. $\epsilon$ \t $N$")
for idx,p in enumerate(experiments):
    ma = p["model_architecture"]
    nh = p["n_hidden"]
    ea = p["attack_epsilon"]
    ge = "$-$"
    if("gaussian_eps" in p.keys()):
        ge = p["gaussian_eps"]
    L = "$-$"
    if("mismatch_levels" in p.keys()):
        L = p["mismatch_levels"]
    beta = p["beta_lipschitzness"]
    rl = "$\\xmark$"
    if(p["relative_epsilon"]):
        rl = "$\\cmark$"
    nas = p["num_attack_steps"]
    
    if(ma == "lsnn"):
        ma = "sRNN"
    else:
        ma = "CNN"
    if(ma == "CNN"):
        cl = "[TODO]"
        fc = "$-$"
    else:
        cl = "$-$"
        fc = f"[{nh}]"
    print(f"{idx} \t {ma} \t\t {cl} \t\t {fc} \t\t {ea} \t\t {ge} \t\t {L} \t\t {beta} \t\t {rl} \t\t {nas}")
