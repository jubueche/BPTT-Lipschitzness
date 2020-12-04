import os
from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from RNN_Jax import RNN
import ujson as json
import jax.numpy as jnp
import numpy as onp
import jax.random as jax_random
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
import math

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

def test_accuracy_after_mismatch_attack(model):
    theta = model["theta"]
    attacked_theta = {}
    for key in theta:
        attacked_theta[key] = theta[key] * (1 + model["mm_std"] * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    model["attacked_theta"]=attacked_theta

    if model["architecture"] == "speech":
        set_size = model["audio_processor"].set_size('testing')
    
    if model["architecture"] == "ecg":
        pass
    
    total_accuracy = 0.0
    rnn = model["rnn"]

    audio_processor = model["audio_processor"]
    for i in range(0, set_size, model["batch_size"]):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(model["batch_size"], i, rnn.model_settings ,0.0, 0.0, 0.0, 'testing'))
        X = validation_fingerprints.numpy()
        y = validation_ground_truth.numpy()
        logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **attacked_theta)
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * model["batch_size"]) / set_size
    
    return onp.float64(total_accuracy)