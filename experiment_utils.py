from jax import config
config.update('jax_disable_jit', False)

import os
from TensorCommands import input_data
from TensorCommands.data_loader import SpeechDataLoader
from TensorCommands.extract_data import prepare_npy
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from RNN_Jax import RNN
import ujson as json
import numpy as onp
from Hessian import hessian_computation, lanczos
from Hessian import density as density_lib
import jax.numpy as jnp
import jax.random as jax_random
from jax import grad
from loss_jax import loss_kl, categorical_cross_entropy, loss_normal, _get_logits
import matplotlib
import numpy as onp
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['image.cmap']='RdBu'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from architectures import standard_defaults
from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from copy import copy, deepcopy
from loss_jax import attack_network
from datajuicer import cachable, get
from architectures import speech_lsnn, ecg_lsnn, cnn
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class Namespace:
    def __init__(self,d):
        self.__dict__.update(d)

def get_batched_accuracy(y, logits):
    y = onp.array(y)
    logits = onp.array(logits)
    if(y.ndim > 1):
        y = onp.argmax(y, axis=1)
    correct_prediction = jnp.array(jnp.argmax(logits, axis=1) == y, dtype=jnp.float32)
    return jnp.mean(correct_prediction)

def quantise(M, bits):
    if(bits == -1):
        return M
    else:
        # - Include 0 in number of possible states
        base_weight = (onp.max(M)-onp.min(M)) / (2**bits - 1)
        if(base_weight == 0):
            return M
        else:
            return base_weight * onp.round(M / base_weight)

def get_lr_schedule(iteration, lrs):
    ts = onp.arange(1,sum(iteration),1)
    lr_sched = onp.zeros((len(ts),))
    for i in range(1,len(iteration)):
        iteration[i] += iteration[i-1]
    def get_lr(t):
        if(t < iteration[0]):
            return lrs[0]
        for i in range(1,len(iteration)):
            if(t < iteration[i] and t >= iteration[i-1]):
                return lrs[i]
    for idx,t in enumerate(ts):
        lr_sched[idx] = get_lr(t)
    lr_sched = jnp.array(lr_sched)
    def lr_schedule(t):
        return lr_sched[t]
    return lr_schedule

def get_loader(FLAGS, data_dir):
    if FLAGS.architecture=="speech_lsnn":
        loader = SpeechDataLoader(path=data_dir, batch_size=FLAGS.batch_size)
    elif FLAGS.architecture=="ecg_lsnn":
        loader = ECGDataLoader(path=data_dir, batch_size=FLAGS.batch_size)
    elif FLAGS.architecture=="cnn":
        loader = CNNDataLoader(FLAGS.batch_size, FLAGS.data_dir)
    return loader, loader.N_test

def get_X_y_pair(loader):
    X,y = loader.get_batch("test")
    return onp.array(X), onp.array(y)

def get_surface_data(model, surface_dist, data_dir):
    theta_star = {}
    theta = model["theta"]
    for key in theta:
        theta_star[key] = theta[key] + surface_dist * onp.sign(onp.random.uniform(low=-1, high=1, size=theta[key].shape))
    test_acc, _ = get_test_acc(model, theta_star, data_dir, ATTACK=False) 
    return test_acc

def _get_mismatch_data(model, theta, mm_level, data_dir, mode):
    theta_star = {}
    for key in theta:
        theta_star[key] = theta[key] * (1 + mm_level * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    acc, _, _, _ = _get_acc(model, theta_star, data_dir, False, mode)
    return acc

def get_mismatch_data(model, mm_level, data_dir, mode):
    theta = model["theta"]
    return _get_mismatch_data(model, theta, mm_level, data_dir, mode)

def get_whole_attacked_test_acc(model, data_dir, n_attack_steps, attack_size_mismatch, attack_size_constant, initial_std_mismatch, initial_std_constant):
    FLAGS = Namespace(model)
    max_size = 1000

    FLAGS.n_attack_steps = n_attack_steps
    FLAGS.attack_size_mismatch = attack_size_mismatch
    FLAGS.attack_size_constant = attack_size_constant
    FLAGS.initial_std_mismatch = initial_std_mismatch
    FLAGS.initial_std_constant = initial_std_constant

    loader, set_size = get_loader(FLAGS, data_dir)
    logits = _get_logits(max_size, FLAGS.network, loader.X_test, FLAGS.network.unmasked(), model["theta"])
    _, logits_theta_star = attack_network(loader.X_test, model["theta"], logits, FLAGS.network, FLAGS, jax_random.PRNGKey(onp.random.randint(1e15)))
    attacked_test_acc = get_batched_accuracy(loader.y_test, logits_theta_star)
    return attacked_test_acc

def _get_acc(model, theta, data_dir, ATTACK, mode):
    """ Returns (test_acc, attacked_test_acc, loss_over_time, loss) where attacked_test_acc and loss_over_time is None if ATTACK=False  """
    class Namespace:
        def __init__(self,d):
            self.__dict__.update(d)
    FLAGS = Namespace(model)
    loader, set_size = get_loader(FLAGS, data_dir)
    if(mode == "train"):
        X = loader.X_train; y = loader.y_train
    elif(mode == "val"):
        X = loader.X_val; y = loader.y_val
    elif(mode == "test"):
        X = loader.X_test; y = loader.y_test
    else:
        print(f"Unknown mode {mode}"); raise Exception
    return _get_acc_batch(X, y, theta, FLAGS, ATTACK)

def _get_acc_batch(X, y, theta, FLAGS, ATTACK):
    max_size = 1000
    dropout_mask = FLAGS.network.unmasked()
    logits = _get_logits(max_size, FLAGS.network, X, dropout_mask, theta)
    loss = categorical_cross_entropy(y, logits)
    acc = onp.float64(get_batched_accuracy(y, logits))
    attacked_acc = loss_over_time = None
    if(ATTACK):
        loss_over_time, logits_theta_star = attack_network(X, theta, logits, FLAGS.network, FLAGS, jax_random.PRNGKey(onp.random.randint(1e15)))
        attacked_acc = onp.float64(get_batched_accuracy(y, logits_theta_star))
        loss_over_time = list(onp.array(loss_over_time, dtype=onp.float64))
    return acc, attacked_acc, loss_over_time, onp.float64(loss)

def get_train_acc(model, theta, data_dir, ATTACK=False):
    return _get_acc(model, theta, data_dir, ATTACK, mode="train")

def get_val_acc(model, theta, data_dir, ATTACK=False):
    return _get_acc(model, theta, data_dir, ATTACK, mode="val")

def get_test_acc(model, theta, data_dir, ATTACK=False):
    return _get_acc(model, theta, data_dir, ATTACK, mode="test")

@cachable(dependencies = ["model:{architecture}_session_id", "model:architecture", "n_attack_steps", "attack_size_mismatch", "attack_size_constant", "initial_std_mismatch", "initial_std_constant"])
def min_whole_attacked_test_acc(num, model, data_dir, n_attack_steps, attack_size_mismatch, attack_size_constant, initial_std_mismatch, initial_std_constant):
    with ThreadPoolExecutor(max_workers=5) as executor:
        parallel_results = []
        futures = [executor.submit(get_whole_attacked_test_acc, model, data_dir, n_attack_steps, attack_size_mismatch, attack_size_constant, initial_std_mismatch, initial_std_constant) for i in range(num)]
        for future in as_completed(futures):
            result = future.result()
            parallel_results.append(result)
    print("Done")
    min_attacked_acc = onp.min(onp.array(parallel_results))
    return min_attacked_acc

@cachable(dependencies= ["model:{architecture}_session_id", "bits", "model:architecture"])
def get_quantized_acc(bits, model, data_dir):
    theta_star = {}
    theta = model["theta"]
    for key in theta:
        theta_star[key] = quantise(theta[key], bits)
    test_acc, _ = get_test_acc(model, theta_star, data_dir, ATTACK=False)
    return test_acc

@cachable(dependencies = ["model:{architecture}_session_id", "n_iterations", "mm_level", "model:architecture"])
def get_mismatch_list(n_iterations, model, mm_level, data_dir):
    l = []
    for i in range(n_iterations):
        l.append(get_mismatch_data(model,mm_level, data_dir, "test"))
        print(i,"/",n_iterations,flush=True)
    return l

@cachable(dependencies = ["model:{architecture}_session_id", "n_iterations", "surface_dist", "model:architecture"])
def get_surface_mean(n_iterations, model, surface_dist, data_dir):
    l = []
    for _ in range(n_iterations):
        l.append(get_surface_data(model,surface_dist, data_dir, False))
    return onp.mean(l)

@cachable(dependencies = ["model:{architecture}_session_id", "model:architecture"])
def get_attacked_test_acc(model, data_dir):
    _, attacked_test_acc = get_test_acc(model, model["theta"], data_dir, ATTACK=True)
    return get_attacked_test_acc

######################## Plotting ########################

def remove_all_splines(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.spines['left'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)

def remove_all_but_left_btm(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    
def remove_top_right_splines(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    
def remove_all_ticks(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.set_yticks([])
        a.set_xticks([])

def get_axes_constant_attack_figure(fig, gridspec,betas):
    top_axes = [fig.add_subplot(gridspec[0,i]) for i in range(3)]
    remove_all_splines(top_axes[1:])
    remove_all_ticks(top_axes[1:])
    twin_axes = plt.twinx(top_axes[0])
    remove_all_ticks(twin_axes)
    remove_all_splines(twin_axes)
    twin_axes.set_ylabel(str(betas[0])+r"[$\beta$]")
    top_axes[1].yaxis.set_label_position("right")
    top_axes[2].yaxis.set_label_position("right")
    top_axes[0].set_title(r"$\textbf{a}$", x=0, fontdict={'fontsize':15})
    top_axes[0].spines['right'].set_visible(False)
    top_axes[0].spines['top'].set_visible(False)
    top_axes[0].set_ylabel("Training accuracy")
    bottom_axes = fig.add_subplot(gridspec[1,:])
    bottom_axes.set_title(r"$\textbf{b}$", x=0, fontdict={'fontsize':15})
    bottom_axes.spines['right'].set_visible(False)
    bottom_axes.spines['top'].set_visible(False)
    bottom_axes.set_xticks(onp.linspace(0,len(betas)-1, len(betas)))
    bottom_axes.set_xticklabels(betas)
    bottom_axes.set_ylim([0.2,1.0])
    bottom_axes.set_xlim([-0.1,3.1])
    bottom_axes.set_xlabel(r"$\beta$")
    bottom_axes.set_ylabel("Test accuracy")
    return top_axes,bottom_axes

def get_axes_method_figure(fig, gridspec):
    axes_top = fig.add_subplot(gridspec[0,:])
    remove_all_splines(axes_top)
    remove_all_ticks(axes_top)
    axes_top.set_title(r"\textbf{a}", x=0, fontdict={"fontsize":13})
    
    axes_middle = [fig.add_subplot(gridspec[1,:3]),fig.add_subplot(gridspec[1,3:])]
    remove_top_right_splines(axes_middle)
    axes_middle[-1].spines['bottom'].set_visible(False)
    axes_middle[-1].set_xticks([])
    axes_middle[0].set_title(r"\textbf{b}", x=0, fontdict={"fontsize":13})
    axes_middle[0].set_ylabel(r"$\mathcal{L}_\textnormal{KL}$")

    axes_bottom = [fig.add_subplot(gridspec[2,i*2:(i+1)*2]) for i in range(3)]
    axes_bottom[1].set_xticks([])
    axes_bottom[-1].set_xticks([])
    remove_top_right_splines(axes_bottom[1:])
    axes_bottom[1].spines['bottom'].set_visible(False)
    axes_bottom[-1].spines['bottom'].set_visible(False)
    remove_top_right_splines(axes_bottom[0])
    axes_bottom[0].set_title(r"\textbf{c}", x=0, fontdict={"fontsize":13})
    axes_bottom[0].set_ylabel(r"$\mathcal{L}_\textnormal{KL}$")
    axes_bottom[0].set_xticks([0,10])
    axes_bottom[0].set_xticklabels(["0",r"$N_\textnormal{attack}$"])
    axes_bottom[1].ticklabel_format(style="sci", scilimits=(0,0))
    axes_bottom[-1].ticklabel_format(style="sci", scilimits=(0,0))
    return axes_top,axes_middle,axes_bottom

def get_axes_main_figure(fig, gridspec, N_cols, N_rows, id, mismatch_levels, btm_ylims):
    # - Get the top axes
    if(id == "speech"):
        top_axes = [fig.add_subplot(gridspec[:int(N_rows/2),:int(N_cols/2)]),fig.add_subplot(gridspec[:int(N_rows/2),int(N_cols/2):2*int(N_cols/2)])]
        for ax in top_axes:
            remove_all_splines(ax)
    elif(id == "ecg"):
        r = int(N_rows/6)
        top_axes = [fig.add_subplot(gridspec[:r,:N_cols]),fig.add_subplot(gridspec[r:2*r,:N_cols]),fig.add_subplot(gridspec[2*r:3*r,:N_cols])]
        for ax in top_axes:
            remove_all_splines(ax)
    elif(id == "cnn"):
        r = int(N_rows/8)
        inner_grid = gridspec[0:int(N_rows/2),:].subgridspec(4, N_cols, wspace=0.05, hspace=0.05)
        top_axes = [fig.add_subplot(inner_grid[i,j]) for i in range(4) for j in range(N_cols)]
        top_axes[10].text(x=-10, y=10, s="Robust", fontdict={"rotation": 90})
        top_axes[30].text(x=-10, y=10, s="Normal", fontdict={"rotation": 90})
    # - Bottom axes
    r = int(N_cols/2)
    btm_axes = [fig.add_subplot(gridspec[int(N_rows/2):,int(i*2):int((i+1)*2)]) for i in range(r)]
    for i,ax in enumerate(btm_axes):
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlabel(mismatch_levels[i])
        ax.set_xticks([])
        ax.set_ylim(btm_ylims)
        if(i>0): ax.set_yticks([])
    for i,ax in enumerate(top_axes):
        ax.set(xticks=[], yticks=[])
    axes = {"top": top_axes, "btm": btm_axes}
    return axes

def get_axes_hessian(fig, architectures):
    Nc = 9
    gridspec = fig.add_gridspec(2, Nc, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
    axes_top = [fig.add_subplot(gridspec[0,:int(Nc / 3)]), fig.add_subplot(gridspec[0,int(Nc / 3):2*int(Nc / 3)]), fig.add_subplot(gridspec[0,2*int(Nc / 3):])]
    axes_btm = [fig.add_subplot(gridspec[1,:int(Nc / 3)]), fig.add_subplot(gridspec[1,int(Nc / 3):2*int(Nc / 3)]), fig.add_subplot(gridspec[1,2*int(Nc / 3):])]
    for ax in (axes_top+axes_btm):
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
    axes_top[0].set_ylabel(r"Normal")
    axes_btm[0].set_ylabel(r"Robust")
    for i,ax in enumerate(axes_top):
        ax.set_title(architectures[i])
    return axes_top, axes_btm

def get_axes_weight_scale_exp(fig, N_rows, N_cols):
    gridspec = fig.add_gridspec(N_rows, N_cols, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
    axes = [fig.add_subplot(gridspec[i,j]) for i in range(N_rows) for j in range(N_cols)]
    remove_all_but_left_btm(axes)
    return axes

def get_axes_worst_case(fig, N_rows, N_cols, attack_sizes):
    gridspec = fig.add_gridspec(N_rows, N_cols, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
    axes = [fig.add_subplot(gridspec[i,j]) for i in range(N_rows) for j in range(N_cols)]
    remove_all_but_left_btm(axes)
    for ax in axes:
        # ax.set_xlim([-0.1,len(attack_sizes)])
        ax.set_xticks(range(len(attack_sizes)))
        ax.set_xticklabels(attack_sizes)
        ax.set_xlabel(r"$\epsilon$")
    return axes

def plot_mm_distributions(axes, data, labels, legend=False):
    N = len(labels)
    colors = ["#4c84e6","#fc033d","#03fc35"]
    test_accs = [onp.mean(el) for el in data[0]]
    for i in range(N):
        print(f"Test acc {labels[i]} is {test_accs[i]}")

    for i,el in enumerate(data[1:]):
        x = []
        y = []
        hue = []
        for j in range(len(el)):
            x = onp.hstack([x, [j] * len(el[j])])
            y = onp.hstack([y, el[j]])
            hue = onp.hstack([hue, [j] * len(el[j])])
        sns.violinplot(ax = axes[i],
            x = x,
            y = y,
            split = False,
            hue = hue,
            inner = 'quartile', cut=0,
            scale = "width", palette = colors, saturation=1.0, linewidth=0.5)
        axes[i].set_xticks([])
        if(i < len(data[1:])-1 or not legend):
            axes[i].get_legend().remove()

    if(legend):
        a = axes[i]
        lines = [Line2D([0,0],[0,0], color=c, lw=3.) for c in colors]
        a.legend(lines, labels, frameon=False, loc=3, prop={'size': 7})
        
def get_data(id):
    if(id == "speech"):
        d = speech_lsnn.default_hyperparameters()
        d["desired_samples"] = int(d["sample_rate"] * d["clip_duration_ms"] / 1000)
        d["window_size_samples"] = int(d["sample_rate"] * d["window_size_ms"] / 1000)
        d["length_minus_window"] = (d["desired_samples"] - d["window_size_samples"])
        d["fingerprint_width"] = d["feature_bin_count"] 
        d["window_stride_samples"] = int(d["sample_rate"] * d["window_stride_ms"] / 1000)
        d["spectrogram_length"] = 1 + int(d["length_minus_window"] / d["window_stride_samples"])
        d["fingerprint_size"] = d["feature_bin_count"] * d["spectrogram_length"]
        audio_processor = input_data.AudioProcessor(
        data_url=d["data_url"], data_dir="TensorCommands/speech_dataset",
        silence_percentage=d["silence_percentage"], unknown_percentage=d["unknown_percentage"],
        wanted_words=d["wanted_words"].split(','), validation_percentage=d["validation_percentage"],
        testing_percentage=d["testing_percentage"], 
        n_thr_spikes=d["n_thr_spikes"], n_repeat=d["in_repeat"], seed=d["seed"]
        )
        train_fingerprints, train_ground_truth = audio_processor.get_data(100, 0, d, d["background_frequency"],d["background_volume"], int((d["time_shift_ms"] * d["sample_rate"]) / 1000), 'training')
        X = train_fingerprints.numpy()
        y = train_ground_truth.numpy()
        input_frequency_size = d['fingerprint_width']
        input_channels = onp.max(onp.array([1, 2*d['n_thr_spikes'] - 1]))
        input_time_size = d['spectrogram_length'] * d['in_repeat']
        X = onp.reshape(X, (-1, input_time_size, input_frequency_size * input_channels))
        X = X[[onp.where(y==2)[0][0],onp.where(y==3)[0][0]],:,:].transpose((0,2,1)) # - Yes and no
        y = [2,3]
    elif(id == "ecg"):
        d = ecg_lsnn.default_hyperparameters()
        ecg_processor = ECGDataLoader(path="ECG/ecg_recordings", batch_size=100)
        _,y,X = ecg_processor.get_sequence()
        y = onp.array(y)
    elif(id == "cnn"):
        d = cnn.default_hyperparameters()
        data_loader = CNNDataLoader(100)
        classes = {0 : "Shirt", 1 : "Trouser", 4: "Jacket", 5: "Shoe"}
        _, X, y = data_loader.get_n_images(5, list(classes.keys()))
        X = onp.squeeze(X)
    return X,y

def plot_images(axes, X, y):
    for i,ax in enumerate(axes[:int(len(axes)/2)]):
        ax.imshow(X[i], cmap="binary")
        if(onp.random.rand() > 0.1):
            plt.setp(ax.spines.values(), color="g", linewidth=1)
        else:
            plt.setp(ax.spines.values(), color="r", linewidth=1)
    for i,ax in enumerate(axes[int(len(axes)/2):]):
        ax.imshow(X[i], cmap="binary")
        if(onp.random.rand() > 0.5):
            plt.setp(ax.spines.values(), color="g", linewidth=1)
        else:
            plt.setp(ax.spines.values(), color="r", linewidth=1)
    axes[0].set_title(r"$\textbf{c}$", x=0, fontdict={'fontsize':15})

def plot_spectograms(axes, X, y):
    x = onp.linspace(0,1,X.shape[2])
    y = onp.linspace(0,1,X.shape[1])
    axes[0].pcolormesh(x,y, X[0])
    axes[1].pcolormesh(x,y, X[1])
    axes[0].text(x=0.8, y=0.8, s="Yes", fontdict={"size": 15})
    axes[1].text(x=0.8, y=0.8, s="No", fontdict={"size": 15})
    axes[0].set_title(r"$\textbf{a}$", x=0, fontdict={'fontsize':15})

def get_y(y, p):
    y_hat = copy(y)
    for i in range(len(y_hat)):
        if(onp.random.rand() > p):
            y_hat[i] = onp.random.choice(onp.unique(y))
    return y_hat

def plot_ecg(axes, X, y):
    channel_colors = ["#34495e", "#2ecc71"]
    class_colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c"]
    def plt_ax(ax, pred, label):
        ax.set_ylabel(label, fontdict={"size": 7})
        ax.plot(X[:,0], color=channel_colors[0])
        ax.plot(X[:,1], color=channel_colors[1])
        for idx,y in enumerate(pred.tolist()):
            ax.axvspan(idx*100, idx*100+100, facecolor=class_colors[int(y)], alpha=0.4)
    plt_ax(axes[0], y, label="Groundtruth")
    plt_ax(axes[1], get_y(y,0.6), label="Normal")
    plt_ax(axes[2], get_y(y,0.9), label="Robust")
    axes[0].set_title(r"$\textbf{b}$", x=0, fontdict={'fontsize':15})
