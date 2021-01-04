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
import numpy as onp
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['image.cmap']='RdBu'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import seaborn as sns
from architectures import standard_defaults
from TensorCommands import input_data
from ECG.ecg_data_loader import ECGDataLoader
from CNN.import_data import CNNDataLoader
from copy import copy
from loss_jax import attack_network
from datajuicer import cachable, get
from architectures import speech_lsnn, ecg_lsnn, cnn

@cachable(dependencies = ["model:{architecture}_session_id", "mm_level", "model:architecture"])
def get_mismatch_list(n_iterations, model, mm_level, data_dir, batch_size=None):
    if(batch_size is not None):
        model["batch_size"] = batch_size
    l = []
    for i in range(n_iterations):
        l.append(get_mismatch_data(model,mm_level, data_dir, False))
        print(i,"/",n_iterations)
    return l

@cachable(dependencies = ["model:{architecture}_session_id", "n_iterations", "surface_dist", "model:architecture"])
def get_surface_mean(n_iterations, model, surface_dist, data_dir):
    l = []
    for _ in range(n_iterations):
        l.append(get_surface_data(model,surface_dist, data_dir, False))
    return onp.mean(l)

@cachable(dependencies = ["model:{architecture}_session_id", "model:architecture"])
def get_attacked_test_acc(model, data_dir):
    return get_test_acc(model, model["theta"], data_dir, ATTACK=True)

def get_mismatch_data(model, mm_level, data_dir, ATTACK=False):
    theta_star = {}
    theta = model["theta"]
    for key in theta:
        theta_star[key] = theta[key] * (1 + mm_level * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    return get_test_acc(model, theta_star, data_dir, ATTACK)

def get_surface_data(model, surface_dist, data_dir, ATTACK=False):
    theta_star = {}
    theta = model["theta"]
    for key in theta:
        theta_star[key] = theta[key] + surface_dist * onp.sign(onp.random.uniform(low=-1, high=1, size=theta[key].shape))
    return get_test_acc(model, theta_star, data_dir, ATTACK)

def get_test_acc(model, theta_star, data_dir, ATTACK=False):
    class Namespace:
        def __init__(self,d):
            self.__dict__.update(d)
    FLAGS = Namespace(model)
    if FLAGS.architecture=="speech_lsnn":
        audio_processor = input_data.AudioProcessor(
            data_url=FLAGS.data_url, data_dir=data_dir,
            silence_percentage=FLAGS.silence_percentage, unknown_percentage=FLAGS.unknown_percentage,
            wanted_words=FLAGS.wanted_words.split(','), validation_percentage=FLAGS.validation_percentage,
            testing_percentage=FLAGS.testing_percentage, 
            n_thr_spikes=FLAGS.n_thr_spikes, n_repeat=FLAGS.in_repeat
        )
        set_size = audio_processor.set_size('testing')
    elif FLAGS.architecture=="ecg_lsnn":
        ecg_processor = ECGDataLoader(path=data_dir, batch_size=FLAGS.batch_size)
        set_size = ecg_processor.N_test
    elif FLAGS.architecture=="cnn":
        cnn_data_loader = CNNDataLoader(FLAGS.batch_size, FLAGS.data_dir)
        set_size = cnn_data_loader.N_test
    def get_X_y(i):
        if FLAGS.architecture=="speech_lsnn":
            validation_fingerprints, validation_ground_truth = (audio_processor.get_data(FLAGS.batch_size, i, FLAGS.network.model_settings ,0.0, 0.0, 0.0, 'testing'))
            X = validation_fingerprints.numpy()
            y = validation_ground_truth.numpy()
            return X, y
        elif FLAGS.architecture=="ecg_lsnn":
            X,y = ecg_processor.get_batch("test")
            return onp.array(X), onp.array(y)
        elif FLAGS.architecture=="cnn":
            X,y= cnn_data_loader.get_batch("test")
            return onp.array(X), onp.array(y)
    total_accuracy = 0.0
    if(ATTACK):
        attacked_total_accuracy = 0.0
    for i in range(0, set_size, FLAGS.batch_size):
        X, y = get_X_y(i)
        logits, _ = FLAGS.network.call(X, jnp.ones(shape=(1,FLAGS.network.units)), **theta_star)
        if(ATTACK):
            _, logits_theta_star = attack_network(X, theta_star, logits, FLAGS.network, FLAGS, FLAGS.network._rng_key)
            FLAGS.network._rng_key, _ = jax_random.split(FLAGS.network._rng_key)
            attacked_batched_validation_acc = get_batched_accuracy(y, logits_theta_star)
            attacked_total_accuracy += (attacked_batched_validation_acc * FLAGS.batch_size) / set_size
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * FLAGS.batch_size) / set_size
    if(ATTACK):
        return onp.float64(attacked_total_accuracy)
    else:
        return onp.float64(total_accuracy)

def get_batched_accuracy(y, logits):
    y =onp.array(y)
    logits = onp.array(logits)
    if(y.ndim > 1):
        y = onp.argmax(y, axis=1)
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

def remove_all_splines(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.spines['left'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)
    
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

def plot_mm_distributions(axes, data):
    test_acc_normal, test_acc_rob = data[0]
    print("Normal test acc.", onp.mean(test_acc_normal),"Robust test acc.", onp.mean(test_acc_rob))
    test_acc_normal = onp.mean(test_acc_normal)
    test_acc_rob = onp.mean(test_acc_rob)
    for i,(norm,rob) in enumerate(data[1:]):
        x = [0]*(len(norm)+len(rob))
        y = onp.hstack((norm,rob))
        hue = onp.hstack(([0] * len(norm), [1] * len(rob)))
        sns.violinplot(ax = axes[i],
            x = x,
            y = y,
            split = True,
            hue = hue,
            inner = 'quartile', cut=0,
            scale = "width", palette = ["#4c84e6","#6ea3ff"], saturation=1.0, linewidth=1.0)
        axes[i].set_xticks([])
        axes[i].get_legend().remove()
        # - Plot horizontal lines
        axes[i].axhline(y=onp.mean(norm)/test_acc_normal, color="r")
        axes[i].axhline(y=onp.mean(rob)/test_acc_rob, color="g")

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
        _,y,X = ecg_processor.get_sequence(N_per_class=10, path="ECG/ecg_recordings")
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