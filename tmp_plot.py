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

def get_mismatch_data(mean, N=500):
    return onp.random.normal(loc=mean, scale=0.1*mean, size=(N,))

def get_mm_levels():
    return [[0.5,0.7,0.9,1.1,1.5], [0.1,0.2,0.3,0.5,0.7], [0.5,0.7,0.9,1.1,1.5]]

def remove_all_splines(ax):
    if(not isinstance(ax, list)):
        ax = [ax]
    for a in ax:
        a.spines['left'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.spines['bottom'].set_visible(False)

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
    for i,(norm,rob) in enumerate(data):
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

def get_data(id):
    d = standard_defaults()
    if(id == "speech"):
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
        ecg_processor = ECGDataLoader(path="ECG/ecg_recordings", batch_size=100)
        _,y,X = ecg_processor.get_sequence(N_per_class=10, path="ECG/ecg_recordings")
        y = onp.array(y)
    elif(id == "cnn"):
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

def plot_main_figure():
    # - Main order is speech, ECG and CNN
    mm_levels = get_mm_levels()
    # - Per general column
    N_cols = 2*len(mm_levels[0]) # - 10
    N_rows = 24 # - Will be determined by subplot that has the most rows

    fig = plt.figure(figsize=(14, 5), constrained_layout=False)
    hs = 5
    gridspecs = [fig.add_gridspec(N_rows, N_cols, left=0.05, right=0.31, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.35, right=0.61, hspace=hs), fig.add_gridspec(N_rows, N_cols, left=0.65, right=0.98, hspace=hs)]

    axes_speech = get_axes_main_figure(fig, gridspecs[0], N_cols, N_rows, "speech", mismatch_levels=mm_levels[0], btm_ylims=[0.0,0.95])
    axes_ecg = get_axes_main_figure(fig, gridspecs[1], N_cols, N_rows, "ecg", mismatch_levels=mm_levels[1], btm_ylims=[0.0,0.95])
    axes_cnn = get_axes_main_figure(fig, gridspecs[2], N_cols, N_rows, "cnn", mismatch_levels=mm_levels[2], btm_ylims=[0.0,0.95])

    # - Fill the plots with data
    plot_mm_distributions(axes_speech["btm"], data=[(get_mismatch_data(mm/2.5),get_mismatch_data(mm/2)) for mm in mm_levels[0][::-1]])
    plot_mm_distributions(axes_ecg["btm"], data=[(get_mismatch_data(mm/1.5),get_mismatch_data(mm/1)) for mm in mm_levels[1][::-1]])
    plot_mm_distributions(axes_cnn["btm"], data=[(get_mismatch_data(mm/2.5),get_mismatch_data(mm/2)) for mm in mm_levels[2][::-1]])

    # - Get the sample data for speech
    X_speech, y_speech = get_data("speech")
    X_ecg, y_ecg = get_data("ecg")
    X_cnn, y_cnn = get_data("cnn")

    plot_images(axes_cnn["top"], X_cnn, y_cnn)
    plot_spectograms(axes_speech["top"], X_speech, y_speech)
    plot_ecg(axes_ecg["top"], X_ecg, y_ecg)

    axes_speech["btm"][0].set_ylabel("Accuracy")
    axes_speech["btm"][2].text(x = -0.5, y = -0.2, s="Mismatch level")

    plt.savefig("Resources/Figures/figure_main.png", dpi=1200)
    plt.show()

# def plot_methods():
#     fig = plt.figure(figsize=(6,6), constrained_layout=False)
#     gridspec = fig.add_gridspec(3,3, hspace=0.2)
#     ax_top = fig.add_subplot(gridspec[0,:])
#     ax_top.set(xticks=[], yticks=[])
#     remove_all_splines(ax=ax_top)
#     ax_top.set_title(r"$\textbf{a}$", x=0, fontdict={"size": 13})
#     ax_middle = [fig.add_subplot(gridspec[1,i]) for i in range(3)]
#     remove_all_splines(ax_middle[1:])

#     plt.show()


plot_main_figure()