import numpy as np
import pandas as pd
import seaborn as sns
import ujson as json
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker


def plot_experiment_a():

    SPLIT = True

    # - Load the data
    experiment_a_path = "Experiments/experiment_a.json"
    with open(experiment_a_path, "r") as f:
        experiment_a_data = json.load(f)

    mismatch_labels = list(experiment_a_data["normal"].keys())[1:]

    fig = plt.figure(figsize=(7.14,3.91))
    if(SPLIT):
        outer = gridspec.GridSpec(1, 1, figure=fig, wspace=0.2)
    else:
        outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)

    c_range = np.linspace(0.0,1.0,len(mismatch_labels))
    colors_mismatch = [(0.9176, 0.8862, i, 1.0) for i in c_range]
    color_baseline = (0.9176, 0.3862, 0.8, 1.0)
    label_mode = ["Normal", "Robust"]
    if(SPLIT):
        modes = ["normal"]
    else:
        modes = ["normal", "robust"]

    for idx_mode,mode in enumerate(modes):

        inner = gridspec.GridSpecFromSubplotSpec(1, len(mismatch_labels),
                        subplot_spec=outer[idx_mode], wspace=0.0)

        for idx_std, mismatch_std in enumerate(mismatch_labels):

            ax = plt.Subplot(fig, inner[idx_std])

            if(SPLIT):
                x = [idx_std] * (len(experiment_a_data["normal"][mismatch_std]) + len(experiment_a_data["robust"][mismatch_std]))
                y = np.hstack((experiment_a_data["normal"][mismatch_std],experiment_a_data["robust"][mismatch_std]))
                hue = np.hstack(([0] * len(experiment_a_data["normal"][mismatch_std]), [1] * len(experiment_a_data["robust"][mismatch_std])))
            else:
                x = [idx_std] * len(experiment_a_data[mode][mismatch_std])
                y = experiment_a_data[mode][mismatch_std]
                hue = [0] * len(experiment_a_data[mode][mismatch_std])

            sns.violinplot(ax = ax,
                    x = x,
                    y = y,
                    split = SPLIT,
                    hue = hue,
                    inner = 'quartile', cut=0,
                    scale = "width", palette = [colors_mismatch[idx_std]], saturation=1.0, linewidth=1.0)

            plt.ylim([0.3, 1.0])
            ax.set_ylim([0.3, 1.0])
            ax.get_legend().remove()
            plt.xlabel('')
            plt.ylabel('')
            
            if (idx_mode == 0 and idx_std == 0):
                plt.ylabel('Accuracy')
            
            if (idx_mode > 0 or idx_std > 0):
                ax.set_yticks([])
                plt.axis('off')

            if(SPLIT):
                if (idx_std == int(len(mismatch_labels)/2)):
                    ax.set_title("Mismatch robustness")
            else:
                if (idx_std == int(len(mismatch_labels)/2)):
                    ax.set_title(label_mode[idx_mode])

            ax.set_xticks([])
            plt.xticks([])
            ax.set_xlim([-1, 1])

            fig.add_subplot(ax)

    custom_lines = [Line2D([0], [0], color=colors_mismatch[i], lw=4) for i in range(len(mismatch_labels))]
    legend_labels = [(f'{str(mismatch_label)}\%') for mismatch_label in mismatch_labels]
    ax.legend(custom_lines, legend_labels, frameon=False, loc=3, fontsize = 7)

    # show only the outside spines
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(False)

    plt.savefig("Figures/experiment_a.png", dpi=1200)
    plt.show(block=True)

def plot_experiment_b():
    pass
def plot_experiment_c():
    pass
def plot_experiment_d():
    pass
def plot_experiment_e():
    pass


plot_experiment_a()