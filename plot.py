import numpy as np
import pandas as pd
import seaborn as sns
import ujson as json
import os
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['image.cmap']='RdBu'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
import matplotlib.collections as clt
from matplotlib.cbook import boxplot_stats
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib import ticker as mticker
from scipy.stats import norm, mannwhitneyu

pp = lambda x  : ("%.4f" % x)

def plot_experiment_a(ATTACK=False):
    SPLIT = True
    # - Load the data
    if(ATTACK):
        experiment_a_path = "Experiments/experiment_a_attack.json"
    else:
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
            if(ATTACK):
                ylim = 0.35
            else:
                ylim = 0.1

            plt.ylim([ylim, 1.0])
            ax.set_ylim([ylim, 1.0])
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
    if(ATTACK):
        legend_labels = [(f'{str(float(mismatch_label))}') for mismatch_label in mismatch_labels]
    else:   
        legend_labels = [(f'{str(int(100*float(mismatch_label)))}\%') for mismatch_label in mismatch_labels]
    fig.get_axes()[0].legend(custom_lines, legend_labels, frameon=False, loc=3, fontsize = 7) 
    # show only the outside spines
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(False)

    if(ATTACK):
        plt.savefig("Figures/experiment_a_attack.png", dpi=1200)
    else:
        plt.savefig("Figures/experiment_a.png", dpi=1200)
    plt.show(block=False)

    # - Print Latex table
    print("$L$ \t Mean Acc. Normal \t Mean Acc. Robust  \t $\Delta$ Mean Acc. \t $\Delta$ Std. \t P-Value")
    dma = np.zeros((len(mismatch_labels,)))
    dstd = np.zeros((len(mismatch_labels,)))
    for idx,L in enumerate(mismatch_labels):
        acc_normal = experiment_a_data["normal"][L]
        acc_robust = experiment_a_data["robust"][L]
        mean_acc_normal = np.mean(acc_normal)
        mean_acc_robust = np.mean(acc_robust)
        std_normal = np.std(acc_normal)
        std_robust = np.std(acc_robust)
        drop_mean_acc = abs(mean_acc_robust - mean_acc_normal)
        drop_std = abs(std_robust - std_normal)
        _, p_value = mannwhitneyu(acc_normal, acc_robust)
        dma[idx] = drop_mean_acc
        dstd[idx] = 1 - (std_normal - std_robust) / std_normal
        print(f"{L} \t ${pp(mean_acc_normal)}\\pm {pp(std_normal)}$ \t ${pp(mean_acc_robust)}\\pm {pp(std_robust)}$ \t {pp(drop_mean_acc)} \t {pp(drop_std)} \t {p_value}")
        
    print(f"Min Drop Acc {pp(np.min(dma))} Max Drop Acc {pp(np.max(dma))} Min Diff Std {pp(np.min(dstd))} Max Diff Std {pp(np.max(dstd))} ")

def plot_experiment_b():
    """Outputs table format that can be pasted directly into https://www.tablesgenerator.com/ and a plot"""
    # - Load the data
    experiment_b_path = "Experiments/experiment_b.json"
    with open(experiment_b_path, "r") as f:
        experiment_b_data = json.load(f)
    num_models = len(experiment_b_data.keys())-1
    # mismatch_level = experiment_b_data["experiment_params"]["mismatch_level"]
    gaussian_eps =  experiment_b_data["experiment_params"]["gaussian_eps"]
    gaussian_attack_eps = experiment_b_data["experiment_params"]["gaussian_attack_eps"]
    x_labels = []

    print(f"N,$\\beta$ \t Test acc \t Gaussian @ {gaussian_attack_eps} \t Gaussian @ {gaussian_eps} \t Gaussian Attack @ {gaussian_attack_eps}")
    for i in range(num_models):
        beta = experiment_b_data[str(i)]["model_params"]["beta_lipschitzness"]
        n_hidden = experiment_b_data[str(i)]["model_params"]["n_hidden"]
        normal_test_acc = np.median(experiment_b_data[str(i)]["normal_test_acc"])
        median_test_acc_gaussian_with_eps_attack = np.median(experiment_b_data[str(i)]["gaussian_with_eps_attack"])
        std_test_acc_gaussian_with_eps_attack = np.std(experiment_b_data[str(i)]["gaussian_with_eps_attack"])
        median_test_acc_gaussian = np.median(experiment_b_data[str(i)]["gaussian"])
        std_test_acc_gaussian = np.std(experiment_b_data[str(i)]["gaussian"])
        median_test_acc_gaussian_attack = np.median(experiment_b_data[str(i)]["gaussian_attack"])
        std_test_acc_gaussian_attack = np.std(experiment_b_data[str(i)]["gaussian_attack"])
        print("%d/%.2f \t %.4f \t %.4f$\pm$%.4f \t %.4f$\pm$%.4f \t %.4f$\pm$%.4f" % (n_hidden,beta,normal_test_acc,median_test_acc_gaussian_with_eps_attack,std_test_acc_gaussian_with_eps_attack,median_test_acc_gaussian,std_test_acc_gaussian,median_test_acc_gaussian_attack,std_test_acc_gaussian_attack))
        x_labels.append(r"$\beta$" + f" {beta}")

    plt.figure(figsize=(7,2))
    x = np.linspace(0,num_models-1, num_models)
    y_median_gaussian_with_eps_attack = [np.median(experiment_b_data[str(i)]["gaussian_with_eps_attack"]) for i in range(num_models)]
    y_std_gaussian_with_eps_attack = [np.std(experiment_b_data[str(i)]["gaussian_with_eps_attack"]) for i in range(num_models)]

    y_median_gaussian = [np.median(experiment_b_data[str(i)]["gaussian"]) for i in range(num_models)]
    y_std_gaussian = [np.std(experiment_b_data[str(i)]["gaussian"]) for i in range(num_models)]

    y_median_gaussian_attack = [np.median(experiment_b_data[str(i)]["gaussian_attack"]) for i in range(num_models)]
    y_std_gaussian_attack = [np.std(experiment_b_data[str(i)]["gaussian_attack"]) for i in range(num_models)]

    label_gaussian_with_eps_attack = f"Gaussian {gaussian_attack_eps}" # r"$\Theta^*=\Theta (1+\epsilon X),X \sim \mathcal{N}(0,1)$"
    label_gaussian = f"Gaussian {gaussian_eps}" # r"$\Theta^* \sim \mathcal{N}(\Theta,\epsilon)$"
    label_gaussian_attack = f"Gaussian attack {gaussian_attack_eps}" # r"$\Theta^*= \max_{\Theta^* \in \mathcal{B}(\Theta,\epsilon)} \mathcal{L}(f_{\Theta}(x),f_{\Theta^*}(x))$"
    plt.errorbar(x, y_median_gaussian_with_eps_attack, y_std_gaussian_with_eps_attack, label=label_gaussian_with_eps_attack, color="C2", linestyle="dashed", marker="o", markevery=list(np.array(x,int)), capsize=3)
    plt.errorbar(x, y_median_gaussian, y_std_gaussian, label=label_gaussian, color="C3", marker="^", linestyle="dotted", markevery=list(np.array(x,int)), capsize=3)
    plt.errorbar(x,y_median_gaussian_attack, y_std_gaussian_attack, label=label_gaussian_attack, color="C4", marker="s", linestyle="solid", markevery=list(np.array(x,int)), capsize=3)
    plt.legend(frameon=False, loc=4, fontsize = 7)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(x)
    ax.set_ylabel("Accuracy")
    # ax.set_ylim([0.2,1.0])
    ax.set_xticklabels(x_labels)
    ax.margins(x=0.1)
    plt.savefig("Figures/experiment_b.png", dpi=1200)
    plt.show(block=False)


def plot_experiment_c():
    # - Load the data
    experiment_c_path = "Experiments/experiment_c.json"
    with open(experiment_c_path, "r") as f:
        experiment_c_data = json.load(f)
    num_models = len(experiment_c_data.keys())-1
    gaussian_attack_eps = experiment_c_data["experiment_params"]["gaussian_attack_eps"]
    x_labels = []
    print(f"$\\beta$ \t Test acc \t Gaussian Attack @ {gaussian_attack_eps}")
    for i in range(num_models):
        beta = experiment_c_data[str(i)]["model_params"]["beta_lipschitzness"]
        normal_test_acc = np.median(experiment_c_data[str(i)]["normal_test_acc"])
        median_test_acc_gaussian_attack = np.median(experiment_c_data[str(i)]["gaussian_attack"])
        std_test_acc_gaussian_attack = np.std(experiment_c_data[str(i)]["gaussian_attack"])
        print("%.2f \t\t %.4f \t %.4f$\pm$%.4f" % (beta,normal_test_acc,median_test_acc_gaussian_attack,std_test_acc_gaussian_attack))
        x_labels.append(r"$\beta$" + f" {beta}")


    fig = plt.figure(figsize=(7,2),constrained_layout=True)
    alpha = 0.3
    filter_length = 50
    gs = fig.add_gridspec(1, num_models)
    axes = [fig.add_subplot(gs[0,i]) for i in range(num_models)]
    for i,ax in enumerate(axes):
        y_train_normal = experiment_c_data[str(i)]["tracking_dict"]["training_accuracies"]
        y_train_attacked = experiment_c_data[str(i)]["tracking_dict"]["attacked_training_accuracies"]
        ma_y_train_normal = np.convolve(y_train_normal, np.ones(filter_length)/filter_length, mode="full")[:len(y_train_normal)]
        ma_y_train_attacked = np.convolve(y_train_attacked, np.ones(filter_length)/filter_length, mode="full")[:len(y_train_attacked)]
        x = np.linspace(0,len(y_train_attacked)-1, len(y_train_attacked))
        ax.plot(x, y_train_normal, label="Normal", color="C1", alpha=alpha)
        ax.plot(x, ma_y_train_normal, color="C1")
        ax.plot(x, y_train_attacked, color="C2", label="Attacked", alpha=alpha)
        ax.plot(x, ma_y_train_attacked, color="C2")
        ax.axhline(y = np.median(experiment_c_data[str(i)]["normal_test_acc"]), color="C4", linestyle="dotted", alpha=0.5, label="Test acc.")
        if(i == 0):
            ax.axhline(y=0.5, color="b", linestyle="dashed", alpha=0.5)
            ax.set_ylabel("Accuracy")
        elif(i == num_models-1):
            ax.legend(frameon=False, loc=5, fontsize = 8)
        if(i > 0):
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(x_labels[i])
        ax.set_ylim([0.2,1.0])
    
    
    plt.savefig("Figures/experiment_c.png", dpi=1200)
    plt.show(block=False)

def plot_experiment_d():
    pass
def plot_experiment_e():
    SHOW_EVO = True
     # - Load the data
    experiment_e_path = "Experiments/experiment_e.json"
    with open(experiment_e_path, "r") as f:
        experiment_e_data = json.load(f)
    num_models = len(experiment_e_data.keys())-1
    x_labels = []
    for i in range(num_models):
        beta = experiment_e_data["betas"][i]
        x_labels.append(r"$\beta$" + f" {beta}")

    fig = plt.figure(figsize=(7,3),constrained_layout=True)
    alpha = 0.3
    filter_length = 50
    if(SHOW_EVO):
        gs = fig.add_gridspec(2, num_models)
    else:
        gs = fig.add_gridspec(1, num_models)
    axes = [fig.add_subplot(gs[0,i]) for i in range(num_models)]
    if(SHOW_EVO):
        axes_sub = [fig.add_subplot(gs[1,i]) for i in range(num_models)]
    for i,ax in enumerate(axes):
        kl_over_time = experiment_e_data[str(i)]["tracking_dict"]["kl_over_time"]
        last_kl_over_time = [el[-1] for el in kl_over_time][0:] # - Skip artifacts if any
        ma_last_kl_over_time = np.convolve(last_kl_over_time, np.ones(filter_length)/filter_length, mode="full")[:len(last_kl_over_time)]
        x = np.linspace(0,len(last_kl_over_time)-1, len(last_kl_over_time))
        ax.plot(x, last_kl_over_time, label=r"$\mathcal{L}(f_{\Theta}(x),f_{\Theta^*}(x))$", color="C2", alpha=alpha)
        ax.plot(x, ma_last_kl_over_time, color="C2")
        ax.set_yscale("log")
        ax.set_ylim([0.001,10])
        if(i == 0):
            ax.set_ylabel("KL divergence")
        elif(i == num_models-1):
            pass
            # ax.legend(frameon=False, loc=2, fontsize = 8)
        if(i > 0):
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.minorticks_off()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(x_labels[i])
    
    # - Plot second row
    if(SHOW_EVO):
        def get_c(c_idx):
            return (c_idx,0.0,0.0,0.3)
        for i,ax in enumerate(axes_sub):
            kl_over_time = experiment_e_data[str(i)]["tracking_dict"]["kl_over_time"]
            c_range = np.linspace(0.0,1.0,len(kl_over_time))
            x = np.linspace(0,len(kl_over_time[0])-1,len(kl_over_time[0]))
            for idx,el in enumerate(kl_over_time):
                if(idx % 3 == 0):
                    ax.plot(x, el, color=get_c(c_range[idx]), linewidth=0.5)            
            ax.set_yscale("log")
            ax.set_ylim([0.001,10])
            if(i == 0):
                ax.set_ylabel("KL(t)")
                custom_lines = [Line2D([0], [0], color=get_c(0.0), lw=2),Line2D([0], [0], color=get_c(1.0), lw=2)]
                legend_labels = [r"$t=0$", r"$t=T$"]
                ax.legend(custom_lines, legend_labels, frameon=False, loc=2, fontsize = 5)
            if(i > 0):
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.minorticks_off()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)


    plt.savefig("Figures/experiment_e.png", dpi=1200)
    plt.show(block=False)

    fig = plt.figure(figsize=(7,2),constrained_layout=False)
    weight_keys = ["W_in", "W_rec", "W_out"]
    weight_labels = [r"$W_\textnormal{in}$",r"$W_\textnormal{rec}$",r"$W_\textnormal{out}$"]
    xlims = [2.0,2.0,6.0]
    axes = [fig.add_subplot(131+i) for i in range(len(weight_keys))]
    for model_idx in range(num_models):
        theta = experiment_e_data[str(model_idx)]["theta"]
        curr_beta = experiment_e_data[str(model_idx)]["model_params"]["beta_lipschitzness"]
        for idx,key in enumerate(weight_keys):
            x = np.array(theta[key]).ravel()
            axes[idx].set_title(weight_labels[idx])
            axes[idx].hist(x, bins=50, color=f"C{model_idx}", density=True, alpha=alpha)
            mu, std = norm.fit(x)
            xmin = -xlims[idx] ; xmax = xlims[idx]
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            axes[idx].plot(x, p, color=f"C{model_idx}", label=f"Beta {curr_beta}", linewidth=2)
            axes[idx].set_xlim([xmin,xmax])
            axes[idx].spines["right"].set_visible(False)
            axes[idx].spines["top"].set_visible(False)
            axes[idx].spines["left"].set_visible(False)
            axes[idx].spines["bottom"].set_visible(False)
            axes[idx].set_yticks([])
            axes[idx].set_xticks([])
    fig.get_axes()[0].legend(frameon=False, loc=2, fontsize = 7)
    plt.savefig("Figures/experiment_e_weight_distributions.png", dpi=1200)
    plt.show(block=True)

plot_experiment_a(ATTACK=False)
# plot_experiment_b()
# plot_experiment_c()
# plot_experiment_e()