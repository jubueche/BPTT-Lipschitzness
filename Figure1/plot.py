import numpy as np
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.markersize'] = 4.0
matplotlib.rcParams['image.cmap']='RdBu'
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

# - Load the data
data = {"syn_tc": [], "mem_tc": [], "weights": []}
tc_ms_syn = [10,30,55]
weights = [1,0.5,0.75]
# - Synaptic TC's
bp = os.path.dirname(__file__)
for weight,tc in zip(weights,tc_ms_syn):
    data["syn_tc"].append(np.load(os.path.join(bp,f'Resources/syn_exc_tau_{tc}_ms.npy')))
    data["weights"].append(np.load(os.path.join(bp,f'Resources/syn_exc_weight_{weight}.npy')))
tau_mems = np.load(os.path.join(bp,'Resources/vals_per_tau_bias_c0c1_0.3_0.45.npy'))
tau_mems = [tm[np.invert(np.isnan(tm))] for tm in tau_mems]
tau_mems = [tm[tm < 100] for tm in tau_mems]
tau_mem_means = [np.mean(tm) for tm in tau_mems]
data["mem_tc"].extend([tau_mems[5],tau_mems[8],tau_mems[13]][::-1])

fig = plt.figure(figsize=(5,4),constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax11 = fig.add_subplot(gs[0,0])
ax12 = fig.add_subplot(gs[0,1])
ax21 = fig.add_subplot(gs[1,0])
ax22 = fig.add_subplot(gs[1,1])
colors = sns.color_palette(["#dbd632", "#dbae32", "#db6232"]).as_hex()

def plot_dist(ax, tcs, x_label, title, is_ms=True):
    tcs = [tc[np.invert(np.isnan(tc))] for tc in tcs]
    tcs = [tc[tc < 100] for tc in tcs]
    for idx,tc in enumerate(tcs):
        ax.hist(tc, bins=10, color=colors[idx], density=True, alpha=0.3)
        mu, std = stats.norm.fit(tc)
        xmin = min(tc) ; xmax = max(tc)
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, color="k", linewidth=2.0)
        ax.plot([mu,mu], [0,stats.norm.pdf(mu, mu, std)], color="k", linestyle="dashed" ,linewidth=0.5)
        ax.plot([mu+std,mu+std], [0,stats.norm.pdf(mu+std, mu, std)], color="k", linestyle="dashed" ,linewidth=0.5)
        ax.plot([mu-std,mu-std], [0,stats.norm.pdf(mu-std, mu, std)], color="k", linestyle="dashed" ,linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])

    ax.text(x = 0, y = max(ax.get_ylim()), s=title)
    ax.set_xticks([np.mean(tc) for tc in tcs])
    if(is_ms):
        ax.set_xticklabels([("%d" % (1000*np.mean(tc))) for tc in tcs])
    else:
        ax.set_xticklabels([("%d" % (np.mean(tc))) for tc in tcs])
    if(not x_label is None):
        ax.set_xlabel(x_label)
    
    ax.set_xlim([-0.001,max(ax.get_xlim())])

plot_dist(ax11, data["syn_tc"],x_label=r"$\tau_\textnormal{syn}$ [ms]",title=r"$\textbf{a}$")
plot_dist(ax12, data["mem_tc"], x_label=r"$\tau_\textnormal{mem}$ [ms]",title=r"$\textbf{b}$")
plot_dist(ax21, data["mem_tc"], x_label=r"$W$",title=r"$\textbf{c}$", is_ms=False)

# for tc in tau_mems:
#     ax22.hist(tc, bins=10, density=True, alpha=0.3)
#     mu, std = stats.norm.fit(tc)
#     xmin = min(tc) ; xmax = max(tc)
#     x = np.linspace(xmin, xmax, 100)
#     p = stats.norm.pdf(x, mu, std)
#     ax22.plot(x, p, color="k", linewidth=2, linestyle="dashed")

def scatter_mm(ax, tcs, color, label, title=None):
    fitted_std = []; fitted_means = []; ms = []
    for tc in tcs:
        mu, std = stats.norm.fit(tc)
        fitted_std.append(std)
        fitted_means.append(mu)
        ms.append(mu / std)
    m,b,_,_,_ = stats.linregress(fitted_means,fitted_std)
    x = np.linspace(min(fitted_means),max(fitted_means),100)
    y = m*x + b 
    ax22.scatter(fitted_means, fitted_std, c=color, label=label, alpha=0.4)
    ax22.plot(x,y, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    if(title is not None):
        ax.text(x = 0, y = max(ax.get_ylim()), s=title)
    ax.set_xlabel(r"$\tau$ [ms]")
    
scatter_mm(ax22, tau_mems[4:], color="#3262db", label=r"$\tau_\textnormal{mem}$", title=r"$\textbf{d}$")
scatter_mm(ax22, data["syn_tc"], color="#db3262", label=r"$\tau_\textnormal{syn}$")
ax22.legend(frameon=False, loc=0, fontsize=7)

plt.savefig(os.path.join(bp,"../Figures/figure1.png"))
plt.show()