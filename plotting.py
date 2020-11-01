import ujson as json
import numpy as np
import os.path as path
import matplotlib
matplotlib.rc('font', family='Sans-Serif')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
import matplotlib.pyplot as plt

fn_lipschitzness = path.join(path.dirname(path.abspath(__file__)), f"Resources/Plotting/track_dict_lipschitzness.json")
fn = path.join(path.dirname(path.abspath(__file__)), f"Resources/Plotting/track_dict_.json")
with open(fn, "r") as f:
    track_dict = json.load(f)
with open(fn_lipschitzness, "r") as f:
    track_dict_lipschitzness = json.load(f)

fig = plt.figure(figsize=(7,5))
fig.add_subplot(221)
plt.title("No Lipschitzness")
plt.plot(track_dict["training_accuracies"], label="Training acc.")
plt.plot(track_dict["attacked_training_accuracies"], label="Attacked training acc.", linestyle="-")
plt.legend()
fig.add_subplot(223)
for l in track_dict["kl_over_time"]:
    plt.plot(l, color="grey", alpha=0.3)
mean_increase = np.mean(np.asarray(track_dict["kl_over_time"]), axis=0)
plt.plot(mean_increase, color="k")

fig.add_subplot(222)
plt.title("Lipschitzness")
plt.plot(track_dict_lipschitzness["training_accuracies"], label="Training acc.")
plt.plot(track_dict_lipschitzness["attacked_training_accuracies"], label="Attacked training acc.", linestyle="-")
plt.legend()
fig.add_subplot(224)
for l in track_dict_lipschitzness["kl_over_time"]:
    plt.plot(l, color="grey", alpha=0.3)
mean_increase = np.mean(np.asarray(track_dict_lipschitzness["kl_over_time"]), axis=0)
plt.plot(mean_increase, color="k")

plt.show()