import experiment_utils
import os
import input_data_eager as input_data
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


def speech_lsnn_loader(session_id):
    try:
        training_data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Resources/TrainingResults/{session_id}.train"),'r'))
    except:
        raise experiment_utils.ModelLoadError
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, f"Resources/Models/{session_id}_model.json")
    try:
        rnn, theta = RNN.load(model_path)
    except:
        raise experiment_utils.ModelLoadError
    hyperparams = rnn.model_settings
    audio_processor = input_data.AudioProcessor(
        hyperparams["data_url"], hyperparams["data_dir"],
        hyperparams["silence_percentage"], hyperparams["unknown_percentage"],
        hyperparams["wanted_words"].split(','), hyperparams["validation_percentage"],
        hyperparams["testing_percentage"],
        hyperparams["n_thr_spikes"], hyperparams["in_repeat"], hyperparams["seed"]
    )
    epochs_list = list(map(int, hyperparams["n_epochs"].split(',')))
    n_steps = sum([math.ceil(epochs * audio_processor.set_size("training")/hyperparams["batch_size"]) for epochs in epochs_list])
    if not len(training_data["training_accuracy"]) == int(n_steps/10): raise experiment_utils.ModelLoadError
    return rnn, theta, training_data, audio_processor

def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

def test_accuracy_after_mismatch_attack_speech_lsnn(speech_lsnn, mm_std):
    rnn, theta, training_data, audio_processor = speech_lsnn
    hyperparams = rnn.model_settings
    theta_star = {}
    for key in theta.keys():
        theta_star[key] = theta[key] * (1 + mm_std * onp.random.normal(loc=0.0, scale=1.0, size=theta[key].shape))
    
    set_size = audio_processor.set_size('testing')
    total_accuracy = 0.0

    for i in range(0, set_size, hyperparams["batch_size"]):
        validation_fingerprints, validation_ground_truth = (
            audio_processor.get_data(hyperparams["batch_size"], i, rnn.model_settings ,0.0, 0.0, 0.0, 'testing'))
        X = validation_fingerprints.numpy()
        y = validation_ground_truth.numpy()
        logits, _ = rnn.call(X, jnp.ones(shape=(1,rnn.units)), **theta_star)
        batched_test_acc = get_batched_accuracy(y, logits)
        total_accuracy += (batched_test_acc * hyperparams["batch_size"]) / set_size
    
    return onp.float64(total_accuracy)

def visualize_exp_a(experiment_results):
    matplotlib.rc('font', family='Sans-Serif')
    matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['lines.linewidth'] = 0.5
    matplotlib.rcParams['lines.markersize'] = 4.0
    matplotlib.rcParams['image.cmap']='RdBu'
    matplotlib.rcParams['axes.xmargin'] = 0


    mismatch_labels = [0.5,0.7,0.9,1.1,1.5]
    fig = plt.figure(figsize=(7.14,3.91))
    outer = gridspec.GridSpec(1, 1, figure=fig, wspace=0.2)
    c_range = onp.linspace(0.0,1.0,len(mismatch_labels))
    colors_mismatch = [(0.9176, 0.8862, i, 1.0) for i in c_range]
    label_mode = ["Normal", "Robust"]
    
    inner = gridspec.GridSpecFromSubplotSpec(1, len(mismatch_labels),
                    subplot_spec=outer[0], wspace=0.0)
    
    for idx_std, mismatch_std in enumerate(mismatch_labels):
        ax = plt.Subplot(fig, inner[idx_std])
        normal_results = experiment_utils.query(experiment_results, "Test_Acc_MM_"+str(float(mismatch_std)), {"beta_robustness":0.0})
        robust_results = experiment_utils.query(experiment_results, "Test_Acc_MM_"+str(float(mismatch_std)), {"beta_robustness":1.0})
        x = [idx_std] * (len(normal_results) + len(robust_results))
        y = onp.hstack((normal_results,robust_results))
        hue = onp.hstack(([0] * len(normal_results), [1] * len(robust_results)))
        sns.violinplot(ax = ax,
                x = x,
                y = y,
                split = True,
                hue = hue,
                inner = 'quartile', cut=0,
                scale = "width", palette = [colors_mismatch[idx_std]], saturation=1.0, linewidth=1.0)
        ylim = 0.1

        plt.ylim([ylim, 1.0])
        ax.set_ylim([ylim, 1.0])
        ax.get_legend().remove()
        plt.xlabel('')
        plt.ylabel('')
        plt.ylabel('Accuracy')
        if (idx_std == int(len(mismatch_labels)/2)):
                ax.set_title("Mismatch robustness")
        ax.set_xticks([])
        plt.xticks([])
        ax.set_xlim([-1, 1])
        fig.add_subplot(ax)
    
    custom_lines = [Line2D([0], [0], color=colors_mismatch[i], lw=4) for i in range(len(mismatch_labels))]
    legend_labels = [(f'{str(int(100*float(mismatch_label)))}\%') for mismatch_label in mismatch_labels]
    fig.get_axes()[0].legend(custom_lines, legend_labels, frameon=False, loc=3, fontsize = 7) 
    # show only the outside spines
    for ax in fig.get_axes():
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(ax.is_first_col())
        ax.spines['right'].set_visible(False)

    
    plt.savefig("Figures/experiment_a.png", dpi=1200)
    plt.show(block=False)

def train_local(n_threads=1):
    experiment_utils.run_grids("python $$MODEL_PATH$$ $$ARGS$$",n_threads=n_threads)

def train_leonhard():
    experiment_utils.run_grids("leonhard", "bsub -o ../logs/$$SESSION_ID$$ -W $$TIME_ESTIMATE$$ -n $$PROCESSORS_ESTIMATE$$ -R \"rusage[mem=$$MEMORY_ESTIMATE$$]\" \"python3 $$MODEL_PATH$$ $$ARGS$$\"")

def run():
    experiment_utils.compute_metrics()
    experiment_utils.visualize_all()

def test():
    speech_lsnn = experiment_utils.Architecture.load('Resources/Architectures/speech_lsnn.arch')
    g = experiment_utils.ModelGrid('test',speech_lsnn,{"n_epochs":'1',"learning_rate":'0.001','n_hidden':32, 'eval_step_interval':20})
    experiment_utils.run_grids("python $$MODEL_PATH$$ $$ARGS$$",grids=[g])

#if __name__ == '__main__':
def setup():
    speech_lsnn = experiment_utils.Architecture("speech_lsnn","main_speech_lsnn.py",speech_lsnn_loader)

    speech_lsnn.add_hyperparameter('clip_duration_ms', t=int, default=1000, help='Expected duration in milliseconds of the wavs')
    speech_lsnn.add_hyperparameter('window_size_ms', t=float, default=30.0, help='How long each spectrogram timeslice is.')
    speech_lsnn.add_hyperparameter('window_stride_ms', t=float, default=10.0, help='How far to move in time between spectogram timeslices.')
    speech_lsnn.add_hyperparameter('feature_bin_count', t=int, default=40, help='How many bins to use for the MFCC / FBANK fingerprint')
    speech_lsnn.add_hyperparameter('sample_rate', t=int, default=16000, help='Expected sample rate of the wavs')
    speech_lsnn.add_hyperparameter('in_repeat', t=int, default=1, help='Number of time steps to repeat every input feature.')
    speech_lsnn.add_hyperparameter('preprocess', t=str, default='mfcc', help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
    speech_lsnn.add_hyperparameter('data_url',t=str, default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz', help='Location of speech training data archive on the web.')
    speech_lsnn.add_hyperparameter('data_dir',t=str, default='tmp/speech_dataset/', help="Where to download the speech training data to.")
    speech_lsnn.add_hyperparameter('silence_percentage', t=float, default=10.0, help="How much of the training data should be silence.")
    speech_lsnn.add_hyperparameter('unknown_percentage', t=float, default=10.0, help="How much of the training data should be unknown words.")
    speech_lsnn.add_hyperparameter('validation_percentage', t=float,default=10.0, help='What percentage of wavs to use as a validation set.')
    speech_lsnn.add_hyperparameter('testing_percentage', t=float,default=10.0, help='What percentage of wavs to use as a test set.')
    speech_lsnn.add_hyperparameter('n_thr_spikes', t=int, default=-1, help='Number of thresholds in thr-crossing analog to spike encoding.')
    speech_lsnn.add_hyperparameter('background_volume',t=float, default=0.1, help="How loud the background noise should be, between 0 and 1.")
    speech_lsnn.add_hyperparameter('background_frequency',t=float, default=0.8, help="How many of the training samples have background noise mixed in.")
    speech_lsnn.add_hyperparameter('time_shift_ms', t=float, default=100.0,help="Range to randomly shift the training audio by in time.")
    speech_lsnn.add_hyperparameter('dt',t=float,default=1.,help='Simulation dt')
    speech_lsnn.add_hyperparameter('tau', t=float, default=20., help='Membrane time constant of ALIF neurons in LSNN.')
    speech_lsnn.add_hyperparameter('beta', t=float, default=2., help='Adaptation coefficient of ALIF neurons in LSNN.')
    speech_lsnn.add_hyperparameter('tau_adaptation', t=float, default=98., help='Tau adaptation coefficient of ALIF neurons in LSNN.')
    speech_lsnn.add_hyperparameter('thr', t=float, default=0.01, help='Neurons threshold')
    speech_lsnn.add_hyperparameter('thr_min', t=float, default=0.005, help='Min. membrane threshold')
    speech_lsnn.add_hyperparameter('refr', t=int, default=2, help='Number of refractory time steps of ALIF neurons in LSNN.')
    speech_lsnn.add_hyperparameter('dampening_factor', t=float, default=0.3, help='Dampening factor.')
    speech_lsnn.add_hyperparameter('dropout_prob',t=float,default=0.0,help='Dropout probability for recurrent models.')
    speech_lsnn.add_hyperparameter('reg',t=float,default=0.001,help='Firing rate regularization coefficient.')

    speech_lsnn.add_hyperparameter('learning_rate', t=str, default='0.001,0.0001', help='How large a learning rate to use when training.')#'0.001,0.0001'
    speech_lsnn.add_hyperparameter('eval_step_interval', t=int, default=80, help='How often to evaluate the training results.')
    speech_lsnn.add_hyperparameter('wanted_words', t=str, default='yes,no', help='Words to use (others will be added to an unknown label)')
    speech_lsnn.add_hyperparameter('n_epochs', t=str, default='8,2', help='The number of epochs used in training.')#'8,2'
    speech_lsnn.add_hyperparameter('n_attack_steps', default=10, t=int, help='Number of steps used to find Theta*')#10
    speech_lsnn.add_hyperparameter('beta_robustness', default=1.0, t=float, help='Beta used for weighting lipschitzness term')
    speech_lsnn.add_hyperparameter('seed', t=int, default=42, help="Seed used to initialize data loader and initial matrices")
    speech_lsnn.add_hyperparameter('n_hidden', t=int, default=2048, help='Number of hidden units in recurrent models.')#2048
    speech_lsnn.add_hyperparameter('n_layer', t=int, default=1, help='Number of stacked layers in recurrent models.')
    speech_lsnn.add_hyperparameter('batch_size', t=int, default=100, help='How many items to train with at once')#100
    speech_lsnn.add_hyperparameter("attack_size_constant",t=float,default=0.01,help='')
    speech_lsnn.add_hyperparameter("attack_size_mismatch",t=float,default=0.0,help='')
    speech_lsnn.add_hyperparameter("initial_std_constant",t=float,default=0.0001,help='')
    speech_lsnn.add_hyperparameter("initial_std_mismatch",t=float,default=0.0,help='')

    speech_lsnn.save()

    #speech_lsnn_cube_beta_sweep = experiment_utils.ModelGrid("speech_lsnn_cube_beta_sweep",speech_lsnn, {"attack_size_constant":0.01, "attack_size_mismatch":0.0,"beta_robustness":[0.0, 0.001, 0.01, 0.1, 1.0, 10]})

    speech_lsnn_cube_beta_0_1 = experiment_utils.ModelGrid("speech_lsnn_cube_beta_sweep",speech_lsnn, {"attack_size_constant":0.01, "attack_size_mismatch":0.0,"beta_robustness":[0.0, 1.0], "seed":[0,1,2,3,4]})

    #speech_lsnn_prism_beta_sweep = experiment_utils.ModelGrid("speech_lsnn_prism_beta_sweep",speech_lsnn, {"attack_size_constant":0.0, "attack_size_mismatch":2.0,"beta_robustness":[0.0, 0.001, 0.01, 0.1, 1.0, 10]})

    #speech_lsnn_prism_beta_0_1 = experiment_utils.ModelGrid("speech_lsnn_prism_beta_0_1",speech_lsnn, {"attack_size_constant":0.0, "attack_size_mismatch":2.0,"beta_robustness":[0.0,1.0]})

    #speech_lsnn_padded_prism_beta_sweep = experiment_utils.ModelGrid("speech_lsnn_prism_beta_sweep",speech_lsnn, {"attack_size_constant":0.01, "attack_size_mismatch":2.0,"beta_robustness":[0.0, 0.001, 0.01, 0.1, 1.0, 10]})

    #speech_lsnn_padded_prism_beta_0_1 = experiment_utils.ModelGrid("speech_lsnn_prism_beta_0_1",speech_lsnn, {"attack_size_constant":0.01, "attack_size_mismatch":2.0,"beta_robustness":[0.0,1.0]})


    # speech_lsnn_cube_beta_sweep.save()
    speech_lsnn_cube_beta_0_1.save()
    # speech_lsnn_prism_beta_sweep.save()
    # speech_lsnn_prism_beta_0_1.save()
    # speech_lsnn_padded_prism_beta_sweep.save()
    # speech_lsnn_padded_prism_beta_0_1.save()

    mismatch_sweep = {"Test_Acc_MM_0.0": (test_accuracy_after_mismatch_attack_speech_lsnn, 0.0),
                    "Test_Acc_MM_0.5": (test_accuracy_after_mismatch_attack_speech_lsnn, 0.5),
                    "Test_Acc_MM_0.7": (test_accuracy_after_mismatch_attack_speech_lsnn, 0.7),
                    "Test_Acc_MM_0.9": (test_accuracy_after_mismatch_attack_speech_lsnn, 0.9),
                    "Test_Acc_MM_1.1": (test_accuracy_after_mismatch_attack_speech_lsnn, 1.1),
                    "Test_Acc_MM_1.5": (test_accuracy_after_mismatch_attack_speech_lsnn, 0.5)}

    exp_a = experiment_utils.Experiment("a",mismatch_sweep,[speech_lsnn_cube_beta_0_1],[],{"raw":str, "violin": visualize_exp_a})

    exp_a.save()

