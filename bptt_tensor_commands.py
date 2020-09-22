import ujson as json
import numpy as np
from jax import vmap
import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
matplotlib.rcParams['axes.xmargin'] = 0
matplotlib.rcParams['figure.figsize'] = [15, 10]
import matplotlib.pyplot as plt
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, FFLIFCurrentInJax_SO, FFExpSynCurrentInJax, RecLIFCurrentInJax_SO
from rockpool.networks import JaxStack
import os
import sys
import argparse
from loss import loss_mse_reg_stack, loss_lipschitzness_verbose
from data_loader import (
        get_latest_model,
        AudioDataset,
        get_label_distribution,
        ClassWeightedRandomSampler,
    )
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from preprocess import StandardizeDataLength, ButterMel, Subsample, Smooth
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TensorCommandsBPTT():

    def init_data_set(self, partition, data_transform, target_transform):
        ds = AudioDataset(path=self.data_path,
                            config="tensorcommands.json",
                            data_partition=partition,
                            key_words=self.key_words,
                            transform=data_transform,
                            target_signal_transform=target_transform,
                            cache=self.cache_path
                        )

        labels = [ds.label_map[k] for k in ds.data_list[:, 1]]
        sampler = ClassWeightedRandomSampler(labels, torch.ones(len(self.key_words) + 1))
        return DataLoader(ds, batch_size=self.batch_size, sampler=sampler)


    def __init__(self,
                    num_neurons,
                    num_epochs,
                    batch_size,
                    key_words,
                    use_lipschitzness,
                    data_path,
                    cache_path,
                    save_path,
                    verbose):

        self.num_neurons = num_neurons
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.key_words = key_words
        self.use_lipschitzness = use_lipschitzness
        self.data_path = data_path
        self.cache_path = cache_path
        self.resources_path = save_path
        self.verbose = verbose
        self.N_out = len(self.key_words)+1 # - Keywords and "nothing"
        self.num_filters = 64
        self.dt = 0.001
        self.time_base = np.linspace(0.0,1.0,100)

        self.tau_mem_rec = 0.05
        self.tau_syn_out = 0.07
        self.w_scale_in = self.tau_mem_rec / (self.dt*self.num_filters)
        self.w_scale_rec = self.tau_mem_rec / (self.dt*self.num_neurons)
        self.w_scale_out = self.tau_syn_out / (self.dt*self.num_neurons) * 0.1

        # - Initialize PyTorch summary writer
        str_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        summary_dir = "/mnt/c/users/Serious/Documents/github/BPTT-Lipschitzness/runs/" + str_time + "/" 
        self.writer = SummaryWriter(summary_dir)

        self.save_path = os.path.join(save_path, str_time + ".model")

        # - Define input and target transformers (pre-processing)
        data_transform = transforms.Compose(
            [
            StandardizeDataLength(data_length=16000),
            ButterMel(sampling_freq=16000, cutoff_freq=100, num_filters=self.num_filters, order=2),
            ]
        )

        target_transform = transforms.Compose(
            [
                StandardizeDataLength(data_length=16000),
                Subsample(downsample=160),
                Smooth(sigma=5.0),
            ]
        )

        # - Initialize the data loaders
        self.train_data_loader = self.init_data_set("train", data_transform, target_transform)
        self.val_data_loader = self.init_data_set("val", data_transform, target_transform)
        self.test_data_loader = self.init_data_set("test", data_transform, target_transform)

        # - Initialize model
        self.net = self.init_model()

    def init_model(self):

        # - Try to find a model
        model_path = get_latest_model(self.resources_path)
        if(model_path is None):
            w_spiking_in = self.w_scale_in * np.random.randn(self.num_filters, self.num_neurons)
            w_spiking_rec =  self.w_scale_rec * np.random.randn(self.num_neurons, self.num_neurons)
            w_spiking_out = self.w_scale_out * np.random.randn(self.num_neurons, self.N_out)
            spiking_bias = -0.01 * np.ones(self.num_neurons)

            lyrLIFInput = FFLIFCurrentInJax_SO(
                w_in = w_spiking_in,
                tau_syn = 0.05,
                tau_mem = 0.05,
                bias = 0.,
                noise_std = 0.0,
                dt = self.dt,
                name = 'LIF_Input',    
            )

            lyrLIFRecurrent = RecLIFCurrentInJax_SO(
                w_recurrent = w_spiking_rec,
                tau_mem = self.tau_mem_rec,
                tau_syn = 0.05,
                bias = spiking_bias,
                noise_std = 0.0,
                dt = self.dt,
                name = 'LIF_Reservoir',
            )

            lyrLIFReadout = FFExpSynCurrentInJax(
                w_out = w_spiking_out,
                tau = self.tau_syn_out,
                noise_std = 0.0,
                dt = self.dt,
                name = 'LIF_Readout',
            )
            net = JaxStack([lyrLIFInput, lyrLIFRecurrent, lyrLIFReadout])
        else:
            net = self.load_net(os.path.join(self.resources_path, model_path))
            print(f"Loaded network {os.path.join(self.resources_path, model_path)}")
        return net

    def save(self, fn):
        savedict = self.net.to_dict()
        with open(fn, "w") as f:
            json.dump(savedict, f)

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        net = Network.load_from_dict(loaddict)
        return JaxStack([l for l in net.evol_order])

    def expand_target_signal(self, target_signals, labels):
        tg = np.zeros((target_signals.shape[0], self.N_out, target_signals.shape[1]))
        for i, sig in enumerate(target_signals):
            tg[i,labels[i],:] = target_signals[i,:]
        return tg

    def train(self):
        mses = lip_losses = losses = []
        epoch_id = batch_id = epoch_loss = 0
        best_val_acc = 1 / self.N_out # - Naive classifier performance
        is_first = True
        loss_func = loss_mse_reg_stack
        if(self.use_lipschitzness):
            if(self.verbose > 0):
                loss_func = loss_lipschitzness_verbose
            else:
                loss_func = loss_lipschitzness_verbose

        loss_params = {'min_tau': 0.01,
                            'reg_tau': 1000.0,
                            'reg_l2_rec': 1000.0,
                            'reg_act1': 0.0,
                            'reg_act2': 0.0,
                            'lambda_mse': 10000.0}
        
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            pbar_epoch = tqdm(self.train_data_loader)
            for signals, labels, target_signals in pbar_epoch:
                target_signals = self.expand_target_signal(target_signals, labels)

                if(self.use_lipschitzness):
                    loss_params['net'] = self.net
                    loss_params['step_size'] = 0.0005
                    loss_params['number_steps'] = 10
                    loss_params['beta'] = 100.0
                    loss_params['initial_std'] = 0.2

                self.net.reset_all()
                fLoss, _, _ = self.net.train_output_target(
                    torch.transpose(signals,1,2).numpy(),
                    np.transpose(target_signals, (0,2,1)),
                    is_first = is_first,
                    batch_axis = 0,
                    loss_aux = True,
                    debug_nans = False,
                    loss_fcn = loss_func,
                    opt_params = {"step_size": 1e-5},
                    loss_params = loss_params)
                is_first = False

                epoch_loss += fLoss[0]
                losses.append(fLoss[0])
                mses.append(np.sum(fLoss[1]["mse"]))
                if(self.use_lipschitzness):
                    lip_losses.append(np.sum(fLoss[1]["loss_lip"]))


                if(self.use_lipschitzness and self.verbose > 0):
                    plt.clf()
                    ax = plt.gca()
                    stagger = np.zeros((target_signals[0].shape))
                    stagger[1,:] +=1.5 ; stagger[2,:] += 3.0
                    l1 = ax.plot(self.time_base, (target_signals[0]+stagger).T, color="r", label="Target")
                    l2 = ax.plot(self.time_base, fLoss[1]["output_batch_t"][0]+stagger.T, color="g", label="Output")
                    l3 = ax.plot(self.time_base, fLoss[1]["batched_output_over_time"][0][0]+stagger.T, color="k", label="Random")
                    lx = []
                    c_range = np.linspace(0.0,1.0,len(fLoss[1]["batched_output_over_time"][1:]))[::-1]
                    for idx,output in enumerate(fLoss[1]["batched_output_over_time"][1:]):
                        lt = ax.plot(self.time_base, output[0]+stagger.T, color=(0.01,c_range[idx],1.0,1.0))
                        lx.append(lt)
                    lines = [l1[0],l2[0],l3[0]] 
                    ax.legend(lines, [r"Target", r"Output", r"Random"], frameon=False, loc=1, prop={'size': 7})
                    plt.draw()
                    plt.pause(0.001)
                elif(self.verbose > 0):
                    plt.clf()
                    ax = plt.gca()
                    stagger = np.zeros((target_signals[0].shape))
                    stagger[1,:] += 1.5 ; stagger[2,:] += 3.0
                    ax.plot(self.time_base, fLoss[1]["output_batch_t"][0]+stagger.T, color="k")
                    ax.plot(self.time_base, (target_signals[0]+stagger).T, color="r")
                    plt.draw()
                    plt.pause(0.001)

                n_iter = epoch_id*self.batch_size+batch_id
                self.writer.add_scalar("Loss/MSE", float(np.mean(fLoss[1]["mse"])) / loss_params["lambda_mse"], n_iter)
                self.writer.add_scalar("Loss/tau_loss", float(np.mean(fLoss[1]["tau_loss"])) / loss_params["reg_tau"], n_iter)
                self.writer.add_scalar("Loss/Loss", float(fLoss[0]), n_iter)
                self.writer.add_scalar("Weights/Rec", np.max(self.net.LIF_Reservoir.w_recurrent), n_iter)
                self.writer.add_scalar("min_tau_mem/Rec", np.min(self.net.LIF_Reservoir.tau_mem), n_iter)
                if(self.use_lipschitzness):
                    self.writer.add_scalar("Loss/Lipschitzness", float(np.mean(fLoss[1]["loss_lip"])) / loss_params["beta"], n_iter)
                    if(self.verbose > 0):
                        self.writer.add_scalar("Loss/DistanceTheta*-ThetaStart", float(np.mean(fLoss[1]["theta_start_star_distance"])), n_iter)

                batch_id += 1

            val_acc = self.validate()
            if(val_acc >= best_val_acc):
                best_val_acc = val_acc
                self.save(self.save_path)
            self.writer.add_scalar("Acc/Val", val_acc, epoch_id)
            epoch_id += 1
        
        return best_val_acc


    def get_prediction(self, output):
        return np.argmax(np.sum(output, axis=1), axis=1)

    def validate(self):
        correct = counter = 0
        pbar_val = tqdm(self.val_data_loader)
        for signals, labels, target_signals in pbar_val:
            target_signals = self.expand_target_signal(target_signals, labels)
            output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, torch.transpose(signals,1,2).numpy())
            predicted_labels = self.get_prediction(output)
            correct += np.sum(np.asarray(predicted_labels == labels.numpy(), dtype=int))
            counter += len(predicted_labels)
        return correct / counter

    def test(self):
        # - Load the best network
        self.best_net = self.load_net(self.save_path)
        correct = counter = 0
        pbar_test = tqdm(self.test_data_loader)
        for signals, labels, target_signals in pbar_test:
            target_signals = self.expand_target_signal(target_signals, labels)
            output, _, _ = vmap(self.best_net._evolve_functional, in_axes=(None, None, 0))(self.best_net._pack(), self.best_net._state, torch.transpose(signals,1,2).numpy())
            predicted_labels = self.get_prediction(output)
            correct += np.sum(np.asarray(predicted_labels == labels.numpy(), dtype=int))
            counter += len(predicted_labels)
        return correct / counter

if __name__ == "__main__":

    num_neurons = 512
    num_epochs = 10
    batch_size = 10
    key_words = ["yes", "no"]
    use_lipschitzness = True
    data_path = "/home/serious/Datasets/TensorCommands/"
    cache_path ="/home/serious/Cached"
    save_path = "/home/serious/Resources"
    verbose = 1


    model = TensorCommandsBPTT(num_neurons=num_neurons,
                                num_epochs=num_epochs,
                                batch_size=batch_size,
                                key_words=key_words,
                                use_lipschitzness=use_lipschitzness,
                                data_path=data_path,
                                cache_path=cache_path,
                                save_path=save_path,
                                verbose=verbose
                                )

    # - Start training
    val_acc = model.train()

    # - Test
    test_acc = model.test()

    print(f"Done. Best val acc {val_acc}, test acc {test_acc} model saved in {model.save_path}")
