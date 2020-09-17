import warnings
warnings.filterwarnings('ignore')
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
from SIMMBA import BaseModel
from SIMMBA.experiments.HeySnipsDEMAND import HeySnipsDEMAND
from rockpool.timeseries import TSContinuous
from rockpool import layers, Network
from rockpool.layers import H_tanh, RecRateEulerJax_IO, FFLIFCurrentInJax_SO, FFExpSynCurrentInJax, RecLIFCurrentInJax_SO
from rockpool.networks import JaxStack
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import argparse
from loss import loss_lipschitzness, loss_mse_reg_stack


class HeySnipsNetworkADS(BaseModel):
    def __init__(self,
                 labels,
                 num_neurons,
                 num_epochs,
                 use_lipschitzness,
                 fs=16000.,
                 verbose=0,
                 name="Snips ADS",
                 version="1.0"):
        
        super(HeySnipsNetworkADS, self).__init__(name,version)

        self.num_neurons = num_neurons
        self.num_epochs = num_epochs
        self.threshold = 0.7
        self.threshold0 = 0.5
        self.best_boundary = 200
        self.fs = fs
        self.verbose = verbose
        self.noise_std = 0.0
        self.dt = 0.001
        self.use_lipschitzness = use_lipschitzness 

        self.base_path = "/home/julian/Documents/BPTT-Lipschitzness/"
        # self.base_path = "/home/julian_synsense_ai/BPTT-Lipschitzness/"

        self.network_name = "Resources/bptt.json"
        if(self.use_lipschitzness):
            self.network_name = "Resources/bptt_lipschitznes.json"

        rate_net_path = os.path.join(self.base_path, "Resources/rate_heysnips_tanh_0_16.model")
        with open(rate_net_path, "r") as f:
            config = json.load(f)

        self.w_in = np.array(config['w_in'])
        self.w_rec = np.array(config['w_recurrent'])
        self.w_out = np.array(config['w_out'])
        self.bias = config['bias']
        self.tau_rate = config['tau']

        self.rate_layer = RecRateEulerJax_IO(w_in=self.w_in,
                                             w_recurrent=self.w_rec,
                                             w_out=self.w_out,
                                             tau=self.tau_rate,
                                             bias=self.bias,
                                             activation_func=H_tanh,
                                             dt=self.dt,
                                             noise_std=self.noise_std,
                                             name="hidden")

        self.N_out = self.w_out.shape[1]
        self.num_units = self.w_rec.shape[0]
        self.rate_layer.reset_state()
        self.lr_params = self.rate_layer._pack()
        self.lr_state = self.rate_layer._state


        # - Create spiking net
        self.model_path_bptt_net = os.path.join(self.base_path, self.network_name)
        if(os.path.exists(self.model_path_bptt_net)):
            print("Network already trained. Exiting. If you would like to re-train, please comment out this line and delete/rename the model.")
            # sys.exit(0)
            self.net = self.load_net(self.model_path_bptt_net)
            print("Loaded pretrained network")
        else:
            # - Create spiking net
            # - Scaling factors for initializing with rate net weights
            
            # - Works well for 512 neurons
            self.w_scale_in = 0.2
            self.w_scale_rec = 0.1
            self.w_scale_out = 0.001

            if(self.num_neurons == self.num_units):
                w_spiking_in = self.w_scale_in * self.w_in
                w_spiking_rec = self.w_scale_rec * self.w_rec
                spiking_tau_mem = self.tau_rate
                spiking_bias = self.bias
                w_spiking_out = self.w_scale_out * self.w_out
            else:
                # - Works well for 512 neurons
                D = 0.01*np.random.randn(self.num_neurons, self.num_units)
                w_spiking_in = self.w_scale_in * (self.w_in @ D.T)
                w_spiking_rec = np.zeros((self.num_neurons, self.num_neurons))
                w_spiking_out = self.w_scale_out * (D @ self.w_out)
                spiking_bias = np.mean(self.bias)
                spiking_tau_mem = 0.05

            lyrLIFInput = FFLIFCurrentInJax_SO(
                w_in = w_spiking_in,
                tau_syn = 0.1,
                tau_mem = 0.1,
                bias = 0.,
                noise_std = self.noise_std,
                dt = self.dt,
                name = 'LIF_Input',    
            )

            lyrLIFRecurrent = RecLIFCurrentInJax_SO(
                w_recurrent = w_spiking_rec,
                tau_mem = spiking_tau_mem,
                tau_syn = 0.07,
                bias = spiking_bias,
                noise_std = self.noise_std,
                dt = self.dt,
                name = 'LIF_Reservoir',
            )

            lyrLIFReadout = FFExpSynCurrentInJax(
                w_out = w_spiking_out,
                tau = 0.07,
                noise_std = self.noise_std,
                dt = self.dt,
                name = 'LIF_Readout',
            )

            self.net = JaxStack([lyrLIFInput, lyrLIFRecurrent, lyrLIFReadout])
            self.best_val_acc = 0.0

            if(os.path.exists(os.path.join(self.base_path, "Resources/bptt.json"))):
                self.net = self.load_net(os.path.join(self.base_path, "Resources/bptt.json"))

    def save(self, fn):
        savedict = self.net.to_dict()
        savedict["threshold0"] = self.threshold0
        savedict["best_val_acc"] = self.best_val_acc
        savedict["best_boundary"] = float(self.best_boundary)
        with open(fn, "w") as f:
            json.dump(savedict, f)

    def load_net(self, fn):
        with open(fn, "r") as f:
            loaddict = json.load(f)
        self.threshold0 = loaddict.pop("threshold0")
        self.best_val_acc = loaddict.pop("best_val_acc")
        self.best_boundary = loaddict.pop("best_boundary")
        net = Network.load_from_dict(loaddict)
        return JaxStack([l for l in net.evol_order])

    def get_data(self, filtered_batch):
        """
        Evolves filtered audio samples in the batch through the rate network to obtain rate output
        :param np.ndarray filtered_batch: Shape: [batch_size,T,num_channels], e.g. [100,5000,16]
        :returns np.ndarray batched_rate_output: Shape: [batch_size,T,N_out] [Batch size is always first dimensions]
        """
        batched_rate_output, _, _ = vmap(self.rate_layer._evolve_functional, in_axes=(None, None, 0))(self.lr_params, self.lr_state, filtered_batch)
        return batched_rate_output

    def train(self, data_loader, fn_metrics):

        mses = []
        lip_losses = []
        losses = []

        loss_func = loss_mse_reg_stack
        if(self.use_lipschitzness):
            loss_func = loss_lipschitzness

        for epoch in range(self.num_epochs):

            epoch_loss = 0

            for batch_id, [batch, _] in enumerate(data_loader.train_set()):

                filtered = np.stack([s[0][1] for s in batch])
                # - Get the output of the rate network for the batch, shape [batch_size, T, N_out]
                tgt_signals = np.stack([s[2] for s in batch])
                target_labels = [s[1] for s in batch]
                batched_rate_output = self.get_data(filtered_batch=filtered)
                tgt_signals = tgt_signals[:,:filtered.shape[1],:]
                time_base = np.arange(0,filtered.shape[1]*self.dt, self.dt)

                loss_params = {'min_tau': 0.01,
                            'reg_tau': 10000.0,
                            'reg_l2_rec': 100000.0,
                            'reg_act1': 2.0,
                            'reg_act2': 2.0,
                            'lambda_mse': 1000.0}
                if(self.use_lipschitzness):
                    loss_params['net'] = self.net

                # - Do the training step
                self.net.reset_all()
                # - Works well for 512
                fLoss, _, _ = self.net.train_output_target(
                    filtered,
                    tgt_signals,
                    is_first = (batch_id == epoch == 0),
                    batch_axis = 0,
                    loss_aux = True,
                    debug_nans = False,
                    loss_fcn = loss_func,
                    opt_params = {"step_size": 1e-4},
                    loss_params = loss_params)

                epoch_loss += fLoss[0]
                losses.append(fLoss[0])
                mses.append(np.sum(fLoss[1]["mse"]))
                if(self.use_lipschitzness):
                    lip_losses.append(np.sum(fLoss[1]["loss_lip"]))

                print("--------------------", flush=True)
                print("Epoch", epoch, "Batch ID", batch_id , flush=True)
                if(self.use_lipschitzness):
                    print("Loss", fLoss[0], "MSE Loss:", np.sum(fLoss[1]["mse"]), "Lipschitzness Loss:", np.sum(fLoss[1]["loss_lip"]), "Mean w_rec", np.mean(self.net.LIF_Reservoir.weights), flush=True)
                    print("Lipschitzness over time", [float(e) for e in np.array(fLoss[1]["lipschitzness_over_time"])])
                    print("Distance theta start, theta star", [float(e) for e in np.array(fLoss[1]["theta_start_star_distance"])])
                    print("lipschitzness_random", fLoss[1]["lipschitzness_random"])
                    print("random_distance_to_start", fLoss[1]["random_distance_to_start"])
                else:
                    print("Loss", fLoss[0], "MSE Loss:", np.sum(fLoss[1]["mse"]), "Mean w_rec", np.mean(self.net.LIF_Reservoir.weights), flush=True)
                print("--------------------", flush=True)

                # plt.clf()
                # plt.subplot(211)
                # plt.hist(fLoss[1]["theta_start"]["tau_mem"], bins=50, label=r"$\Theta_{\tau_{mem}}^{start}$")
                # plt.hist(fLoss[1]["theta_star"]["tau_mem"], bins=50, label=r"$\Theta_{\tau_{mem}}^*$")
                # plt.legend()
                # plt.hist(fLoss[1]["theta_start"]["tau_syn"], bins=50, label=r"$\Theta_{\tau_{syn}}^{start}$")
                # plt.hist(fLoss[1]["theta_star"]["tau_syn"], bins=50, label=r"$\Theta_{\tau_{syn}}^*$")
                # plt.legend()
                # plt.subplot(212)
                # plt.hist(self.net.LIF_Reservoir.tau_mem, bins=50, label=r"$\tau_{mem}$")
                # plt.hist(self.net.LIF_Reservoir.tau_syn, bins=50, label=r"$\tau_{syn}$")
                # plt.legend()
                # plt.draw()
                # plt.pause(0.1)

                if(self.use_lipschitzness):
                    plt.clf()
                    plt.plot(time_base, tgt_signals[0], label="Target")
                    plt.plot(time_base, fLoss[1]["output_batch_t"][0], label="Output")
                    plt.plot(time_base, fLoss[1]["batched_output_over_time"][0][0], label="Random 20%")
                    for idx,output in enumerate(fLoss[1]["batched_output_over_time"][1:]):
                        plt.plot(time_base, output[0], label=f"Iter. {idx}")
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)


            # - End epoch
            yield {"train_loss": epoch_loss}

            # - Validate...
            val_acc = self.perform_validation_set(data_loader=data_loader, fn_metrics=fn_metrics)

            if(val_acc >= self.best_val_acc):
                self.best_val_acc = val_acc
                # - Save model
                self.save(self.model_path_bptt_net)

                if(val_acc > 0.87):
                    return

        # - End for epoch
    # - End train

    def perform_validation_set(self, data_loader, fn_metrics):

        integral_pairs = []
        correct = 0
        correct_rate = 0
        counter = 0

        for batch_id, [batch, _] in enumerate(data_loader.val_set()):
        
            if(batch_id*data_loader.batch_size > 1000):
                break
            
            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_spiking_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)
            tgt_signals = np.stack([s[2] for s in batch])
            tgt_signals = tgt_signals[:,:filtered.shape[1],:]
            batched_rate_output = self.get_data(filtered_batch=filtered)
            counter += batched_spiking_output.shape[0]

            for idx in range(batched_spiking_output.shape[0]):

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = np.copy(batched_spiking_output[idx])
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]
                integral_pairs.append((np.max(integral_final_out),target_labels[idx]))

                predicted_label = 0
                if(np.any(batched_spiking_output[idx] > self.threshold)):
                    predicted_label = 1
                
                if(predicted_label == target_labels[idx]):
                    correct += 1

                predicted_rate_label = 0
                if(np.any(batched_rate_output[idx] > self.threshold)):
                    predicted_rate_label = 1
                
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                print("--------------------", flush=True)
                print("Batch", batch_id, "Idx", idx , flush=True)
                print("VALIDATION: True:", target_labels[idx], "Predicted:", predicted_label, "Rate:", predicted_rate_label, flush=True)
                print("--------------------", flush=True)

            if(batch_id % 10 == 0 and self.verbose > 0):
                # - Also evolve over input and plot a little bit
                self.net.reset_all()
                ts_input = TSContinuous(time_base, filtered[0])
                ts_out = self.net.evolve(ts_input)['LIF_Readout'](time_base)
                plt.clf()
                plt.plot(time_base, ts_out, label="Spiking")
                plt.plot(time_base, tgt_signals[0], label="Target")
                plt.plot(time_base, batched_rate_output[0], label="Rate")
                plt.ylim([-0.3,1.0])
                plt.legend()
                plt.draw()
                plt.pause(0.001)

        # - End for batch
        val_acc = correct / counter
        rate_acc = correct_rate / counter
        print("Validation accuracy is %.3f | Rate accuracy is %.3f" % (val_acc, rate_acc), flush=True)

        # - Find best boundaries for classification using the continuous integral
        min_val = np.min([x for (x,y) in integral_pairs])
        max_val = np.max([x for (x,y) in integral_pairs])
        best_boundary = min_val
        best_acc = 0.5
        for boundary in np.linspace(min_val, max_val, 1000):
            acc = (len([x for (x,y) in integral_pairs if y == 1 and x > boundary]) + len([x for (x,y) in integral_pairs if y == 0 and x <= boundary])) / len(integral_pairs)
            if(acc >= best_acc):
                best_acc = acc
                best_boundary = boundary
        
        self.best_boundary = best_boundary
        print(f"Best validation accuracy after finding boundary is {best_acc}")

        return best_acc


    def test(self, data_loader, fn_metrics):

        correct = correct_rate = counter = 0
        self.net = self.load_net(self.model_path_bptt_net)

        for batch_id, [batch, _] in enumerate(data_loader.test_set()):
        
            if(batch_id*data_loader.batch_size > 1000):
                break

            filtered = np.stack([s[0][1] for s in batch])
            target_labels = [s[1] for s in batch]
            batched_spiking_output, _, _ = vmap(self.net._evolve_functional, in_axes=(None, None, 0))(self.net._pack(), self.net._state, filtered)
            tgt_signals = np.stack([s[2] for s in batch])
            batched_rate_output = self.get_data(filtered_batch=filtered)

            counter += batched_spiking_output.shape[0]

            for idx in range(batched_spiking_output.shape[0]):

                # - Compute the integral for the points that lie above threshold0
                integral_final_out = np.copy(batched_spiking_output[idx])
                integral_final_out[integral_final_out < self.threshold0] = 0.0
                for t,val in enumerate(integral_final_out):
                    if(val > 0.0):
                        integral_final_out[t] = val + integral_final_out[t-1]

                predicted_label = 0
                if(np.max(integral_final_out) > self.best_boundary):
                    predicted_label = 1

                if(predicted_label == target_labels[idx]):
                    correct += 1

                predicted_rate_label = 0
                if(np.any(batched_rate_output[idx] > self.threshold)):
                    predicted_rate_label = 1
                
                if(predicted_rate_label == target_labels[idx]):
                    correct_rate += 1

                print("--------------------", flush=True)
                print("Batch", batch_id, "Idx", idx , flush=True)
                print("TESTING: True:", target_labels[idx], "Predicted:", predicted_label, "Rate:", predicted_rate_label, flush=True)
                print("--------------------", flush=True)

                if(self.verbose > 1):
                    # - Also evolve over input and plot a little bit
                    plt.clf()
                    plt.plot(np.linspace(0,5,batched_spiking_output.shape[1]), batched_spiking_output[0], label="Spiking")
                    plt.plot(np.linspace(0,5,tgt_signals.shape[1]), tgt_signals[0], label="Target")
                    plt.plot(np.linspace(0,5,batched_rate_output.shape[1]), batched_rate_output[0], label="Rate")
                    plt.ylim([-0.3,1.0])
                    plt.legend()
                    plt.draw()
                    plt.pause(0.001)

        # - End for batch
        test_acc = correct / counter
        rate_acc = correct_rate / counter
        print("Test accuracy is %.3f | Rate accuracy is %.3f" % (test_acc, rate_acc), flush=True)

if __name__ == "__main__":

    # np.random.seed(42)

    parser = argparse.ArgumentParser(description='Learn classifier using pre-trained rate network')
    
    parser.add_argument('--num', default=512, type=int, help="Number of neurons in the network, best results obtained using 512 neurons")
    parser.add_argument('--verbose', default=1, type=int, help="Level of verbosity. Default=0. Range: 0 to 2")
    parser.add_argument('--epochs', default=5, type=int, help="Number of training epochs")
    parser.add_argument('--percentage-data', default=1.0, type=float, help="Percentage of total training data used. Example: 0.02 is 2%.")
    parser.add_argument('--lipschitzness', default=False, action='store_true', help="Use Lipschitzness in loss")

    args = vars(parser.parse_args())
    num = args['num']
    verbose = args['verbose']
    num_epochs = args['epochs']
    percentage_data = args['percentage_data']
    use_lipschitzness = args['lipschitzness']

    batch_size = 1
    balance_ratio = 1.0
    snr = 10.

    experiment = HeySnipsDEMAND(batch_size=batch_size,
                            percentage=percentage_data,
                            snr=snr,
                            randomize_after_epoch=True,
                            downsample=1000,
                            is_tracking=False,
                            one_hot=False)

    num_train_batches = int(np.ceil(experiment.num_train_samples / batch_size))
    num_val_batches = int(np.ceil(experiment.num_val_samples / batch_size))
    num_test_batches = int(np.ceil(experiment.num_test_samples / batch_size))

    model = HeySnipsNetworkADS(labels=experiment._data_loader.used_labels,
                                num_neurons=num,
                                num_epochs=num_epochs,
                                verbose=verbose,
                                use_lipschitzness=use_lipschitzness)

    experiment.set_model(model)
    experiment.set_config({'num_train_batches': num_train_batches,
                           'num_val_batches': num_val_batches,
                           'num_test_batches': num_test_batches,
                           'batch size': batch_size,
                           'percentage data': percentage_data,
                           'snr': snr,
                           'balance_ratio': balance_ratio})
    experiment.start()