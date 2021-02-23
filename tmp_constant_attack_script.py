from Experiments import sgd_experiment, hessian_experiment, weight_scale_experiment, mismatch_experiment, methods_experiment, constant_attack_experiment, quantization_experiment, treat_as_constant_experiment
from ECG.ecg_data_loader import ECGDataLoader

# constant_attack_experiment.constant_attack_experiment.visualize()
# methods_experiment.methods_experiment.visualize()
# mismatch_experiment.mismatch_experiment.visualize()
# sgd_experiment.sgd_experiment.visualize()
# hessian_experiment.hessian_experiment.visualize()
# weight_scale_experiment.weight_scale_experiment.visualize()
# quantization_experiment.quantization_experiment.visualize()
treat_as_constant_experiment.treat_as_constant_experiment.visualize()

# import matplotlib.pyplot as plt
# ecg_loader = ECGDataLoader(path = "/home/julian/Documents/BPTT-Lipschitzness/ECG/ecg_recordings", batch_size=100)
# X,y,seq = ecg_loader.get_sequence()
# print(seq.shape)
# plt.plot(seq)
# plt.show()