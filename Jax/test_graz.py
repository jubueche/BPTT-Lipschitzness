from jax import config
config.FLAGS.jax_log_compiles=True

import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
import input_data_eager as input_data
from six.moves import xrange
from CNN_Jax import CNN
from GraphExecution import utils
from jax import random
import jax.numpy as jnp
from loss_jax import categorical_cross_entropy, compute_gradient_and_update, attack_network
from jax.experimental import optimizers
import ujson as json
import matplotlib.pyplot as plt 
from datetime import datetime
import jax.nn.initializers as jini
from import_data import DataLoader
from jax import lax
from CNN_torch_ref import CNNModule
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable


def get_batched_accuracy(y, logits):
    predicted_labels = jnp.argmax(logits, axis=1)
    correct_prediction = jnp.array(predicted_labels == y, dtype=jnp.float32)
    batch_acc = jnp.mean(correct_prediction)
    return batch_acc

if __name__ == '__main__':

    parser = utils.get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...",flush=True)
        print(unparsed,flush=True)
        sys.exit(0)


    # - Paths
    base_path = path.dirname(path.abspath(__file__))
    stored_name = '{}_{}_h{}_b{}_s{}'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),FLAGS.model_architecture, FLAGS.n_hidden,FLAGS.beta_lipschitzness,FLAGS.seed)
    model_name = f"{stored_name}_model.json"
    track_name = f"{stored_name}_track.json"
    model_save_path = path.join(base_path, f"Resources/{model_name}")
    track_save_path = path.join(base_path, f"Resources/Plotting/{track_name}")


    model_settings = utils.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                        len(learning_rates_list)))

    data_loader = DataLoader(FLAGS.batch_size)

    d_Out = data_loader.y_train.shape[1]

    # - TODO Put into utils
    Kernels = [[64,1,4,4], [64,64,4,4] ]
    Dense   = [[1600,256],[256,64],[64, d_Out]]

    model=CNNModule()
    learning_rate=0.015




    K1 = model.cnn1.weight.clone().detach().numpy()
    CB1 = onp.reshape(model.cnn1.bias.clone().detach().numpy(), (1,-1,1,1))
    K2 = model.cnn2.weight.clone().detach().numpy()
    CB2 = onp.reshape(model.cnn2.bias.clone().detach().numpy(), (1,-1,1,1))
    W1 = model.fcl1.weight.clone().detach().numpy().T
    B1 = model.fcl1.bias.clone().detach().numpy()
    W2 = model.fcl2.weight.clone().detach().numpy().T
    B2 = model.fcl2.bias.clone().detach().numpy()
    W3 = model.fcl3.weight.clone().detach().numpy().T
    B3 = model.fcl3.bias.clone().detach().numpy()

    # - Create the model
    cnn = CNN(model_settings)

    init_params = {"K1": K1, "CB1": CB1, "K2": K2, "CB2": CB2, "W1": W1, "W2": W2, "W3": W3, "B1": B1, "B2": B2, "B3": B3}
    
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_state = opt_init(init_params)

    track_dict = {"training_accuracies": [], "attacked_training_accuracies": [], "kl_over_time": [], "validation_accuracy": [], "attacked_validation_accuracy": [], "validation_kl_over_time": [], "model_parameters": model_settings}
    best_val_acc = 0.0

    criterion=nn.CrossEntropyLoss()

    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

    iteration = 5
    Record_torch_weights = []
    Record_jax_weights = []
    Record_torch_outputs = []
    Record_jax_outputs = []
    Record_torch_weights.append([model.cnn1.weight.clone().detach().numpy(), model.cnn2.weight.clone().detach().numpy(), model.fcl1.weight.clone().detach().numpy().T, model.fcl2.weight.clone().detach().numpy().T, model.fcl3.weight.clone().detach().numpy().T])
    params = get_params(opt_state)
    Record_jax_weights.append([onp.array(params['K1']), onp.array(params['K2']), onp.array(params['W1']), onp.array(params['W2']), onp.array(params['W3'])])


    for i in range(iteration):
        # - Get training data
        # - TODO Verify correctness
        (X,y) = data_loader.get_batch()
        y = onp.argmax(y, axis=1)
        #y = y.reshape(y.shape[0])


        images=Variable(torch.Tensor(X))
        labels=Variable(torch.Tensor(y).long())

        optimizer.zero_grad()
        outputs=model(images)
        Record_torch_outputs.append(outputs.clone().detach().numpy())
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        Record_torch_weights.append([model.cnn1.weight.clone().detach().numpy(), model.cnn2.weight.clone().detach().numpy(), model.fcl1.weight.clone().detach().numpy().T, model.fcl2.weight.clone().detach().numpy().T, model.fcl3.weight.clone().detach().numpy().T])
        
        params = get_params(opt_state)
        jax_outputs, _ = cnn.call(X, **params)
        Record_jax_outputs.append(onp.array(jax_outputs))

        opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, cnn, FLAGS, cnn._rng_key)

        params = get_params(opt_state)
        Record_jax_weights.append([onp.array(params['K1']), onp.array(params['K2']), onp.array(params['W1']), onp.array(params['W2']), onp.array(params['W3'])])

    Norms = []
    for i in range(iteration+1):
        for k in range(len(Record_torch_weights[i])):
            norm = onp.linalg.norm(Record_torch_weights[i][k]-Record_jax_weights[i][k])
            print(norm)

    for i in range(iteration):
        norm = onp.linalg.norm(Record_torch_outputs[i]-Record_jax_outputs[i])
        print(norm)

    while (True):
        a+=1









        
