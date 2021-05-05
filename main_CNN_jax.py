from jax import config
config.FLAGS.jax_log_compiles=True
config.update('jax_disable_jit', False)

import time
import numpy as onp
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from CNN.import_data import CNNDataLoader
from CNN_Jax import CNN
from jax import random
import jax.numpy as jnp
from loss_jax import categorical_cross_entropy, compute_gradient_and_update, compute_gradients
from jax.experimental import optimizers
from EntropySGD.entropy_sgd import EntropySGD_Jax
from ABCD.abcd import ABCD_Jax
import ujson as json
import math
import time
from architectures import cnn as arch
from architectures import log
from concurrent.futures import ThreadPoolExecutor, as_completed
from experiment_utils import get_batched_accuracy, _get_acc_batch, get_val_acc, _get_mismatch_data, get_val_acc, get_lr_schedule

if __name__ == '__main__':
    t0 = time.time()
    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.json")

    data_loader = CNNDataLoader(FLAGS.batch_size, FLAGS.data_dir)
    flags_dict = vars(FLAGS)
    epochs_list = list(map(int, FLAGS.n_epochs.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(epochs_list) != len(learning_rates_list):
        raise Exception(
            '--n_epochs and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(epochs_list),
                                                        len(learning_rates_list)))
    
    steps_list = [math.ceil(epochs * data_loader.N_train/FLAGS.batch_size) for epochs in epochs_list]

    d_Out = data_loader.d_out

    Kernels = FLAGS.Kernels
    Dense   = FLAGS.Dense

    rng_key = random.PRNGKey(FLAGS.seed)
    _, *sks = random.split(rng_key, 10)
    K1 = onp.array(random.truncated_normal(sks[1],-2,2,(Kernels[0][0], Kernels[0][1], Kernels[0][2], Kernels[0][3]))* (onp.sqrt(6/(Kernels[0][0]*Kernels[0][1]*Kernels[0][2] + Kernels[0][0]*Kernels[0][1]*Kernels[0][3]))))
    K2 = onp.array(random.truncated_normal(sks[2],-2,2,(Kernels[1][0], Kernels[1][1], Kernels[1][2], Kernels[1][3]))* (onp.sqrt(6/(Kernels[1][0]*Kernels[1][1]*Kernels[1][2] + Kernels[1][0]*Kernels[1][1]*Kernels[1][3]))))
    CB1 = onp.array(random.truncated_normal(sks[6],-2,2,(1, Kernels[0][1], 1, 1))* (onp.sqrt(6/(Kernels[0][0]*Kernels[0][1]*Kernels[0][2] + Kernels[0][0]*Kernels[0][1]*Kernels[0][3]))))
    CB2 = onp.array(random.truncated_normal(sks[7],-2,2,(1, Kernels[1][1], 1, 1))* (onp.sqrt(6/(Kernels[1][0]*Kernels[1][1]*Kernels[1][2] + Kernels[1][0]*Kernels[1][1]*Kernels[1][3]))))
    W1 = onp.array(random.truncated_normal(sks[3],-2,2,(Dense[0][0], Dense[0][1]))* (onp.sqrt(6/(Dense[0][0] + Dense[0][1]))))
    W2 = onp.array(random.truncated_normal(sks[4],-2,2,(Dense[1][0], Dense[1][1]))* (onp.sqrt(6/(Dense[1][0] + Dense[1][1]))))
    W3 = onp.array(random.truncated_normal(sks[4],-2,2,(Dense[2][0], d_Out))* (onp.sqrt(6/(Dense[2][0] + d_Out))))
    B1 = onp.zeros((Dense[0][1],))
    B2 = onp.zeros((Dense[1][1],))
    B3 = onp.zeros((d_Out,))

    # - Create the model
    cnn = CNN(vars(FLAGS))
    FLAGS.network = cnn
    FLAGS.architecture = "cnn"

    init_params = {"K1": K1, "CB1": CB1, "K2": K2, "CB2": CB2, "W1": W1, "W2": W2, "W3": W3, "B1": B1, "B2": B2, "B3": B3}
    iteration = onp.array(steps_list, int)
    lrs = onp.array(FLAGS.learning_rate.split(","),float)
    
    if(FLAGS.optimizer == "adam"):
        opt_init, opt_update, get_params = optimizers.adam(get_lr_schedule(iteration,lrs), 0.9, 0.999, 1e-08)
    elif(FLAGS.optimizer == "sgd"):
        opt_init, opt_update, get_params = optimizers.sgd(get_lr_schedule(iteration,lrs))
    elif(FLAGS.optimizer == "esgd"):
        config = dict(momentum=0.9, damp=0.0, nesterov=True, weight_decay=0.0, L=10, eps=1e-4, g0=1e-2, g1=1e-3, langevin_lr=0.1, langevin_beta1=0.75, b1=0.0, b2=1.0, eps_adam=1e-8)
        opt_init, opt_update, get_params = EntropySGD_Jax(get_lr_schedule(iteration,lrs), config)
    elif(FLAGS.optimizer == "abcd"):
        config = dict(L=FLAGS.abcd_L, eta_A=FLAGS.abcd_etaA, b1=0.9, b2=0.999, eps=1e-8)
        opt_init, opt_update, get_params = ABCD_Jax(get_lr_schedule(iteration,lrs), config)
    else:
        print("Invalid optimizer")
        sys.exit(0)
    opt_state = opt_init(init_params)
    best_val_acc = best_mean_mm_val_acc = 0.0
    for i, e in zip(range(sum(iteration)), [item for sublist in [[i]*a for i,a in enumerate(steps_list)] for item in sublist]):
        # - Get training data
        (X,y) = data_loader.get_batch("train")

        if(X.shape[0] == 0):
            continue

        def get_grads(params):
            grads = compute_gradients(X, y, params, cnn, FLAGS, cnn._rng_key, e)
            return grads

        if(FLAGS.optimizer == "esgd"):
            opt_state = opt_update(i, get_params(opt_state), opt_state, get_grads)
        elif(FLAGS.optimizer == "abcd"):
            opt_state = opt_update(i, get_params(opt_state), opt_state, get_grads, FLAGS, cnn._rng_key)
        else:
            opt_state = compute_gradient_and_update(i, X, y, opt_state, opt_update, get_params, cnn, FLAGS, cnn._rng_key, e)
        cnn._rng_key, _ = random.split(cnn._rng_key)

        if((i+1) % 10 == 0):
            params = get_params(opt_state)
            training_accuracy, attacked_training_accuracy, loss_over_time, loss = _get_acc_batch(X, y, params, FLAGS, ATTACK=True)
            elapsed_time = float(onp.round(100 * (time.time() - t0) / 3600) / 100)
            print(f"{elapsed_time} h Epoch {data_loader.n_epochs} i {i} Loss is {loss} Lipschitzness loss over time {loss_over_time} Accuracy {training_accuracy} Attacked accuracy {attacked_training_accuracy}",flush=True)
            log(FLAGS.session_id,"training_accuracy",training_accuracy)
            log(FLAGS.session_id,"attacked_training_accuracy",attacked_training_accuracy)
            log(FLAGS.session_id,"kl_over_time",loss_over_time)

        if((i+1) % (2*FLAGS.eval_step_interval) == 0):
            params = get_params(opt_state)
            mismatch_accuracies = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_get_mismatch_data, vars(FLAGS), params, 0.3, FLAGS.data_dir, "val") for i in range(50)]
                for future in as_completed(futures):
                    mismatch_accuracies.append(future.result())
            mismatch_accuracies = onp.array(mismatch_accuracies, dtype=onp.float64)
            
            mean_mm_val_acc = onp.mean(mismatch_accuracies)
            log(FLAGS.session_id,"mm_val_robustness",list(mismatch_accuracies))
            print(f"Epoch {data_loader.n_epochs} i {i} MM robustness @0.3 {mean_mm_val_acc}+-{onp.std(mismatch_accuracies)}")
            

        if((i) % FLAGS.eval_step_interval == 0):
            params = get_params(opt_state)
            val_acc, attacked_val_acc, loss_over_time, loss = get_val_acc(vars(FLAGS), params, FLAGS.data_dir, ATTACK=True)
            log(FLAGS.session_id,"validation_accuracy",val_acc)
            log(FLAGS.session_id,"attacked_validation_accuracies",attacked_val_acc)
            log(FLAGS.session_id,"validation_kl_over_time",list(loss_over_time))
            print(f"Epoch {data_loader.n_epochs} i {i} Validation accuracy {val_acc} Attacked val. accuracy {attacked_val_acc}")
            if(val_acc > best_val_acc):
                best_val_acc = val_acc
                cnn.save(model_save_path, params)
                print(f"Saved model under {model_save_path}")