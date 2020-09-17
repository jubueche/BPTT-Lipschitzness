from jax import jit, grad
import jax.random as rand
import jax.numpy as jnp
import numpy as onp

# - Define function to compute Lipschitzness of network w.r.t. parameters Theta
def lipschitzness(theta, theta_start, net, input_batch_t, output_batch_t):
    # - Apply theta to the network
    params = net._pack()
    params[1]["tau_mem"] = theta["tau_mem"]
    params[1]["tau_syn"] = theta["tau_syn"]
    params[1]["bias"] = theta["bias"]
    # - Evaluate: f(X,Theta*)
    spiking_output, _, _ = net._evolve_functional(params=params, all_states=net._state, ext_inputs=input_batch_t)
    # - Compute the loss
    lip = jnp.mean((output_batch_t - spiking_output)**2) / dict_norm(theta_start,theta)
    return lip

def dict_norm(d1,d2):
    norm = 0.0
    for key in d1.keys():
        norm += jnp.linalg.norm(d1[key]-d2[key]) / jnp.sqrt(jnp.linalg.norm(d1[key])*jnp.linalg.norm(d2[key]))
    return norm

def evolve_using(theta, net, input_batch_t):
    params = net._pack()
    params[1]["tau_mem"] = theta["tau_mem"]
    params[1]["tau_syn"] = theta["tau_syn"]
    params[1]["bias"] = theta["bias"]
    # - Evaluate: f(X,Theta*)
    spiking_output, _, _ = net._evolve_functional(params=params, all_states=net._state, ext_inputs=input_batch_t)
    return spiking_output

def loss_lipschitzness_verbose(
            params,
            states_t,
            input_batch_t,
            output_batch_t,
            target_batch_t,
            min_tau,
            net,
            lambda_mse,
            reg_tau,
            reg_l2_rec,
            reg_act1,
            reg_act2,
    ):  
        lyrIn = params[0]
        lyrRes = params[1]
        lyrRO = params[2]

        # - Lipschitzness loss
        theta_start = {"tau_mem": lyrRes["tau_mem"], "tau_syn": lyrRes["tau_syn"], "bias": lyrRes["bias"]}

        step_size = 0.005
        number_steps = 5
        beta = 1.0
        initial_std = 0.05
        key = rand.PRNGKey(jnp.sum(input_batch_t).astype(int))
        _, *sks = rand.split(key, 7)

        theta_random = {}
        theta_random["bias"] =  theta_start["bias"] + theta_start["bias"]*initial_std*rand.normal(key = sks[0])
        theta_random["tau_syn"] = jnp.abs(theta_start["tau_syn"] + theta_start["tau_syn"]*initial_std*rand.normal(key = sks[1]))
        theta_random["tau_mem"] = jnp.abs(theta_start["tau_mem"] + theta_start["tau_mem"]*initial_std*rand.normal(key = sks[2]))

        theta_star = {}
        theta_star["bias"] =  theta_start["bias"]  + theta_start["bias"]*initial_std*rand.normal(key = sks[3])
        theta_star["tau_syn"] = jnp.abs(theta_start["tau_syn"] + theta_start["tau_syn"]*initial_std*rand.normal(key = sks[4]))
        theta_star["tau_mem"] = jnp.abs(theta_start["tau_mem"] + theta_start["tau_mem"]*initial_std*rand.normal(key = sks[5]))
        
        lipschitzness_over_time = []
        theta_start_star_distance = []
        batched_output_over_time = []

        batched_output_over_time.append(evolve_using(theta_random, net, input_batch_t))
        for _ in range(number_steps):
            # - Evaluate Lipschitzness for each Theta
            lipschitzness_over_time.append(lipschitzness(theta_star, theta_start, net, input_batch_t, output_batch_t))
            theta_start_star_distance.append(dict_norm(theta_star,theta_start))
            # - Compute the gradient w.r.t. theta
            theta_grad = grad(lipschitzness, argnums=0)(theta_star, theta_start, net, input_batch_t, output_batch_t)
            batched_output_over_time.append(evolve_using(theta_star, net, input_batch_t))

            # - Update theta_star
            for key in theta_star.keys():
                # - Normalize gradient and do update
                theta_star[key] = theta_star[key] + step_size*theta_grad[key] / jnp.linalg.norm(theta_grad[key])
                # - Compute deviation and clamp to
                # TODO

        lipschitzness_over_time.append(lipschitzness(theta_star, theta_start, net, input_batch_t, output_batch_t))
        theta_start_star_distance.append(dict_norm(theta_star,theta_start))
        lipschitzness_random = lipschitzness(theta_random, theta_start, net, input_batch_t, output_batch_t)

        loss_lip = beta*lipschitzness(theta_star, theta_start, net, input_batch_t, output_batch_t)

        fLoss_reg, loss_dict_reg = loss_mse_reg_stack(params=params,
                                                    states_t=states_t,
                                                    input_batch_t=input_batch_t,
                                                    output_batch_t=output_batch_t,
                                                    target_batch_t=target_batch_t,
                                                    min_tau=min_tau,
                                                    lambda_mse=lambda_mse,
                                                    reg_tau=reg_tau,
                                                    reg_l2_rec=reg_l2_rec,
                                                    reg_act1=reg_act1,
                                                    reg_act2=reg_act2)
        
        # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
        fLoss = fLoss_reg + loss_lip

        loss_dict = {}
        loss_dict["fLoss"] = fLoss
        loss_dict["mse"] = loss_dict_reg["mse"]
        loss_dict["tau_loss"] = loss_dict_reg["tau_loss"]
        loss_dict["w_res_norm"] = loss_dict_reg["w_res_norm"]
        loss_dict["loss_lip"] = loss_lip
        loss_dict["lipschitzness_over_time"] = lipschitzness_over_time
        loss_dict["theta_start_star_distance"] = theta_start_star_distance
        loss_dict["theta_start"] = theta_start
        loss_dict["theta_star"] = theta_star
        loss_dict["lipschitzness_random"] = lipschitzness_random
        loss_dict["random_distance_to_start"] = dict_norm(theta_start, theta_random)
        loss_dict["batched_output_over_time"] = batched_output_over_time
        loss_dict["output_batch_t"] = output_batch_t

        # - Return loss
        return (fLoss, loss_dict)


def loss_lipschitzness(
            params,
            states_t,
            input_batch_t,
            output_batch_t,
            target_batch_t,
            min_tau,
            net,
            lambda_mse,
            reg_tau,
            reg_l2_rec,
            reg_act1,
            reg_act2,
    ):  
        lyrIn = params[0]
        lyrRes = params[1]
        lyrRO = params[2]

        # - Lipschitzness loss
        theta_start = {"tau_mem": lyrRes["tau_mem"], "tau_syn": lyrRes["tau_syn"], "bias": lyrRes["bias"]}
        
        step_size = 0.005
        number_steps = 5
        beta = 1.0
        initial_std = 0.05
        key = rand.PRNGKey(jnp.sum(input_batch_t).astype(int))
        _, *sks = rand.split(key, 4)

        theta_star = {}
        theta_star["bias"] =  theta_start["bias"]  + theta_start["bias"]*initial_std*rand.normal(key = sks[0])
        theta_star["tau_syn"] = jnp.abs(theta_start["tau_syn"] + theta_start["tau_syn"]*initial_std*rand.normal(key = sks[1]))
        theta_star["tau_mem"] = jnp.abs(theta_start["tau_mem"] + theta_start["tau_mem"]*initial_std*rand.normal(key = sks[2]))

        for _ in range(number_steps):
            # - Compute the gradient w.r.t. theta
            theta_grad = grad(lipschitzness, argnums=0)(theta_star, theta_start, net, input_batch_t, output_batch_t)
            # - Update theta_star
            for key in theta_star.keys():
                # - Normalize gradient and do update
                theta_star[key] = theta_star[key] + step_size*theta_grad[key] / jnp.linalg.norm(theta_grad[key])
                # - Compute deviation and clamp to
                # TODO

        loss_lip = beta*lipschitzness(theta_star, theta_start, net, input_batch_t, output_batch_t)

        fLoss_reg, loss_dict_reg = loss_mse_reg_stack(params=params,
                                                    states_t=states_t,
                                                    input_batch_t=input_batch_t,
                                                    output_batch_t=output_batch_t,
                                                    target_batch_t=target_batch_t,
                                                    min_tau=min_tau,
                                                    lambda_mse=lambda_mse,
                                                    reg_tau=reg_tau,
                                                    reg_l2_rec=reg_l2_rec,
                                                    reg_act1=reg_act1,
                                                    reg_act2=reg_act2)
        
        # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
        fLoss = fLoss_reg + loss_lip

        loss_dict = {}
        loss_dict["fLoss"] = fLoss
        loss_dict["mse"] = loss_dict_reg["mse"]
        loss_dict["tau_loss"] = loss_dict_reg["tau_loss"]
        loss_dict["w_res_norm"] = loss_dict_reg["w_res_norm"]
        loss_dict["loss_lip"] = loss_lip

        # - Return loss
        return (fLoss, loss_dict)


@jit
def loss_mse_reg_stack(
            params,
            states_t,
            input_batch_t,
            output_batch_t,
            target_batch_t,
            min_tau,
            lambda_mse,
            reg_tau,
            reg_l2_rec,
            reg_act1,
            reg_act2,
    ):
        # - Measure output-target loss
        mse = lambda_mse * jnp.mean((output_batch_t - target_batch_t) ** 2)

        # - Get loss for tau parameter constraints
        # - Measure recurrent L2 norms
        tau_loss = 0.0
        w_res_norm = 0.0
        act_loss = 0.0
        
        lyrIn = params[0]
        lyrRes = params[1]
        lyrRO = params[2]
        
        taus = jnp.concatenate((
            lyrIn["tau_mem"],
            lyrIn["tau_syn"],
            lyrRes["tau_mem"],
            lyrRes["tau_syn"],
            lyrRO["tau_syn"],
        ))
        
        tau_loss += reg_tau * jnp.mean(
            jnp.where(
                taus < min_tau,
                jnp.exp(-(taus - min_tau)),
                0,
            )
        )
        
        w_res_norm += reg_l2_rec * jnp.mean(lyrIn["w_in"] ** 2)    
        w_res_norm += reg_l2_rec * jnp.mean(lyrRes["w_recurrent"] ** 2)

        act_loss += reg_act1 * jnp.mean(states_t[1]["surrogate"])
        act_loss += reg_act2 * jnp.mean(states_t[1]["Vmem"] ** 2)
        
        # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
        fLoss = mse + tau_loss + w_res_norm + act_loss

        loss_dict = {}
        loss_dict["fLoss"] = fLoss
        loss_dict["mse"] = mse
        loss_dict["tau_loss"] = tau_loss
        loss_dict["w_res_norm"] = w_res_norm

        # - Return loss
        return (fLoss, loss_dict)