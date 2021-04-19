from jax.experimental.optimizers import make_schedule
import jax.numpy as jnp
import numpy as np
from copy import deepcopy, copy


def EntropySGD_Jax(step_size, config):

    step_size = make_schedule(step_size)

    def init(w0):
        state = {}
        state['params'] = deepcopy(w0)
        state['t'] = 0
        state['wc'] = {}
        state['mdw'] = {}
        state['m'] = {}
        state['v'] = {}
        for key in w0:
            state['wc'][key] = jnp.zeros_like(w0[key])
            state['mdw'][key] = jnp.zeros_like(w0[key])
            state['m'][key] = jnp.zeros_like(w0[key])
            state['v'][key] = jnp.zeros_like(w0[key])

        state['langevin'] = {'mw': deepcopy(state['wc']), 'mdw': deepcopy(state['mdw']), 'eta': deepcopy(
            state['mdw']), 'lr': config["langevin_lr"], 'beta1': config["langevin_beta1"]}
        return state

    def update(batch_id, params, state, get_grads):
        """ get_grads: Theta -> PyTree Gradients, batch_id must start from 0 """
        grads = get_grads(params)
        if(batch_id == 0):
            for key in grads:
                state['mdw'][key] = deepcopy(grads[key])
            state['langevin']['mdw'] = deepcopy(state['mdw'])
            state['langevin']['eta'] = deepcopy(state['mdw'])

        for key in params:
            state['wc'][key] = copy(params[key])
            state['langevin']['mw'][key] = copy(params[key])
            state['langevin']['mdw'][key] = jnp.zeros_like(params[key])
            state['langevin']['eta'][key] = np.random.normal(
                loc=0.0, scale=1.0, size=params[key].shape)

        g = config["g0"]*(1+config["g1"])**state['t']
        for i in range(config["L"]):
            grads = get_grads(params)
            for key in params:
                dw = grads[key]
                if(config["weight_decay"] > 0):
                    dw += config["weight_decay"] * params[key]
                if(config["momentum"] > 0):
                    state['langevin']['mdw'][key] = config["momentum"] * \
                        state['langevin']['mdw'][key] + \
                        (1-config["damp"])*grads[key]
                    if(config["nesterov"]):
                        grads[key] += config["momentum"] * \
                            state['langevin']['mdw'][key]
                    else:
                        grads[key] = state['langevin']['mdw'][key]

                state['langevin']['eta'][key] = np.random.normal(
                    loc=0.0, scale=1.0, size=params[key].shape)
                grads[key] += -g*(state['wc'][key] - params[key]) + config["eps"] / \
                    np.sqrt(0.5*config["langevin_lr"]) * \
                    state['langevin']['eta'][key]
                params[key] += -config["langevin_lr"]*grads[key]
                state['langevin']['mw'][key] = config["langevin_beta1"] * \
                    state['langevin']['mw'][key] + \
                    (1-config["langevin_beta1"]) * params[key]

        # - End Langevin for loop
        if(config["L"] > 0):
            for key in params:
                params[key] = copy(state['wc'][key])
                grads[key] = copy(params[key] - state['langevin']['mw'][key])

        for key in params:
            if(config["weight_decay"] > 0):
                grads[key] += config["weight_decay"]*params[key]
            if(config["momentum"] > 0):
                state['mdw'][key] = config["momentum"] * \
                    state['mdw'][key] + (1-config["damp"]) * grads[key]
                if(config["nesterov"]):
                    grads[key] += config["momentum"] * state['mdw'][key]
                else:
                    grads[key] = state['mdw'][key]
            
            g = grads[key]
            if key == "W_rec":
                np.fill_diagonal(g, 0.0)
            m = state["m"][key]
            v = state["v"][key]
            m = (1 - config["b1"]) * g + config["b1"] * m
            v = (1 - config["b2"]) * \
            		jnp.square(g) + config["b2"] * v
            mhat = m / (1 - config["b1"] ** (batch_id + 1))
            vhat = (v + config["eps_adam"]) / (1 - config["b2"] ** (batch_id + 1)  + config["eps_adam"])
            params[key] += -step_size(batch_id) * \
            mhat / (jnp.sqrt(vhat) + config["eps_adam"])
            state["m"][key] = m
            state["v"][key] = v
            state['params'][key] = params[key]

        return state

    def get_params(state):
        return state['params']

    return init, update, get_params