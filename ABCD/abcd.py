from jax.experimental.optimizers import make_schedule
import jax.numpy as jnp
import numpy as np
from copy import deepcopy
from jax import random


def ABCD_Jax(step_size,config):

    step_size = make_schedule(step_size)

    def init(w0):
        state = {}
        state['params'] = deepcopy(w0)
        state['m'] = {}
        state['v'] = {}
        for key in w0:
            state['m'][key] = jnp.zeros_like(w0[key])
            state['v'][key] = jnp.zeros_like(w0[key])
        state['t'] = 0
        return state

    def update(batch_id, params, state, get_grads, FLAGS, rand_key):
        """ get_grads: Theta -> PyTree Gradients, batch_id must start from 0 """        
        for l in range(config["L"]):
            # - Mask
            M_a = {}
            M_b = {}
            for key in params:
                rand_key, subkey = random.split(rand_key)
                M_a[key] = jnp.round(random.uniform(subkey, shape=params[key].shape))
                M_b[key] = M_a[key]*-1 + 1

            grads_w_l_1 = get_grads(params)

            w_l_0_5 = {}
            for key in params:
                g = grads_w_l_1[key]
                if key == "W_rec":
                    diag_indices = jnp.arange(0,g.shape[0],1)
                    g = g.at[diag_indices,diag_indices].set(0.0)
                w_l_0_5[key] = params[key] + FLAGS.abcd_etaA * M_a[key] * g
                # - Clip between [w-|w|attack,w+|w|attack]
                w_l_0_5[key] = jnp.clip(w_l_0_5[key], w_l_0_5[key]-FLAGS.attack_size_mismatch*jnp.abs(w_l_0_5[key]),w_l_0_5[key]+FLAGS.attack_size_mismatch*jnp.abs(w_l_0_5[key]))

            g_w_l_0_5 = get_grads(w_l_0_5)

            for key in g_w_l_0_5:
                g = M_b[key] * g_w_l_0_5[key]
                if key == "W_rec":
                    diag_indices = jnp.arange(0,g.shape[0],1)
                    g = g.at[diag_indices,diag_indices].set(0.0)
                m = state["m"][key]
                v = state["v"][key]
                m = (1 - config["b1"]) * g + config["b1"] * m
                v = (1 - config["b2"]) * jnp.square(g) + config["b2"] * v       
                mhat = m / (1 - config["b1"] ** (batch_id + 1))
                vhat = v / (1 - config["b2"] ** (batch_id + 1))
                params[key] = w_l_0_5[key] - step_size(batch_id) * mhat / (jnp.sqrt(vhat) + config["eps"])
                state["m"][key] = m
                state["v"][key] = v

        for key in state["params"]:
            state["params"][key] = params[key]
    
        return state

    def get_params(state):
        return state['params']

    return init, update, get_params