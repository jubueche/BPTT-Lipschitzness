from jax import vmap, jit, custom_gradient, random, partial
from jax.lax import scan
import numpy as onp
import jax.numpy as jnp
import ujson as json
import jax
import os
import errno
from copy import deepcopy

from interval_bound_propagation_utils import *

class RNN:

    def __init__(self,params):

        self.units = params["n_hidden"]
        self.dt = params["dt"]
        if jnp.isscalar(params["tau"]): tau = jnp.ones(self.units) * jnp.mean(params["tau"])
        if jnp.isscalar(params["thr"]): thr = jnp.ones(self.units) * jnp.mean(params["thr"])
        # - Create variable from numpy array
        self.tau = jnp.reshape(tau, (1,-1))
        self.thr = jnp.reshape(thr, (1,-1))
        self.dampening_factor = params["dampening_factor"]
        # Parameters
        self.n_refractory = float(params["refr"])
        self.n_in = params['fingerprint_width']
        self.decay = jnp.reshape(jnp.exp(-1*self.dt / self.tau), (1,-1))
        self.tau_adaptation = float(params['spectrogram_length'] * params['in_repeat'])
        self.beta = jnp.ones(shape=(1,self.units)) * params["beta"]
        self.min_beta = jnp.min(params["beta"])
        self.decay_b = jnp.reshape(jnp.exp(-1*self.dt / params["tau_adaptation"]), (1,-1))
        self.thr_min = params["thr_min"]
        self.model_settings = params
        self.noise_std = 0.0
        self._rng_key = random.PRNGKey(params["seed"])
        self.dropout_prob = params["dropout_prob"]

    @partial(jit, static_argnums=(0,2))
    def split_and_get_dropout_mask(self, rand_key, dropout_prob):
        key, subkey = random.split(rand_key)
        val = (random.uniform(subkey, shape=(1,self.units)) > dropout_prob).astype(jnp.float32)
        return key, val

    def unmasked(self):
        return jnp.ones(shape=(1,self.units))

    def call(self,
                fingerprint_input,
                dropout_mask,
                W_in,
                W_rec,
                W_out,
                b_out):
        
        input_frequency_size = self.model_settings['fingerprint_width']
        input_channels = jnp.max(jnp.array([1, 2*self.model_settings['n_thr_spikes'] - 1]))
        #input_channels = max(1,2*self.model_settings['n_thr_spikes'] - 1)
        input_time_size = self.model_settings['spectrogram_length'] * self.model_settings['in_repeat']
        fingerprint_3d = jnp.reshape(fingerprint_input, (-1, input_time_size, input_frequency_size * input_channels)) # - [T,BS,In]
        fingerprint_3d = jnp.transpose(fingerprint_3d, axes=(1,0,2))
        bs = fingerprint_3d.shape[1]
        # - Initial state
        state0_b = {"_v":jnp.zeros((bs, self.units)), "_z":jnp.zeros((bs, self.units)), "_b":jnp.zeros((bs, self.units)), "_r":jnp.zeros((bs,self.units))}
        rnn_out, spikes = _evolve_RNN(state0_b, W_in, W_rec, W_out, b_out, self.tau, self.thr, self.dampening_factor, self.n_refractory, self.decay, self.tau_adaptation, self.beta, self.decay_b, self.dt, self.noise_std, fingerprint_3d, dropout_mask)
        return rnn_out, spikes

    def call_verbose(self,
                fingerprint_input,
                dropout_mask,
                W_in,
                W_rec,
                W_out,
                b_out):
        
        input_frequency_size = self.model_settings['fingerprint_width']
        input_channels = jnp.max(jnp.array([1, 2*self.model_settings['n_thr_spikes'] - 1]))
        #input_channels = max(1,2*self.model_settings['n_thr_spikes'] - 1)
        input_time_size = self.model_settings['spectrogram_length'] * self.model_settings['in_repeat']
        fingerprint_3d = jnp.reshape(fingerprint_input, (-1, input_time_size, input_frequency_size * input_channels)) # - [T,BS,In]
        fingerprint_3d = jnp.transpose(fingerprint_3d, axes=(1,0,2))
        bs = fingerprint_3d.shape[1]
        # - Initial state
        state0_b = {"_v":jnp.zeros((bs, self.units)), "_z":jnp.zeros((bs, self.units)), "_b":jnp.zeros((bs, self.units)), "_r":jnp.zeros((bs,self.units))}
        rnn_out, output_dic = _evolve_RNN_verbose(state0_b, W_in, W_rec, W_out, b_out, self.tau, self.thr, self.dampening_factor, self.n_refractory, self.decay, self.tau_adaptation, self.beta, self.decay_b, self.dt, self.noise_std, fingerprint_3d, dropout_mask)
        return rnn_out, output_dic

    def call_verbose_IBP(self,
                fingerprint_input,
                dropout_mask,
                W_in,
                W_rec,
                W_out,
                b_out):
        
        input_frequency_size = self.model_settings['fingerprint_width']
        input_channels = jnp.max(jnp.array([1, 2*self.model_settings['n_thr_spikes'] - 1]))
        #input_channels = max(1,2*self.model_settings['n_thr_spikes'] - 1)
        input_time_size = self.model_settings['spectrogram_length'] * self.model_settings['in_repeat']
        fingerprint_3d = jnp.reshape(fingerprint_input, (-1, input_time_size, input_frequency_size * input_channels)) # - [T,BS,In]
        fingerprint_3d = jnp.transpose(fingerprint_3d, axes=(1,0,2))
        bs = fingerprint_3d.shape[1]
        # - Initial state
        state0_b = {"_v":create_interval(jnp.zeros((bs, self.units))), "_z":create_interval(jnp.zeros((bs, self.units))), "_b":create_interval(jnp.zeros((bs, self.units))), "_r":create_interval(jnp.zeros((bs,self.units)))}
        rnn_out, output_dic = _evolve_RNN_verbose_IBP(
            state0_b,
            W_in, W_rec, W_out, b_out,
            create_interval(self.tau),
            create_interval(self.thr),
            create_interval(self.dampening_factor),
            create_interval(self.n_refractory),
            create_interval(self.decay),
            create_interval(self.tau_adaptation),
            create_interval(self.beta),
            create_interval(self.decay_b),
            create_interval(self.dt),
            create_interval(self.noise_std),
            create_interval(fingerprint_3d),
            create_interval(dropout_mask))
        return rnn_out, output_dic

    def save(self,fn,theta):
        theta_tmp = deepcopy(theta)
        save_dict = {}
        save_dict["params"] = deepcopy(self.model_settings)
        save_dict["rng_key"] = onp.array(list(self._rng_key),onp.int64).tolist()
        for key in theta_tmp.keys():
            if(not type(theta_tmp[key]) is list):
                theta_tmp[key] = theta_tmp[key].tolist()
        save_dict["theta"] = theta_tmp
        if(save_dict["params"]["architecture"] is not None):
            save_dict["params"].pop("architecture")
        if(save_dict["params"]["network"] is not None):
            save_dict["params"].pop("network")
        try:
            os.makedirs(os.path.dirname(fn))
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(fn)):
                pass
            else: raise
        with open(fn, "w+") as f:
            json.dump(save_dict, f)

    @classmethod
    def load(self,fn):
        with open(fn, "r") as f:
            load_dict = json.load(f)
        rnn = RNN(load_dict["params"])
        rnn._rng_key = jnp.array(load_dict["rng_key"], jnp.uint32)
        theta = load_dict["theta"]
        for key in theta.keys():
            theta[key] = jnp.array(theta[key], jnp.float32)
        return rnn, load_dict["theta"] 

@jit
def _evolve_RNN(state0,
                W_in,
                W_rec,
                W_out,
                b_out,
                tau,
                thr,
                dampening_factor,
                n_refractory,
                decay,
                tau_adaptation,
                beta,
                decay_b,
                dt,
                noise_std,
                I_input,
                dropout_mask):

    batch_size = I_input.shape[1]
    T = I_input.shape[0]
    N = W_rec.shape[0]

    static_params = {}
    static_params["W_in"] = W_in
    static_params["W_rec"] = W_rec
    static_params["tau"] = tau
    static_params["thr"] = thr
    static_params["dampening_factor"] = dampening_factor
    static_params["n_refractory"] = n_refractory
    static_params["_decay"] = decay
    static_params["tau_adaptation"] = tau_adaptation
    static_params["_beta"] = beta
    static_params["decay_b"] = decay_b
    static_params["dt"] = dt
    static_params["dropout_mask"] = dropout_mask

    @custom_gradient # z = f(V,DF) is computed here. Given dE/dz, we can compute dE/dV using the chain rule: dE/dz * dz/dV 
    def SpikeFunction(v_scaled):
        z_ = (v_scaled > 0.).astype(jnp.float32)
        def grad(dE_dz):
            dz_dv_scaled = jnp.maximum(1 - jnp.abs(v_scaled), 0)
            dz_dv_scaled *= 0.3
            dE_dv_scaled = dE_dz * dz_dv_scaled
            return (dE_dv_scaled,)
        return z_, grad

    @jit
    def compute_z(v, adaptive_thr, dt):
        v_scaled = (v - adaptive_thr) / adaptive_thr
        z = SpikeFunction(v_scaled)
        z = z * 1 / dt
        return z

    @jit
    def forward(evolve_state, I_input):
        (state, static_params) = evolve_state
        new_b = static_params["decay_b"] * state["_b"] + (jnp.ones(shape=(1,static_params["W_rec"].shape[0])) - static_params["decay_b"]) * state["_z"]
        thr = static_params["thr"] + new_b * static_params["_beta"]
        I_in = I_input @ static_params["W_in"]
        I_rec = state["_z"] @ static_params["W_rec"]
        I_t = I_in + I_rec
        I_reset = state["_z"] * thr * static_params["dt"]
        new_v = static_params["_decay"] * state["_v"] + (1. - static_params["_decay"]) * I_t - I_reset
        not_refractory = (state["_r"] <= .1).astype(jnp.float32)
        new_z = static_params["dropout_mask"] * not_refractory * compute_z(new_v, thr, static_params["dt"])
        new_r = jnp.clip(state["_r"] + static_params["n_refractory"] * new_z - 1., 0., static_params["n_refractory"])
        state["_v"] = new_v ; state["_z"] = new_z ; state["_b"] = new_b ; state["_r"] = new_r
        return (state, static_params), (state["_z"])

    (_,_), (Z) = scan(forward,(state0, static_params),I_input)
    Z = jnp.transpose(Z, axes=(1,0,2))
    rnn_out = jnp.mean(Z, axis=1)
    rnn_out = rnn_out @ W_out + b_out

    return rnn_out, Z


@jit
def _evolve_RNN_verbose(state0,
                W_in,
                W_rec,
                W_out,
                b_out,
                tau,
                thr,
                dampening_factor,
                n_refractory,
                decay,
                tau_adaptation,
                beta,
                decay_b,
                dt,
                noise_std,
                I_input,
                dropout_mask):

    batch_size = I_input.shape[1]
    T = I_input.shape[0]
    N = W_rec.shape[0]

    static_params = {}
    static_params["W_in"] = W_in
    static_params["W_rec"] = W_rec
    static_params["tau"] = tau
    static_params["thr"] = thr
    static_params["dampening_factor"] = dampening_factor
    static_params["n_refractory"] = n_refractory
    static_params["_decay"] = decay
    static_params["tau_adaptation"] = tau_adaptation
    static_params["_beta"] = beta
    static_params["decay_b"] = decay_b
    static_params["dt"] = dt
    static_params["dropout_mask"] = dropout_mask

    @custom_gradient # z = f(V,DF) is computed here. Given dE/dz, we can compute dE/dV using the chain rule: dE/dz * dz/dV 
    def SpikeFunction(v_scaled):
        z_ = (v_scaled > 0.).astype(jnp.float32)
        def grad(dE_dz):
            dz_dv_scaled = jnp.maximum(1 - jnp.abs(v_scaled), 0)
            dz_dv_scaled *= 0.3
            dE_dv_scaled = dE_dz * dz_dv_scaled
            return (dE_dv_scaled,)
        return z_, grad

    @jit
    def compute_z(v, adaptive_thr, dt):
        v_scaled = (v - adaptive_thr) / adaptive_thr
        z = SpikeFunction(v_scaled)
        z = z * 1 / dt
        return z

    @jit
    def forward(evolve_state, I_input):
        (state, static_params) = evolve_state
        new_b = static_params["decay_b"] * state["_b"] + (jnp.ones(shape=(1,static_params["W_rec"].shape[0])) - static_params["decay_b"]) * state["_z"]
        thr = static_params["thr"] + new_b * static_params["_beta"]
        I_in = I_input @ static_params["W_in"]
        I_rec = state["_z"] @ static_params["W_rec"]
        I_t = I_in + I_rec
        I_reset = state["_z"] * thr * static_params["dt"]
        new_v = static_params["_decay"] * state["_v"] + (1. - static_params["_decay"]) * I_t - I_reset
        not_refractory = (state["_r"] <= .1).astype(jnp.float32)
        new_z = static_params["dropout_mask"] * not_refractory * compute_z(new_v, thr, static_params["dt"])
        new_r = jnp.clip(state["_r"] + static_params["n_refractory"] * new_z - 1., 0., static_params["n_refractory"])
        state["_v"] = new_v ; state["_z"] = new_z ; state["_b"] = new_b ; state["_r"] = new_r
        return (state, static_params), (state["_z"],state["_v"],thr)

    (_,_), (Z,V,T) = scan(forward,(state0, static_params),I_input)
    Z = jnp.transpose(Z, axes=(1,0,2))
    V = jnp.transpose(V, axes=(1,0,2))
    T = jnp.transpose(T, axes=(1,0,2))
    output_dic = {"Z":Z, "V":V, "T":T}
    rnn_out = jnp.mean(Z, axis=1)
    rnn_out = rnn_out @ W_out + b_out

    return rnn_out, output_dic

@jit
def _evolve_RNN_verbose_IBP(state0,
                W_in,
                W_rec,
                W_out,
                b_out,
                tau,
                thr,
                dampening_factor,
                n_refractory,
                decay,
                tau_adaptation,
                beta,
                decay_b,
                dt,
                noise_std,
                I_input,
                dropout_mask):

    batch_size = I_input[0].shape[1]
    T = I_input[0].shape[0]
    N = W_rec[0].shape[0]

    static_params = {}
    static_params["W_in"] = W_in
    static_params["W_rec"] = W_rec
    static_params["tau"] = tau
    static_params["thr"] = thr
    static_params["dampening_factor"] = dampening_factor
    static_params["n_refractory"] = n_refractory
    static_params["_decay"] = decay
    static_params["tau_adaptation"] = tau_adaptation
    static_params["_beta"] = beta
    static_params["decay_b"] = decay_b
    static_params["dt"] = dt
    static_params["dropout_mask"] = dropout_mask

    @jit
    def compute_z(v, adaptive_thr, dt):
        v_scaled = div(subtract(v,adaptive_thr), adaptive_thr)
        z = SpikeFunction(v_scaled)
        z = elem_mult(z, div(create_interval(1.),dt))
        return z

    @jit
    def SpikeFunction(v_scaled):
        z_ = greater(v_scaled,create_interval(0.))
        return z_

    @jit
    def forward(evolve_state, I_input):
        (state, static_params) = evolve_state
        x1 = elem_mult(static_params["decay_b"],state["_b"])
        x2 = elem_mult(subtract(create_interval(jnp.ones(shape=(1,static_params["W_rec"][0].shape[0]))), static_params["decay_b"]), state["_z"])
        new_b = add(x1,x2)
        # new_b_cont = static_params["decay_b"][0] * state["_b"][0] + (jnp.ones(shape=(1,static_params["W_rec"][0].shape[0])) - static_params["decay_b"][0]) * state["_z"][0]
        # assert jnp.allclose(new_b[0] , new_b_cont)
        thr = add(static_params["thr"], elem_mult(new_b,static_params["_beta"]))
        # thr_cont = thr_cont = static_params["thr"][0] + new_b_cont * static_params["_beta"][0]
        # assert jnp.allclose(thr[0],thr_cont)
        I_in = mat_mul(I_input, static_params["W_in"])
        # I_in_cont = I_input[0] @ static_params["W_in"][0]
        # assert jnp.allclose(I_in[0] , I_in_cont)
        I_rec = mat_mul(state["_z"],static_params["W_rec"])
        # I_rec_cont = state["_z"][0] @ static_params["W_rec"][0]
        # assert jnp.allclose(I_rec[0],I_rec_cont)
        I_t = add(I_in,I_rec)
        # I_t_cont = I_in_cont + I_rec_cont
        # assert jnp.allclose(I_t[0],I_t_cont)
        I_reset = elem_mult(elem_mult(state["_z"],thr),static_params["dt"])
        # I_reset_cont = state["_z"][0] * thr_cont * static_params["dt"][0]
        # assert jnp.allclose(I_reset[0] , I_reset_cont)
        x1 = elem_mult(static_params["_decay"],state["_v"])
        x2 = elem_mult(subtract(create_interval(1.0),static_params["_decay"]),I_t)
        new_v = subtract(add(x1,x2),I_reset)
        # new_v_cont = static_params["_decay"][0] * state["_v"][0] + (1. - static_params["_decay"][0]) * I_t_cont - I_reset_cont
        # assert jnp.allclose(new_v[0],new_v_cont)
        not_refractory = leq(state["_r"],create_interval(.1))
        # not_refractory_cont = (state["_r"][0] <= .1).astype(jnp.float32)
        # assert jnp.allclose(not_refractory[0],not_refractory_cont)
        new_z = elem_mult(elem_mult(static_params["dropout_mask"],not_refractory), compute_z(new_v, thr, static_params["dt"]))
        # new_z_cont = static_params["dropout_mask"][0] * not_refractory_cont * compute_z_cont(new_v_cont, thr_cont, static_params["dt"][0])
        # assert jnp.allclose(new_z[0],new_z_cont)
        new_r = clip(add(state["_r"],subtract(elem_mult(static_params["n_refractory"],new_z), create_interval(1.))), create_interval(0.), static_params["n_refractory"])
        # new_r_cont = jnp.clip(state["_r"][0] + static_params["n_refractory"][0] * new_z_cont - 1., 0., static_params["n_refractory"][0])
        # assert jnp.allclose(new_r[0] , new_r_cont)
        state["_v"] = new_v ; state["_z"] = new_z ; state["_b"] = new_b ; state["_r"] = new_r
        return (state, static_params), (state["_z"],state["_v"],state["_v"])

    (_,_), (Z,V,T) = scan(forward,(state0, static_params),I_input)
    Z = [jnp.transpose(z, axes=(1,0,2)) for z in Z]
    V = [jnp.transpose(v, axes=(1,0,2)) for v in V]
    T = [jnp.transpose(t, axes=(1,0,2)) for t in T]
    output_dic = {"Z":Z, "V":V, "T":T}
    rnn_out = [jnp.mean(z, axis=1) for z in Z]
    rnn_out = add(mat_mul(rnn_out,W_out),b_out)
    return rnn_out, output_dic