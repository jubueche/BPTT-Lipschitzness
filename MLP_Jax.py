
import ujson as json
import os
import errno
from copy import deepcopy
import numpy.random as npr
from jax import jit, grad, nn, lax
from jax.scipy.special import logsumexp
import jax.numpy as jnp

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = nn.relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)

def loss(params, batch, weight_increase):
    inputs, targets = batch
    preds = predict(params, inputs)
    weight_magnitude = -1.0 * weight_increase * lax.max(0.0,2.0-jnp.sum([jnp.mean(jnp.abs(p[0])) for p in params[:-1]]))
    return -jnp.mean(jnp.sum(preds * targets, axis=1)) - weight_magnitude

def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

@jit
def update(params, batch, weight_increase, step_size):
    grads = grad(loss)(params, batch, weight_increase)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

class MLP:

    def __init__(self,theta):
        self.params = theta

    def save(self,fn):
        theta_tmp = []
        for el in self.params:
            theta_tmp.append([el[0].tolist(),el[1].tolist()])
        save_dict = {}
        save_dict["theta"] = theta_tmp
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
        params = []
        for el in load_dict["theta"]:
            params.append(tuple([jnp.array(e) for e in el]))
        mlp = MLP(params)
        return mlp, params