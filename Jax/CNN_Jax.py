from jax import lax 
from jax import jit
import numpy as onp
import jax.numpy as jnp
from jax import random as rand
import ujson as json
from jax.nn import relu, normalize
from jax.experimental.stax import BatchNorm

class CNN:
    def __init__(self,params):
        # - Create variable from numpy array
        self.model_settings = params
        # Parameters
        self._rng_key = rand.PRNGKey(params["seed"])

    def call(self,
                input,
                K1,
                CB1,
                K2,
                CB2,
                W1,
                W2,
                W3,
                B1,
                B2,
                B3):

        # - Initial state
        cnn_out = _evolve_CNN(K1, CB1, K2, CB2, W1, W2, W3, B1, B2, B3, input)
        return cnn_out


    # - TODO needs verification
    def save(self,fn,theta):
        save_dict = {}
        save_dict["params"] = self.model_settings
        save_dict["rng_key"] = onp.array(list(self._rng_key),onp.int64).tolist()
        for key in theta.keys():
            if(not type(theta[key]) is list):
                theta[key] = theta[key].tolist()
        save_dict["theta"] = theta
        with open(fn, "w") as f:
            json.dump(save_dict, f)

    @classmethod
    def load(self,fn):
        with open(fn, "r") as f:
            load_dict = json.load(f)
        cnn = CNN(load_dict["params"])
        cnn._rng_key = jnp.array(load_dict["rng_key"], jnp.uint32)
        theta = load_dict["theta"]
        for key in theta.keys():
            theta[key] = jnp.array(theta[key], jnp.float32)
        return cnn, load_dict["theta"] 

@jit
def _evolve_CNN(K1,
                CB1,
                K2,
                CB2,
                W1,
                W2,
                W3,
                B1,
                B2,
                B3,
                P_input):

    # - https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    def MaxPool(mat,ksize=(2,2),method='max',pad=False):
        m, n = mat.shape[2:]
        ky,kx=ksize
        _ceil=lambda x,y: int(jnp.ceil(x/float(y)))
        if pad:
            ny=_ceil(m,ky)
            nx=_ceil(n,kx)
            size=(ny*ky, nx*kx)+mat.shape[2:]
            mat_pad=jnp.full(size,onp.nan)
            mat_pad[:m,:n,...]=mat
        else:
            ny=m//ky
            nx=n//kx
            mat_pad=mat[:, :, :ny*ky, :nx*kx]
        new_shape= mat.shape[:2] + (ny,ky,nx,kx)
        if method=='max':
            result=jnp.nanmax(mat_pad.reshape(new_shape),axis=(3,5))
        else:
            result=jnp.nanmean(mat_pad.reshape(new_shape),axis=(3,5))
        return result

    batch_size = P_input.shape[0]

    x = normalize(P_input, axis=(0,2,3))
    strides = (1,1)
    x = lax.conv_general_dilated(x, K1, strides, padding = [(2,1),(2,1)]) + CB1 #'SAME'
    x = relu(x)
    x = MaxPool(x)
    x = lax.conv_general_dilated(x, K2, strides, padding = [(0,0),(0,0)]) + CB2 #'VALID'
    x = relu(x)
    x = MaxPool(x)
    x = x.reshape(batch_size,-1)
    x = x @ W1 + B1
    x = relu(x)
    x = x @ W2 + B2
    x = relu(x)
    x = normalize(x, axis=0)
    x = x @ W3 + B3
    return x, jnp.array([[0]])