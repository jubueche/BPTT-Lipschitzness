from jax import lax 
from jax import vmap, jit, custom_gradient
from jax.lax import scan
import numpy as onp
import skimage.measure as skm
import jax.numpy as jnp
from jax import random as rand
import ujson as json
import jax
from import_data import extract
import jax.experimental.stax as stax
from jax.nn import relu, normalize, softmax


class CNN:

    def __init__(self,params):

        self.units = params["n_hidden"]
        # - Create variable from numpy array
        self.model_settings = params
        # Parameters
        self._rng_key = rand.PRNGKey(params["seed"])

    def call(self,
                input,
                K1,
                K2,
                W1,
                W2,
                W3):

        _, self._rng_key = rand.split(self._rng_key)
        # - Initial state
        cnn_out = _evolve_RNN(K1, K2, W1, W2, W3, input, self._rng_key)
        return cnn_out

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
        rnn = RNN(load_dict["params"])
        rnn._rng_key = jnp.array(load_dict["rng_key"], jnp.uint32)
        theta = load_dict["theta"]
        for key in theta.keys():
            theta[key] = jnp.array(theta[key], jnp.float32)
        return rnn, load_dict["theta"] 

@jit
def _evolve_RNN(K1,
                K2,
                W1,
                W2,
                W3,
                #noise_std,
                P_input,
                key):

    batch_size = P_input.shape[1]

    _, subkey = rand.split(key)
    #noise_ts = noise_std * rand.normal(subkey, shape=(T, batch_size, N))

    static_params = {}
    static_params["K1"] = K1
    static_params["K2"] = K2
    static_params["W1"] = W1
    static_params["W2"] = W2
    static_params["W3"] = W3

    def MaxPool(frame, pool_size=(1,2,2)):
        return(skm.block_reduce(frame, pool_size, onp.max))

    def MaxPool_raw(mat,ksize=(2,2),method='max',pad=False):  # CHECK https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
        '''Non-overlapping pooling on 2D or 3D data.
        <mat>: ndarray, input array to pool.
        <ksize>: tuple of 2, kernel size in (ky, kx).
        <method>: str, 'max for max-pooling, 
                    'mean' for mean-pooling.
        <pad>: bool, pad <mat> or not. If no pad, output has size
            n//f, n being <mat> size, f being kernel size.
            if pad, output has size ceil(n/f).
        Return <result>: pooled matrix.
        '''
        m, n = mat.shape[:2]
        ky,kx=ksize
        _ceil=lambda x,y: int(numpy.ceil(x/float(y)))

        if pad:
            ny=_ceil(m,ky)
            nx=_ceil(n,kx)
            size=(ny*ky, nx*kx)+mat.shape[2:]
            mat_pad=numpy.full(size,numpy.nan)
            mat_pad[:m,:n,...]=mat
        else:
            ny=m//ky
            nx=n//kx
            mat_pad=mat[:ny*ky, :nx*kx, ...]

        new_shape=(ny,ky,nx,kx)+mat.shape[2:]

        if method=='max':
            result=numpy.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
        else:
            result=numpy.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

        return result

    def Dropout(frame, drop_prob):

        binary_value = np.random.rand(frame.shape[1], frame.shape[2]) < drop_prob
        res = np.multiply(frame, binary_value)
        res /= (1-drop_prob)  # this line is called inverted dropout technique
        print(res)



    x = normalize(P_input) #check if this is a problem

    x = lax.conv_general_dilated(x, static_params["K1"], padding = 'SAME') #x must be (batch_size, 1, R,C) with (R,C) dimension of 1 frame and K1 must be (64,1,R,C)
    x = relu(x)

    x = MaxPool(x)

    x = Dropout(x, 0.1)

    x = lax.conv_general_dilated(x, static_params["K2"], padding = 'VALID')
    x = relu(x)

    x = MaxPool(x)

    x = Dropout(x, 0.3)

    x = jnp.ravel(x)

    x = x @ static_params["W1"] #W2 = 256xdim jnp.ravel x batch
    x = relu(x)

    x = Dropout(x, 0.5)

    x = x @ static_params["W2"] #W3 = 64x256
    x = relu(x)

    x = normalize(x)

    x = x @ static_params["W3"] #W3 = Numclass x 64
    x = softmax(x)


    #Maxpool will need to be implemented manually

    #CHECK IF REQUIRES INTERMEDIATE STATES TO BE REGISTERED SOMEHOW



    return x


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = extract()

    cnn = CNN(model_settings)
    logits, outputs = cnn.call(X_train, W, b)