import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as onp
import tensorflow_datasets as tfds
from torch_esgd import EntropySGD
from entropy_sgd import EntropySGD_Jax
import random 
from jax import jit, grad
import jax.numpy as jnp
from jax import random as rand
from jax.nn import relu, softmax, one_hot
from copy import deepcopy

def set_seeds():
    random.seed(42)
    onp.random.seed(42)
    torch.manual_seed(42)

def eq(tensor,jax_np):
    return onp.allclose(jax_np,tensor.detach().numpy(),atol=1e-4)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 120, bias=False)
        self.fc2 = nn.Linear(120, 10, bias=False)
        self.sm = nn.Softmax()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sm(x)
        return x

class NetJax:
    def call(self,inp,fc1,fc2):
        x = relu(jnp.matmul(fc1,inp.T))
        x = jnp.matmul(fc2,x)
        x = softmax(x.T)
        return x

set_seeds()

# - Load dataset
batch_size = 2
ds = tfds.load('mnist', split='train', batch_size=batch_size)

# - Initialise net and get params
net_torch = Net()
net_jax = NetJax()
theta = {"fc1":deepcopy(net_torch.fc1.weight.detach().numpy()), "fc2":deepcopy(net_torch.fc2.weight.detach().numpy())}

optimizer = EntropySGD(net_torch.parameters(),
        config = dict(lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0,
        L=10, eps=0.0, g0=1e-2, g1=1e-3))
config = dict(momentum=0.9, damp=0.0, nesterov=True, weight_decay=0.0, L=10, eps=0.0, g0=1e-2, g1=1e-3, langevin_lr=0.1, langevin_beta1=0.75)
opt_init, opt_update, get_params = EntropySGD_Jax(0.01, config)
opt_state = opt_init(theta)

# @jit
# def cce_jax(y, logits):
#     return -jnp.mean(jnp.take_along_axis(jnp.log(logits), jnp.expand_dims(y, axis=1), axis=1))

@jit
def l2(y, logits):
    y_oh = one_hot(y, num_classes=10)
    return jnp.mean((y_oh-logits)**2)

# loss = nn.CrossEntropyLoss()
loss = nn.MSELoss()

for batch_id,example in enumerate(tfds.as_numpy(ds)):
    X_np = onp.array(onp.reshape(example["image"], newshape=(batch_size, 28**2)), dtype=onp.float32)
    y_np = example["label"]
    X = torch.tensor(torch.from_numpy(X_np), dtype=torch.float32)
    # y = torch.tensor(y_np) # - Use for CCE
    y = torch.tensor(onp.array(one_hot(y_np, num_classes=10)))

    output = net_torch(X)
    output_jax = net_jax.call(X_np, **theta)
    
    loss_jax = l2(y_np, output_jax)
    loss_torch = loss(output, y)
    # loss_jax = cce_jax(y_np, output_jax)


    def helper():
        def feval():
            optimizer.zero_grad()
            output = net_torch(X)
            f = loss(output, y)
            f.backward()
            return (f.data, 0.1)
        return feval

    def get_grads(theta):
        def training_loss(X,y,theta):
            logits = net_jax.call(X, **theta)
            return l2(y, logits)
            # return cce_jax(y, logits)
        grads = grad(training_loss, argnums=2)(X_np, y_np, theta)
        return grads

    f, err = optimizer.step(helper(), net_torch, loss)
    opt_state = opt_update(batch_id, theta, opt_state, get_grads)
    theta = get_params(opt_state)
    print("Loss jax", loss_jax, "Loss torch", loss_torch.detach().numpy())
    print("Same output:",eq(output,output_jax),"Same loss",eq(f,loss_jax))