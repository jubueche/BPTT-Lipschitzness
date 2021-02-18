from jax.experimental.optimizers import optimizer, make_schedule
import jax.numpy as jnp
import numpy as np

@optimizer
def entropy_sgd(lr=0.01, momentum=0, damp=0,
                 weight_decay=0, nesterov=True,
                 L=1, eps=1e-4, gamma=1e-2, langevin_lr=0.1, langevin_beta1=0.75):

  def init(x0):
    return {"x": x0, "xd":x0, "mu":x0, 't':0}
  def update(i, g, state):
    dxd = g - gamma * (state["x"] - state["xd"])
    state["xd"] = state["xd"] - langevin_lr*dxd + jnp.sqrt(langevin_lr)*eps*np.random.normal(0,1)
    state["mu"] = (1-langevin_beta1*state["mu"]) + langevin_beta1 * state["xd"]
    if (state["t"]+1)%L ==0:
        state["x"] = state["xd"]=state["mu"]=x -lr*gamma(state["x"] - state["mu"])
    state["t"]+=1
    return state
  def get_params(x):
    return x
  return init, update, get_params

