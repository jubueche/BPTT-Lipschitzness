#%%
import numpy as onp
from jax import config
config.FLAGS.jax_log_compiles=True
import jax.numpy as jnp
from jax import random, grad, value_and_grad, jit, partial, ops
from copy import copy, deepcopy
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import time

#%%

onp.random.seed(42)

def f(theta):
    x = theta[0]; y = theta[1]
    return _f(x,y)

# def _f(x,y):
#     return 3*(1-x)**2*jnp.exp(-(x**2)-(y+1)**2)-10*(x/5 - x**3 - y**5)*jnp.exp(-x**2-y**2) - 1/3*jnp.exp(-(x+1)**2 - y**2)

@jit
def _f(x,y):
    return jnp.sin(x**2 + y**2)

    # GeoGebra: 3*(1-x)^2*exp(-(x^2)-(y+1)^2)-10*(x/5 - x^3 - y^5)*exp(-x^2-y^2) - 1/3*exp(-(x+1)^2 - y^2)

def split_and_sample(key, shape):
    key, subkey = random.split(key)
    val = random.normal(subkey, shape=shape)
    return key, val

def robust_loss(theta, theta_star, num_steps, eps_ball):
    if(theta_star == None):
        return _robust_loss(theta, num_steps, eps_ball)
    else:
        return lip_loss(theta,theta_star)

@partial(jit, static_argnums=(1))
def _robust_loss(theta, num_steps, eps_ball):
    theta_star = make_theta_star(theta, num_steps, eps_ball)
    return lip_loss(theta,theta_star)

@partial(jit, static_argnums=(1))
def make_theta_star(theta, num_steps, eps_ball): 
    theta_star = theta + 0.001*onp.random.normal(loc=0.0, scale=1.0, size=theta.shape)
    step_size = eps_ball / num_steps
    for i in range(num_steps):
        grads_theta_star = grad(lip_loss, argnums=0)(theta_star, theta)
        theta_star = theta_star + step_size[i] * jnp.sign(grads_theta_star)
    return theta_star

def lip_loss(theta, theta_star):
    return 0.5*(f(theta)-f(theta_star))**2

def compute_gradients(theta, num_steps, eps_ball, treat_as_constant=False):
    theta_star = None
    if(treat_as_constant):
        theta_star = make_theta_star(theta, num_steps, eps_ball)

    def loss_general(theta):
        loss = f(theta) + robust_loss(theta, theta_star, num_steps, eps_ball)
        return loss

    value,grads = value_and_grad(loss_general, argnums=0)(theta)
    return value,grads

def compute_gradients_rob(theta, num_steps, eps_ball, treat_as_constant):
    theta_star = None
    if(treat_as_constant):
        theta_star = make_theta_star(theta, num_steps, eps_ball)
    value, grads = value_and_grad(robust_loss, argnums=0)(theta, theta_star, num_steps, eps_ball)
    return value, grads

def vec2xy(theta):
    return theta[0], theta[1]

def xy2vec(x,y):
    return jnp.array([x,y])

def get_surf(lim=6):
    X = onp.arange(-lim, lim, 0.25)
    Y = onp.arange(-lim, lim, 0.25)
    X, Y = onp.meshgrid(X, Y)
    Z = _f(X,Y)
    # return go.Surface(x=X, y=Y, z=Z, surfacecolor=0.5+0*X, colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']], opacity=1.0)
    return go.Surface(x=X, y=Y, z=Z)

def sphere(radius, x0, y0, z0):
    u = onp.linspace(0, 2 * onp.pi, 16)
    v = onp.linspace(0, onp.pi, 16)

    X = x0 + radius * onp.outer(onp.cos(u), onp.sin(v))
    Y = y0 + radius * onp.outer(onp.sin(u), onp.sin(v))
    Z = z0 + radius * onp.outer(onp.ones(onp.size(u)), onp.cos(v))
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=0.5+0*X, colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']], opacity=0.4)

def single_scatter(x,y,z,color="(0,0,0)"):
    return go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(
        size=20,
        color=[0.0],                # set color to an array/list of desired values
        colorscale=[[0, f'rgb{color}'], [1, f'rgb{color}']],   # choose a colorscale
        opacity=1.0
    ))

def get_trajectory(eps_ball, x0, y0, z0, mode="normal", treat_as_constant=False):
    plot_objs = []
    plot_objs.append(get_surf())
    plot_objs.append(single_scatter(x0,y0,z0, color="(0,0,0)"))
    plot_objs.append(sphere(eps_ball[0], x0, y0, z0))

    theta = xy2vec(x0,y0)

    # Plot 5 Adversarial points in the beginning of training
    for i in range(10):
        theta_star = make_theta_star(theta,10,eps_ball)
        x1,y1 = vec2xy(theta_star)
        z1 = f(theta_star)
        plot_objs.append(single_scatter(x1,y1,z1, color="(0,255,0)"))

    for i in range(200):
        if(mode == "rob"):
            loss, grads = compute_gradients_rob(theta, 10, eps_ball, treat_as_constant)
        elif(mode == "sgd"):
            loss, grads = value_and_grad(f, argnums=0)(theta)
        elif(mode == "normal"):
            loss, grads = compute_gradients(theta, 10, eps_ball, treat_as_constant)
        
        theta = theta - 0.01*grads
        print("f(theta)",f(theta),"Rob Loss",loss)
        if(i % 5 == 0):
            xi,yi = vec2xy(theta)
            zi = f(theta)
            plot_objs.append(single_scatter(xi,yi,zi, color=f"(0,{int(i*5)},0)"))
    plot_objs.append(sphere(eps_ball[0], xi, yi, zi))
    
    # Plot 5 Adversarial points at the end of training
    for i in range(10):
        theta_star = make_theta_star(theta,10,eps_ball)
        x1,y1 = vec2xy(theta_star)
        z1 = f(theta_star)
        plot_objs.append(single_scatter(x1,y1,z1, color="(0,0,255)"))

    return plot_objs

eps_ball_scalar = 0.5
eps_ball = xy2vec(eps_ball_scalar,eps_ball_scalar)
x0 = -0.8
y0 = 1.0
z0 = _f(x0,y0)

#%%

fig = go.Figure(data=get_trajectory(eps_ball, x0, y0, z0, mode="normal", treat_as_constant=False))
# fig.write_html("Resources/Figures/vis_3d._constantFalse.html")
fig.show()

# fig = go.Figure(data=get_trajectory(eps_ball, x0, y0, z0, mode="normal", treat_as_constant=True))
# fig.write_html("Resources/Figures/vis_3d_constantTrue.html")
# fig.show()

# %%

# @partial(jit, static_argnums=(0,2,3))
def get_grid(resolution,beta_decay,beta_rob,n_attack_steps,attack_size):
    t0 = time.time()
    x = jnp.linspace(-1, 1, resolution)
    y = jnp.linspace(-0.5, 3.0, resolution)

    X, Y = jnp.meshgrid(x, y)
    Z = jnp.zeros((resolution, resolution))
    if beta_rob !=0:
        for i in range(resolution):
            for j in range(resolution):
                theta = xy2vec(x=X[i][j],y=Y[i][j])
                effective_attack_size = xy2vec(x=attack_size * abs(X[i][j]), y=attack_size * abs(Y[i][j]))
                Z = ops.index_update(Z, ops.index[i,j], beta_rob * _robust_loss(theta, n_attack_steps, effective_attack_size))
    
    Z += beta_decay * 0.5*(X**2 + Y**2)
    Z += _f(X,Y)
    t1 = time.time()
    total = t1-t0
    return X,Y,Z,total

#%%
def plot(beta_decay=0.0,beta_rob=0.1,n_attack_steps=3, attack_size= 0.1, view_1 = 50, view_2=195):
    resolution=90
    x_scale=3.5
    y_scale=7
    z_scale=1
    
    X,Y,Z,total = get_grid(resolution,beta_decay,beta_rob,n_attack_steps,attack_size)
    print(f"Done creating grid in {total}")
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig.show()

    fig=plt.figure(figsize=(16,10))
    ax = fig.add_subplot(projection='3d')

    scale=onp.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return onp.dot(Axes3D.get_proj(ax), scale)

    ax.set_zticks([-1, 0, 1, 4])
    ax.get_proj=short_proj
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,)
    ax.view_init(view_1, view_2)

#%%
interact_manual(plot,beta_decay=(0,2,0.1), beta_rob=(0, 2, 0.1), n_attack_steps=(1,10,1), attack_size=(0.1,3.0,0.1))
# %%
