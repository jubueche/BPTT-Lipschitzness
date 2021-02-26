import numpy as onp
import jax.numpy as jnp
from jax import random, grad, value_and_grad
from copy import copy, deepcopy
import plotly.graph_objects as go

onp.random.seed(42)

def f(theta):
    x = theta[0]; y = theta[1]
    return _f(x,y)

# def _f(x,y):
#     return 3*(1-x)**2*jnp.exp(-(x**2)-(y+1)**2)-10*(x/5 - x**3 - y**5)*jnp.exp(-x**2-y**2) - 1/3*jnp.exp(-(x+1)**2 - y**2)

def _f(x,y):
    return jnp.sin(x**2 + y**2)

    # GeoGebra: 3*(1-x)^2*exp(-(x^2)-(y+1)^2)-10*(x/5 - x^3 - y^5)*exp(-x^2-y^2) - 1/3*exp(-(x+1)^2 - y^2)

def split_and_sample(key, shape):
    key, subkey = random.split(key)
    val = random.normal(subkey, shape=shape)
    return key, val

def robust_loss(theta, theta_star, num_steps, eps_ball):
    if(theta_star == None):
        theta_star = make_theta_star(theta, num_steps, eps_ball)
    return lip_loss(theta,theta_star)

def make_theta_star(theta, num_steps, eps_ball): 
    theta_star = theta + 0.001*onp.random.normal(loc=0.0, scale=1.0, size=theta.shape)
    step_size = eps_ball / num_steps
    for _ in range(num_steps):
        grads_theta_star = grad(lip_loss, argnums=0)(theta_star, theta)
        theta_star = theta_star + step_size * jnp.sign(grads_theta_star)
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
    plot_objs.append(sphere(eps_ball, x0, y0, z0))

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
    plot_objs.append(sphere(eps_ball, xi, yi, zi))
    
    # Plot 5 Adversarial points at the end of training
    for i in range(10):
        theta_star = make_theta_star(theta,10,eps_ball)
        x1,y1 = vec2xy(theta_star)
        z1 = f(theta_star)
        plot_objs.append(single_scatter(x1,y1,z1, color="(0,0,255)"))

    return plot_objs

eps_ball = 0.5
x0 = -0.8
y0 = 1.0
z0 = _f(x0,y0)

fig = go.Figure(data=get_trajectory(eps_ball, x0, y0, z0, mode="normal", treat_as_constant=True))
fig.write_html("Resources/Figures/vis_3d_sinxsqysq_normal_constantTrue.html")
fig.show()

# fig = go.Figure(data=get_trajectory(eps_ball, x0, y0, z0, treat_as_constant=True, rob_only=True))
# fig.write_html("Resources/Figures/vis_3d_peaks_constantTrue_robOnlyTrue.html")
# fig.show()