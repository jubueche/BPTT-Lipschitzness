from jax.nn import softmax, log_softmax
from jax import jit, random, partial, grad, value_and_grad
import jax.numpy as jnp

@jit
def categorical_cross_entropy(y, logits):
    """ Calculates cross entropy and applies regularization to average firing rate"""
    logits_s = jnp.log(softmax(logits))
    nll = jnp.take_along_axis(logits_s, jnp.expand_dims(y, axis=1), axis=1)
    cce = -jnp.mean(nll)
    return cce

@jit
def loss_normal(y, logits, average_fr, regularizer):
    cce = categorical_cross_entropy(y, logits)

    # Regularization on the firing rate
    regularization_f0 = 10 / 1000  # 10Hz
    loss_reg = jnp.sum(jnp.square(average_fr - regularization_f0) * regularizer)
    cce += loss_reg
    return cce

@jit
def loss_kl(logits, logits_theta_star):
    # - Apply softmax
    logits_s = softmax(logits)
    logits_theta_star_s = softmax(logits_theta_star)
    # - Assumes [BatchSize,Output] shape
    kl = jnp.mean(jnp.sum(logits_s * jnp.log(logits_s / logits_theta_star_s), axis=1))
    return kl

@partial(jit, static_argnums=(4,5,6,7,8))
def compute_gradient_and_update(batch_id, X, y, opt_state, opt_update, get_params, rnn, FLAGS, key):
    params = get_params(opt_state)

    def training_loss(X, y, params, l2_reg):
        logits, spikes = rnn.call(X, **params)
        avg_firing = jnp.mean(spikes, axis=1) 
        return loss_normal(y, logits, avg_firing, l2_reg)

    def lip_loss(X, theta_star, theta, logits):
        logits_theta_star, _ = rnn.call(X, **theta_star)
        return loss_kl(logits, logits_theta_star)

    def robust_loss(X, y, params, FLAGS, rand_key):
        _, *sks = random.split(rand_key, len(params.keys())+2)
        if(FLAGS.use_epsilon_ball):
            initial_std = 0.001
        else:
            initial_std = 0.2
        theta_star = {}
        # - Initialize theta_star randomly
        for i,key in enumerate(params.keys()):
            theta_star[key] = params[key] * (1 + initial_std*random.normal(key = sks[i+1], shape=params[key].shape))

        logits, _ = rnn.call(X, **params)
        for i in range(FLAGS.num_steps_lipschitzness):
            grads_theta_star = grad(lip_loss, argnums=1)(X, theta_star, params, logits)
            for key in theta_star.keys():
                if(FLAGS.use_epsilon_ball):
                    theta_star[key] = theta_star[key] + FLAGS.epsilon_lipschitzness/FLAGS.num_steps_lipschitzness * jnp.sign(grads_theta_star[key])
                else:
                    theta_star[key] = theta_star[key] + FLAGS.step_size_lipschitzness * grads_theta_star[key]
        loss_kl = lip_loss(X, theta_star, params, logits)
        return loss_kl

    def loss_general(X, y, params, FLAGS):
        loss_n = training_loss(X, y, params, FLAGS.reg)
        loss_r = robust_loss(X, y, params, FLAGS, key)
        return loss_n + FLAGS.beta_lipschitzness*loss_r

    # - Differentiate w.r.t element at argnums (deault 0, so first element)
    if(FLAGS.lipschitzness):
        grads = grad(loss_general, argnums=2)(X, y, params, FLAGS)
    else:
        grads = grad(training_loss, argnums=2)(X, y, params, FLAGS.reg)
    diag_indices = jnp.arange(0,grads["W_rec"].shape[0],1)
    # - Remove the diagonal of W_rec from the gradient
    grads["W_rec"] = grads["W_rec"].at[diag_indices,diag_indices].set(0.0)
    # clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
    return opt_update(batch_id, grads, opt_state)

@partial(jit, static_argnums=(3,4,5))
def attack_network(X, theta, logits, rnn, FLAGS, key):

    def lip_loss(X, theta_star, logits):
        logits_theta_star, _ = rnn.call(X, **theta_star)
        return loss_kl(logits, logits_theta_star)

    _, *sks = random.split(key, len(theta.keys())+2)
    rnn._rng_key = sks[0]
    if(FLAGS.use_epsilon_ball):
        initial_std = 0.001
    else:
        initial_std = 0.2
    theta_star = {}
    # - Initialize theta_star randomly
    for i,key in enumerate(theta.keys()):
        theta_star[key] = theta[key] * (1 + initial_std*random.normal(key = sks[i+1], shape=theta[key].shape))

    loss_over_time = []
    for i in range(FLAGS.num_steps_lipschitzness):
        value, grads_theta_star = value_and_grad(lip_loss, argnums=1)(X, theta_star, logits)
        loss_over_time.append(value)
        for key in theta_star.keys():
            if(FLAGS.use_epsilon_ball):
                theta_star[key] = theta_star[key] + FLAGS.epsilon_lipschitzness/FLAGS.num_steps_lipschitzness * jnp.sign(grads_theta_star[key])
            else:
                theta_star[key] = theta_star[key] + FLAGS.step_size_lipschitzness * grads_theta_star[key]
    loss_over_time.append(lip_loss(X, theta_star, logits))
    logits_theta_star , _ = rnn.call(X, **theta_star)
    return loss_over_time, logits_theta_star
