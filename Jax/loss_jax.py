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

@partial(jit, static_argnums=(1))
def split_and_sample(key, shape):
  key, subkey = random.split(key)
  val = random.normal(subkey, shape=shape)
  return key, val

@partial(jit, static_argnums=(1,2))
def split_and_get_dropout_mask(key, shape, dp):
    key, subkey = random.split(key)
    val = (random.uniform(subkey, shape=shape) > dp).astype(jnp.float32)
    return key, val

@partial(jit, static_argnums=(4,5,6,7))
def compute_gradient_and_update(batch_id, X, y, opt_state, opt_update, get_params, rnn, FLAGS, rand_key):
    params = get_params(opt_state)

    _, subkey = random.split(rand_key)
    subkey, dropout_mask = split_and_get_dropout_mask(subkey, (1,rnn.units), rnn.dropout_prob)

    def training_loss(X, y, params, l2_reg, dropout_mask):
        logits, spikes = rnn.call(X, dropout_mask, **params)
        avg_firing = jnp.mean(spikes, axis=1) 
        return loss_normal(y, logits, avg_firing, l2_reg)

    def lip_loss(X, theta_star, theta, logits, dropout_mask):
        logits_theta_star, _ = rnn.call(X, dropout_mask, **theta_star)
        return loss_kl(logits, logits_theta_star)


    def robust_loss(X, y, params, FLAGS, rand_key, dropout_mask):
        rand_key, random_normal_var = split_and_sample(rand_key, (1,))
        epsilon_std = (FLAGS.mean_attack_epsilon - FLAGS.minimum_attack_epsilon) * jnp.sqrt(jnp.pi) / jnp.sqrt(2.)
        epsilon = FLAGS.minimum_attack_epsilon + jnp.abs(random_normal_var) * epsilon_std
        step_size = {}
        theta_star = {}
        # - Initialize theta_star randomly 
        for key in params.keys():
            rand_key, random_normal_var = split_and_sample(rand_key, params[key].shape)
            if(FLAGS.relative_initial_std):
                theta_star[key] = params[key] * (1 + FLAGS.initial_std*random_normal_var)
            else:
                theta_star[key] = params[key]+FLAGS.initial_std*random_normal_var
            if(FLAGS.relative_epsilon):
                step_size[key] = epsilon * params[key] /FLAGS.num_attack_steps
            else:    
                step_size[key] = epsilon / FLAGS.num_attack_steps

        logits, _ = rnn.call(X, dropout_mask, **params)
        for _ in range(FLAGS.num_attack_steps):
            grads_theta_star = grad(lip_loss, argnums=1)(X, theta_star, params, logits, dropout_mask)
            for key in theta_star.keys():
                theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])
        loss_kl = lip_loss(X, theta_star, params, logits, dropout_mask)
        return loss_kl

    def loss_general(X, y, params, FLAGS, rand_key, dropout_mask):
        loss_n = training_loss(X, y, params, FLAGS.reg, dropout_mask)
        loss_r = robust_loss(X, y, params, FLAGS, rand_key, dropout_mask)
        return loss_n + FLAGS.beta_lipschitzness*loss_r

    # - Differentiate w.r.t element at argnums (deault 0, so first element)
    if(FLAGS.beta_lipschitzness!=0):
        grads = grad(loss_general, argnums=2)(X, y, params, FLAGS, subkey, dropout_mask)
    else:
        grads = grad(training_loss, argnums=2)(X, y, params, FLAGS.reg, dropout_mask)
    diag_indices = jnp.arange(0,grads["W_rec"].shape[0],1)
    # - Remove the diagonal of W_rec from the gradient
    grads["W_rec"] = grads["W_rec"].at[diag_indices,diag_indices].set(0.0)
    # clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
    return opt_update(batch_id, grads, opt_state)

@partial(jit, static_argnums=(3,4))
def attack_network(X, params, logits, rnn, FLAGS, rand_key):
    #In contrast to the training attacker this attackers epsilon is deterministic (equal to the mean epsilon)
    dropout_mask = jnp.ones(shape=(1,rnn.units))

    def lip_loss(X, theta_star, logits):
        logits_theta_star, _ = rnn.call(X, dropout_mask, **theta_star)
        return loss_kl(logits, logits_theta_star)

    _, rand_key = random.split(rand_key)

    rand_key, random_normal_var = split_and_sample(rand_key, (1,))
    epsilon_std = (FLAGS.mean_attack_epsilon - FLAGS.minimum_attack_epsilon) * jnp.sqrt(jnp.pi) / jnp.sqrt(2.)
    epsilon = FLAGS.minimum_attack_epsilon + jnp.abs(random_normal_var) * epsilon_std
    step_size = {}
    theta_star = {}

    # - Initialize theta_star randomly 
    for key in params.keys():
        rand_key, random_normal_var = split_and_sample(rand_key, params[key].shape)
        if(FLAGS.relative_initial_std):
            theta_star[key] = params[key]+FLAGS.initial_std*random_normal_var
        else:
            theta_star[key] = params[key] * (1 + FLAGS.initial_std*random_normal_var)
        if(FLAGS.relative_epsilon):
            step_size[key] = epsilon * params[key] /FLAGS.num_attack_steps
        else:    
            step_size[key] = epsilon /FLAGS.num_attack_steps

    loss_over_time = []
    logits, _ = rnn.call(X, dropout_mask, **params)
    for _ in range(FLAGS.num_attack_steps):
        value, grads_theta_star = value_and_grad(lip_loss, argnums=1)(X, theta_star, logits)
        loss_over_time.append(value)
        for key in theta_star.keys():
            theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])
    loss_over_time.append(lip_loss(X, theta_star, logits))
    logits_theta_star , _ = rnn.call(X, dropout_mask, **theta_star)
    return loss_over_time, logits_theta_star 