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

def compute_gradients(X, y, params, rnn, FLAGS, rand_key):
    _, subkey = random.split(rand_key)
    subkey, dropout_mask = split_and_get_dropout_mask(subkey, (1,rnn.units), rnn.dropout_prob)

    def training_loss(X, y, params, l2_reg, dropout_mask):
        logits, spikes = rnn.call(X, dropout_mask, **params)
        avg_firing = jnp.mean(spikes, axis=1) 
        return loss_normal(y, logits, avg_firing, l2_reg)

    def lip_loss(X, theta_star, theta, logits, dropout_mask):
        logits_theta_star, _ = rnn.call(X, dropout_mask, **theta_star)
        if FLAGS.boundary_loss == "kl":
            return loss_kl(logits, logits_theta_star)
        if FLAGS.boundary_loss == "reverse_kl":
            return loss_kl(logits_theta_star, logits)
        if FLAGS.boundary_loss == "l2":
            return jnp.mean(jnp.linalg.norm(logits-logits_theta_star,axis=1))

    def make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, logits):
        step_size = {}
        theta_star = {}
        # - Initialize theta_star randomly 
        for key in params.keys():
            rand_key, random_normal_var1 = split_and_sample(rand_key, params[key].shape)
            rand_key, random_normal_var2 = split_and_sample(rand_key, params[key].shape)
            theta_star[key] = params[key] * (1 + FLAGS.initial_std_mismatch*random_normal_var1)+FLAGS.initial_std_constant*random_normal_var2
            step_size[key] = (FLAGS.attack_size_mismatch * params[key] + FLAGS.attack_size_constant) /FLAGS.n_attack_steps

        
        for _ in range(FLAGS.n_attack_steps):
            grads_theta_star = grad(lip_loss, argnums=1)(X, theta_star, params, logits, dropout_mask)
            for key in theta_star.keys():
                theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])
        return theta_star
    
    def robust_loss(X, y, params, FLAGS, rand_key, dropout_mask, theta_star):
        logits, _ = rnn.call(X, dropout_mask, **params)
        if theta_star is None:
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, logits)
        
        return lip_loss(X, theta_star, params, logits, dropout_mask)

    def loss_general(X, y, params, FLAGS, rand_key, dropout_mask, theta_star):
        loss_n = training_loss(X, y, params, FLAGS.reg, dropout_mask)
        loss_r = robust_loss(X, y, params, FLAGS, rand_key, dropout_mask, theta_star)
        return loss_n + FLAGS.beta_robustness*loss_r

    # - Differentiate w.r.t. element at argnums (deault 0, so first element)
    if(FLAGS.beta_robustness!=0):
        if FLAGS.treat_as_constant:
            logits, _ = rnn.call(X, dropout_mask, **params)
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, logits)
        else:
            theta_star=None
        grads = grad(loss_general, argnums=2)(X, y, params, FLAGS, subkey, dropout_mask, theta_star)
    else:
        grads = grad(training_loss, argnums=2)(X, y, params, FLAGS.reg, dropout_mask)
    if("W_rec" in grads.keys()):
        diag_indices = jnp.arange(0,grads["W_rec"].shape[0],1)
        # - Remove the diagonal of W_rec from the gradient
        grads["W_rec"] = grads["W_rec"].at[diag_indices,diag_indices].set(0.0)
    return grads

@partial(jit, static_argnums=(4,5,6,7))
def compute_gradient_and_update(batch_id, X, y, opt_state, opt_update, get_params, rnn, FLAGS, rand_key):
    params = get_params(opt_state)
    grads = compute_gradients(X,y,params,rnn,FLAGS,rand_key)
    return opt_update(batch_id, grads, opt_state)

def attack_network(X, params, logits, rnn, FLAGS, rand_key):
    #In contrast to the training attacker this attackers epsilon is deterministic (equal to the mean epsilon)
    dropout_mask = jnp.ones(shape=(1,rnn.units))

    def lip_loss(X, theta_star, logits):
        logits_theta_star, _ = rnn.call(X, dropout_mask, **theta_star)
        return loss_kl(logits, logits_theta_star)

    n_attack_steps = FLAGS.n_attack_steps
    initial_std_constant = FLAGS.initial_std_constant
    initial_std_mismatch = FLAGS.initial_std_mismatch
    attack_size_constant = FLAGS.attack_size_constant
    attack_size_mismatch = FLAGS.attack_size_mismatch

    step_size = {}
    theta_star = {}

    # - Initialize theta_star randomly 
    for key in params.keys():
        rand_key, random_normal_var1 = split_and_sample(rand_key, params[key].shape)
        rand_key, random_normal_var2 = split_and_sample(rand_key, params[key].shape)
        theta_star[key] = params[key] * (1 + initial_std_mismatch*random_normal_var1)+initial_std_constant*random_normal_var2
        step_size[key] = (attack_size_mismatch * params[key] + attack_size_constant) /n_attack_steps

    loss_over_time = []
    logits, _ = rnn.call(X, dropout_mask, **params)
    for _ in range(n_attack_steps):
        value, grads_theta_star = value_and_grad(lip_loss, argnums=1)(X, theta_star, logits)
        loss_over_time.append(value)
        for key in theta_star.keys():
            theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])
    loss_over_time.append(lip_loss(X, theta_star, logits))
    logits_theta_star , _ = rnn.call(X, dropout_mask, **theta_star)
    return loss_over_time, logits_theta_star