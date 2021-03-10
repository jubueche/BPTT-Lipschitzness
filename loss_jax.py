from jax.nn import softmax, log_softmax
from jax import jit, random, partial, grad, value_and_grad
import jax.numpy as jnp
import numpy as onp
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def compute_gradients(X, y, params, model, FLAGS, rand_key):
    _, subkey = random.split(rand_key)
    subkey, dropout_mask = model.split_and_get_dropout_mask(rand_key, FLAGS.dropout_prob)

    def _training_loss(X, y, params, l2_reg, dropout_mask):
        logits, spikes = model.call(X, dropout_mask, **params)
        avg_firing = jnp.mean(spikes, axis=1) 
        l2 = FLAGS.l2_weight_decay * jnp.sum(jnp.array([jnp.linalg.norm(params[el],'fro') for el in FLAGS.l2_weight_decay_params]))
        l1 = FLAGS.l1_weight_decay * jnp.sum(jnp.array([jnp.sum(jnp.abs(params[el])) for el in FLAGS.l1_weight_decay_params]))
        return loss_normal(y, logits, avg_firing, l2_reg) + l2 + l1

    def training_loss(X, y, params, l2_reg, dropout_mask):
        grads = grad(_training_loss, argnums=2)(X, y, params, FLAGS.reg, model.unmasked())
        loss_contractive = FLAGS.contractive * jnp.sum(jnp.array([jnp.linalg.norm(jnp.ravel(grads[el]), ord=jnp.inf) for el in FLAGS.contractive_params]))
        return _training_loss(X, y, params, l2_reg, dropout_mask) + loss_contractive

    def lip_loss(X, theta_star, logits, dropout_mask):
        logits_theta_star, _ = model.call(X, dropout_mask, **theta_star)
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
            theta_star[key] = params[key] + jnp.abs(params[key]) * FLAGS.initial_std_mismatch*random_normal_var1 + FLAGS.initial_std_constant*random_normal_var2
            step_size[key] = (FLAGS.attack_size_mismatch * jnp.abs(params[key]) + FLAGS.attack_size_constant) /FLAGS.n_attack_steps

        for _ in range(FLAGS.n_attack_steps):
            grads_theta_star = grad(lip_loss, argnums=1)(X, theta_star, logits, dropout_mask)
            for key in theta_star.keys():
                theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])
        return theta_star
    
    def robust_loss(X, y, params, FLAGS, rand_key, dropout_mask, theta_star):
        logits, _ = model.call(X, dropout_mask, **params)
        if theta_star is None:
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, logits)
        
        return lip_loss(X, theta_star, logits, dropout_mask)

    def loss_general(X, y, params, FLAGS, rand_key, dropout_mask, theta_star):
        loss_n = training_loss(X, y, params, FLAGS.reg, dropout_mask)
        loss_r = robust_loss(X, y, params, FLAGS, rand_key, dropout_mask, theta_star)
        return loss_n + FLAGS.beta_robustness*loss_r

    # - Differentiate w.r.t. element at argnums (deault 0, so first element)
    if(FLAGS.beta_robustness!=0):
        if FLAGS.treat_as_constant:
            logits, _ = model.call(X, dropout_mask, **params)
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
def compute_gradient_and_update(batch_id, X, y, opt_state, opt_update, get_params, model, FLAGS, rand_key):
    params = get_params(opt_state)
    grads = compute_gradients(X,y,params,model,FLAGS,rand_key)
    return opt_update(batch_id, grads, opt_state)

def _get_logits(max_size, model, X, dropout_mask, theta):
    """ Memory friendly parallel execution of the model """
    if(X.shape[0] <= max_size):
        logits, _ = model.call(X, dropout_mask, **theta)
    else:
        _f = lambda X, dropout_mask, theta, idx : (model.call(X, dropout_mask, **theta), idx)
        N = X.shape[0]
        intervals = [(i,i+max_size) for i in range(0,N - (N % max_size),max_size)] + (lambda i : [(N - (N % max_size),N)] if i else [])(N % max_size)
        with ThreadPoolExecutor(max_workers=10) as executor:
            parallel_results = []
            futures = [executor.submit(_f, X[el[0]:el[1]], dropout_mask, theta, idx) for idx,el in enumerate(intervals)]
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
            parallel_results = sorted(parallel_results, key=lambda k: k[1])
            logits_list = [p[0] for (p,_) in parallel_results]
        logits = jnp.vstack(logits_list)
    return logits

def attack_network(X, params, logits, model, FLAGS, rand_key):
    #In contrast to the training attacker this attackers epsilon is deterministic (equal to the mean epsilon)
    dropout_mask = model.unmasked()
    max_size = 1000
    N = X.shape[0]

    def lip_loss(X, theta_star, logits, dropout_mask):
        logits_theta_star = _get_logits(max_size, model, X, dropout_mask, theta_star)
        if FLAGS.boundary_loss == "kl":
            return loss_kl(logits, logits_theta_star)
        if FLAGS.boundary_loss == "reverse_kl":
            return loss_kl(logits_theta_star, logits)
        if FLAGS.boundary_loss == "l2":
            return jnp.mean(jnp.linalg.norm(logits-logits_theta_star,axis=1))

    step_size = {}
    theta_star = {}

    # - Initialize theta_star randomly 
    for key in params.keys():
        rand_key, random_normal_var1 = split_and_sample(rand_key, params[key].shape)
        rand_key, random_normal_var2 = split_and_sample(rand_key, params[key].shape)
        theta_star[key] = params[key] + jnp.abs(params[key]) * FLAGS.initial_std_mismatch * random_normal_var1 + FLAGS.initial_std_constant * random_normal_var2
        step_size[key] = (FLAGS.attack_size_mismatch * jnp.abs(params[key]) + FLAGS.attack_size_constant) / FLAGS.n_attack_steps

    loss_over_time = []
    logits = _get_logits(max_size, model, X, dropout_mask, params)
    for _ in range(FLAGS.n_attack_steps):
        if(N <= max_size):
            value, grads_theta_star = value_and_grad(lip_loss, argnums=1)(X, theta_star, logits, dropout_mask)
        else: 
            def _f(X, logits, N, theta_star):
                v,g = value_and_grad(lip_loss, argnums=1)(X, theta_star, logits, dropout_mask)
                return (v,g,N)
            def _add_dict(a, b):
                if(a == {}):
                    return b
                elif(b == {}):
                    return a
                else:
                    res = {}
                    for key in a:
                        res[key] = a[key] + b[key]
                    return res
            def _mult_dict(a, scalar):
                res = {}
                for key in a:
                    res[key] = scalar * a[key]
                return res
            value = 0
            grads_theta_star = {}
            intervals = [(i,i+max_size) for i in range(0,N - (N % max_size),max_size)] + (lambda i : [(N - (N % max_size),N)] if i else [])(N % max_size)
            with ThreadPoolExecutor(max_workers=10) as executor:
                parallel_results = []
                futures = [executor.submit(_f, X[el[0]:el[1]], logits[el[0]:el[1]], idx, theta_star) for idx,el in enumerate(intervals)]
                for future in as_completed(futures):
                    result = future.result()
                    parallel_results.append(result)
                # - Sort the results
                parallel_results = sorted(parallel_results, key=lambda k: k[2])
                for idx in range(len(intervals)):
                    v,g,_ = parallel_results[idx]
                    Nb = intervals[idx][1]-intervals[idx][0]
                    grads_theta_star = _add_dict(grads_theta_star, _mult_dict(g, Nb/N))
                    value += v*Nb/N

        loss_over_time.append(value)
        for key in theta_star.keys():
            theta_star[key] = theta_star[key] + step_size[key] * jnp.sign(grads_theta_star[key])

    loss_over_time.append(lip_loss(X, theta_star, logits, dropout_mask))
    logits_theta_star = _get_logits(max_size, model, X, dropout_mask, theta_star)
    return loss_over_time, logits_theta_star