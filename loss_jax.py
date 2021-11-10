from jax.nn import softmax, log_softmax
from jax import jit, random, partial, grad, value_and_grad, vmap
from jax.lax import stop_gradient
import jax.numpy as jnp
import numpy as onp
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def fast_gradient_method(
    model_fn,
    x,
    eps,
    y
):
    def loss_adv(image, label):
        pred = model_fn(image[None])
        loss = -jnp.sum(jnp.log(softmax(pred)) * label)
        return loss
    grads_fn = vmap(grad(loss_adv), in_axes=(0, 0), out_axes=0)
    grads = grads_fn(x, y)
    perturbation = eps * jnp.sign(grads)
    adv_x = x + perturbation
    return adv_x

def l_inf_pga(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    y,
    k
):
    assert eps_iter <= eps, (eps_iter, eps)
    y = one_hot(y, k)
    # Initialize loop variables
    rand_minmax = eps
    eta = onp.random.uniform(low=-rand_minmax, high=rand_minmax, size=x.shape)
    # Clip eta
    eta = clip_eta(eta, eps)
    adv_x = x + eta
    for _ in range(nb_iter):
        adv_x = fast_gradient_method(
            model_fn,
            adv_x,
            eps_iter,
            y=y
        )
        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, eps)
        adv_x = x + eta
    return adv_x

def clip_eta(eta, eps):
    eta = jnp.clip(eta, a_min=-eps, a_max=eps)
    return eta

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
    kl = jnp.mean(jnp.sum(logits_s * jnp.log(logits_s / jnp.where(logits_theta_star_s >= 1e-6, logits_theta_star_s, 1e-6) ), axis=1))
    return kl

@jit
def loss_js(logits_1, logits_2):
    logits_1 = softmax(logits_1)
    logits_2 = softmax(logits_2)
    M = (logits_1 + logits_2) * 0.5 
    kl = lambda p, q: jnp.mean(jnp.sum(p * jnp.log(p / jnp.where(q >= 1e-6, q, 1e-6) ), axis=1))
    return 0.5 * (kl(logits_1, M) + kl(logits_2, M))

@partial(jit, static_argnums=(1))
def split_and_sample(key, shape):
    key, subkey = random.split(key)
    val = random.normal(subkey, shape=shape)
    return key, val

def _training_loss(X, y, params, FLAGS, model, dropout_mask, rand_key):
    theta = {}
    for key in params:
        rand_key, random_normal_var1 = split_and_sample(rand_key, params[key].shape)
        theta[key] = params[key] + stop_gradient(jnp.abs(params[key]) * FLAGS.noisy_forward_std*random_normal_var1)

    logits, spikes = model.call(X, dropout_mask, **theta)
    avg_firing = jnp.mean(spikes, axis=1) 
    l2 = FLAGS.l2_weight_decay * jnp.sum(jnp.array([jnp.linalg.norm(params[el],'fro') for el in FLAGS.l2_weight_decay_params]))
    l1 = FLAGS.l1_weight_decay * jnp.sum(jnp.array([jnp.sum(jnp.abs(params[el])) for el in FLAGS.l1_weight_decay_params]))
    return loss_normal(y, logits, avg_firing, FLAGS.reg) + l2 + l1

def training_loss(X, y, params, FLAGS, model, dropout_mask, rand_key):
    grads = grad(_training_loss, argnums=2)(X, y, params, FLAGS, model, model.unmasked(),rand_key)
    loss_contractive = FLAGS.contractive * jnp.sum(jnp.array([jnp.linalg.norm(jnp.ravel(grads[el]), ord=jnp.inf) for el in FLAGS.contractive_params]))
    _, subkey = random.split(rand_key)
    return _training_loss(X, y, params, FLAGS, model, dropout_mask, subkey) + loss_contractive

def make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, model, logits):
    make_step = {
        "inf": jnp.sign,
        "2": lambda grad: grad/jnp.linalg.norm(grad)
    }
    step_size = {}
    theta_star = {}
    # - Initialize theta_star randomly 
    for key in params.keys():
        rand_key, random_normal_var1 = split_and_sample(rand_key, params[key].shape)
        rand_key, random_normal_var2 = split_and_sample(rand_key, params[key].shape)
        theta_star[key] = params[key] + jnp.abs(params[key]) * FLAGS.initial_std_mismatch*random_normal_var1 + FLAGS.initial_std_constant*random_normal_var2
        step_size[key] = (FLAGS.attack_size_mismatch * jnp.abs(params[key]) + FLAGS.attack_size_constant) /FLAGS.n_attack_steps

    for _ in range(FLAGS.n_attack_steps):
        grads_theta_star = grad(lip_loss, argnums=2)(X,y, theta_star, logits, FLAGS, model, dropout_mask, rand_key)
        for key in theta_star.keys():
            theta_star[key] = theta_star[key] + step_size[key] * make_step[FLAGS.p_norm](grads_theta_star[key])
    return theta_star

def lip_loss(X, y, theta_star, logits, FLAGS, model, dropout_mask, rand_key):
    logits_theta_star, _ = model.call(X, dropout_mask, **theta_star)
    if FLAGS.boundary_loss == "kl":
        return loss_kl(logits, logits_theta_star)
    if FLAGS.boundary_loss == "reverse_kl":
        return loss_kl(logits_theta_star, logits)
    if FLAGS.boundary_loss == "l2":
        return jnp.mean(jnp.linalg.norm(logits-logits_theta_star,axis=1))
    if FLAGS.boundary_loss == "js":
        return loss_js(logits_theta_star, logits)
    if FLAGS.boundary_loss == "madry":
        return training_loss(X, y, theta_star, FLAGS, model, dropout_mask, rand_key)

def dict_difference(dict1, dict2):
    diff = 0
    for key in dict1:
        diff += jnp.linalg.norm(dict1[key] - dict2[key]) ** 2
    return jnp.sqrt(diff)

def robust_loss(X, y, params, FLAGS, model, rand_key, dropout_mask, theta_star):
    logits, _ = model.call(X, dropout_mask, **params)
    if theta_star is None:
        theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, model, logits)
    if FLAGS.hessian_robustness:
        _, subkey = random.split(rand_key)
        nabla_theta = grad(training_loss, argnums=2)(X, y, params, FLAGS, model, dropout_mask, subkey)
        _, subkey = random.split(subkey)
        nabla_theta_star = grad(training_loss, argnums=2)(X, y, theta_star, FLAGS, model, dropout_mask, subkey)
        return dict_difference(nabla_theta, nabla_theta_star)
    _, subkey = random.split(rand_key)
    return lip_loss(X, y, theta_star, logits, FLAGS, model, dropout_mask, subkey)

def loss_general(X, y, params, FLAGS, model, rand_key, dropout_mask, theta_star):
    loss_n = training_loss(X, y, params, FLAGS, model, dropout_mask, rand_key)
    _, subkey = random.split(rand_key)
    loss_r = robust_loss(X, y, params, FLAGS, model, subkey, dropout_mask, theta_star)
    return loss_n + FLAGS.beta_robustness*loss_r

def compute_gradients(X, y, params, model, FLAGS, rand_key, epoch):
    
    _, subkey = random.split(rand_key)
    subkey, dropout_mask = model.split_and_get_dropout_mask(rand_key, FLAGS.dropout_prob)
    
    if(FLAGS.awp and epoch>=FLAGS.warmup):
        logits, _ = model.call(X, dropout_mask, **params)
        
        if FLAGS.awp_perturb_input:
            if FLAGS.architecture == "cnn":
                k = 10
            elif FLAGS.architecture == "speech_lsnn":
                k = 8
            else:
                raise NotImplementedError
            
            def model_fn(x):
                ret = model.call(x, dropout_mask, **params)
                return ret[0]

            X_star = l_inf_pga(
                model_fn=model_fn,
                x=X,
                eps=FLAGS.eps_pga,
                eps_iter=FLAGS.eps_pga / FLAGS.nb_iter,
                nb_iter=FLAGS.nb_iter,
                y=y,
                k=k
            )
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, model, logits)
            for key in theta_star:
                theta_star[key] = params[key] + FLAGS.awp_gamma * (theta_star[key] - params[key])
            grads = grad(training_loss, argnums=2)(X_star, y, theta_star, FLAGS, model, dropout_mask, subkey)
        else:
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, model, logits)
            for key in theta_star:
                theta_star[key] = params[key] + FLAGS.awp_gamma * (theta_star[key] - params[key])
            grads = grad(training_loss, argnums=2)(X, y, theta_star, FLAGS, model, dropout_mask, subkey)

    elif(FLAGS.beta_robustness==0 or epoch < FLAGS.warmup):
        grads = grad(training_loss, argnums=2)(X, y, params, FLAGS, model, dropout_mask, subkey)
    else:
        if FLAGS.treat_as_constant:
            logits, _ = model.call(X, dropout_mask, **params)
            theta_star = make_theta_star(X, y, params, FLAGS, rand_key, dropout_mask, model, logits)
        else:
            theta_star=None
        grads = grad(loss_general, argnums=2)(X, y, params, FLAGS, model, subkey, dropout_mask, theta_star)

    if("W_rec" in grads.keys()):
        diag_indices = jnp.arange(0,grads["W_rec"].shape[0],1)
        # - Remove the diagonal of W_rec from the gradient
        grads["W_rec"] = grads["W_rec"].at[diag_indices,diag_indices].set(0.0)
    return grads

@partial(jit, static_argnums=(4,5,6,7,9))
def compute_gradient_and_update(batch_id, X, y, opt_state, opt_update, get_params, model, FLAGS, rand_key, epoch):
    params = get_params(opt_state)
    grads = compute_gradients(X,y,params,model,FLAGS,rand_key, epoch)
    return opt_update(batch_id, grads, opt_state)

def _get_logits(max_size, model, X, dropout_mask, theta):
    """ Memory friendly parallel execution of the model """
    if(X.shape[0] <= max_size):
        logits, _ = model.call(X, dropout_mask, **theta)
    else:
        _f = lambda X, dropout_mask, theta, idx : (model.call(X, dropout_mask, **theta), idx)
        N = X.shape[0]
        intervals = [(i,i+max_size) for i in range(0,N - (N % max_size),max_size)] + (lambda i : [(N - (N % max_size),N)] if i else [])(N % max_size)
        with ThreadPoolExecutor(max_workers=2) as executor:
            parallel_results = []
            futures = [executor.submit(_f, X[el[0]:el[1]], dropout_mask, theta, idx) for idx,el in enumerate(intervals)]
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result)
            parallel_results = sorted(parallel_results, key=lambda k: k[1])
            logits_list = [p[0] for (p,_) in parallel_results]
        logits = jnp.vstack(logits_list)
    return logits

def attack_network(X, y, params, model, FLAGS, rand_key):
    loss_over_time, logits_theta_star, _ = _attack_network(X,y, params, model, FLAGS, rand_key)
    return loss_over_time, logits_theta_star

def _attack_network(X,y, params, model, FLAGS, rand_key):
    #In contrast to the training attacker this attackers epsilon is deterministic (equal to the mean epsilon)
    dropout_mask = model.unmasked()
    max_size = 1000
    N = X.shape[0]

    step_size = {}
    theta_star = {}

    # def KL(X, logits, theta_star):
    #     logits_theta_star, _ = model.call(X, dropout_mask, **theta_star)
    #     return loss_kl(logits, logits_theta_star)

    if FLAGS.boundary_loss == "madry":
        def attack_loss(X, y, logits, theta_star):
            logits_theta_star, _ = model.call(X, dropout_mask, **theta_star)
            return categorical_cross_entropy(y, logits_theta_star)
    else:
        def attack_loss(X, y, logits, theta_star):
            logits_theta_star, _ = model.call(X, dropout_mask, **theta_star)
            return loss_kl(logits, logits_theta_star)

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
            value, grads_theta_star = value_and_grad(attack_loss, argnums=3)(X, y, logits, theta_star)
        else: 
            def _f(X,y, logits, N, theta_star):
                v,g = value_and_grad(attack_loss, argnums=3)(X, y, logits, theta_star)
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
                futures = [executor.submit(_f, X[el[0]:el[1]], y[el[0]:el[1]], logits[el[0]:el[1]], idx, theta_star) for idx,el in enumerate(intervals)]
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

    loss_over_time.append(attack_loss(X, y, logits, theta_star))
    logits_theta_star = _get_logits(max_size, model, X, dropout_mask, theta_star)
    return loss_over_time, logits_theta_star, theta_star