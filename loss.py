import tensorflow as tf
import tensorflow_probability as tfp

# - From https://stackoverflow.com/questions/41863814/is-there-a-built-in-kl-divergence-loss-function-in-tensorflow
def kl(x, y):
    X = tfp.distributions.Categorical(probs=x)
    Y = tfp.distributions.Categorical(probs=y)
    return tfp.distributions.kl_divergence(X, Y)

def get_distance(theta, theta_star, eps=1e-6):
    dist = 0.
    for t,t_star in zip(theta,theta_star):
        t = tf.reshape(t, [-1])
        t_star = tf.reshape(t_star, [-1])
        dist += tf.math.sqrt(tf.norm((t-t_star)/(t+eps))**2 / t.shape[0])
    dist /= len(theta)
    return dist

def lipschitzness(logits, logits_star, theta, FLAGS):
    if(FLAGS.lipschitzness_loss == "mse"):
        raise NotImplementedError
    else:
        # - FIXME Do the logits that come out of the model actually encode probabilities?
        theta_star = [e for e in tf.compat.v1.global_variables() if e.name[:5] != "final"]
        kl_loss = tf.compat.v1.reduce_mean(kl(tf.nn.softmax(logits), tf.nn.softmax(logits_star)))
        dist_theta_theta_star = get_distance(theta,theta_star)
        return tf.compat.v1.tuple([kl_loss / dist_theta_theta_star, kl_loss, dist_theta_theta_star, logits, logits_star], name="lipschitzness_tuple") 

def normal_loss(target_output, logits, average_fr, FLAGS):
    
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
    if FLAGS.model_architecture == 'lsnn':
        regularization_f0 = 10 / 1000  # 10Hz
        loss_reg = tf.reduce_sum(tf.square(average_fr - regularization_f0) * FLAGS.reg)
        cross_entropy_mean += loss_reg
    return cross_entropy_mean

def evaluate_loss_function(target_output, logits, average_fr, FLAGS):
    
    loss = normal_loss(target_output, logits, average_fr, FLAGS)
    return loss



# ########################## BEGIN Lipschitzness loss ##########################
# debug_dict = {"lipschitzness_tuples": [], "theta_star": []}

# # - The variables that we want to attack
# mismatch_parameters = [e for e in tf.compat.v1.global_variables() if e.name[:5] != "final"]
# logits_initial = tf.compat.v1.identity(logits, name="initial_logits")

# # - Save the variables, tensorflow performs a deep copy when identity operation is performed
# theta = [tf.compat.v1.identity(e) for e in mismatch_parameters]

# initial_std = 0.2
# num_updates = 5
# learning_rate_theta = 0.001
# beta = 0.1

# # - Initialize theta_star as a random perturbation
# theta_star = [tf.compat.v1.assign_add(e, tf.random.normal(shape=e.shape, dtype=e.dtype, mean=0.0,stddev=initial_std*e)) for e in mismatch_parameters]

# debug_dict["theta_star"].append(theta_star)

# # - Function to compute lipschitzness loss
# def lipschitzness_loss():
#     # - Return the negative loss since the optimizers minimize function
#     return lipschitzness(logits_initial, logits, theta, theta_star, FLAGS)

# # - Only returns the loss and not also the distance
# def wrapper_lipschitzness_loss():
#     l = lipschitzness_loss()
#     return l[0]
    
# # - Initialize stochastic gradient descent optimizer
# sgd_optimizer_theta = tf.compat.v1.train.GradientDescentOptimizer(learning_rate_theta)
# for i in range(num_updates):
#     # - Compute gradients
#     theta_grads_and_vars = sgd_optimizer_theta.compute_gradients(wrapper_lipschitzness_loss, theta_star)

#     # - Apply the calculated gradients to  the variables
#     # theta_star = [tf.compat.v1.assign_add(v,g) for (g,v) in theta_grads_and_vars]
#     theta_star = [tf.compat.v1.assign_add(e,tf.random.normal(shape=e.shape, dtype=e.dtype)) for e in theta_star]

#     debug_dict["theta_star"].append(theta_star)

#     # - Get lipschitzness output
#     debug_dict["lipschitzness_tuples"].append(lipschitzness_loss())

# # - Add the lipschitzness loss to the total loss using regularizer beta
# loss_with_lipschitzness = loss + beta * wrapper_lipschitzness_loss()

# #  - Reassign the original parameters to the graph to not mess with the training
# mismatch_parameters = [tf.compat.v1.assign(e, theta[i]) for i,e in enumerate(mismatch_parameters)]

# ########################## END Lipschitzness loss ##########################