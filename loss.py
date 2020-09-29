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

def lipschitzness(theta, theta_star, FLAGS):
    if(FLAGS.lipschitzness_loss == "mse"):
        raise NotImplementedError
    else:
        kl_loss = 0.
        dist_theta_theta_star = get_distance(theta,theta_star)

        return kl_loss / dist_theta_theta_star

def normal_loss(target_output, logits, average_fr, FLAGS):
    
    cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
    if FLAGS.model_architecture == 'lsnn':
        regularization_f0 = 10 / 1000  # 10Hz
        loss_reg = tf.reduce_sum(tf.square(average_fr - regularization_f0) * FLAGS.reg)
        cross_entropy_mean += loss_reg
    return cross_entropy_mean

def evaluate_loss_function(target_output, logits, data_input, output_spikes, average_fr, model, FLAGS, continuous_outputs=None):
    
    loss = normal_loss(target_output, logits, average_fr, FLAGS)

    if(FLAGS.lipschitzness):
        # - Get the tensors that make up Theta
        theta = [node for node in tf.compat.v1.global_variables() if node.name[:5] != "final"]

        # - Create theta_star by initially randomizing values in theta
        initial_std = 0.2
        theta_star = [tf.compat.v1.assign_add(t, tf.compat.v1.random_normal(shape=t.shape, mean=0, stddev=initial_std*tf.compat.v1.math.reduce_std(t), dtype=t.dtype)) for t in theta]
        lip_loss = lipschitzness(theta, theta_star, FLAGS) 
        loss += lip_loss

    return loss




"""
Abstract
"""
input_placeholder = tf.compat.v1.placeholder(...)
targets_placeolder = tf.compat.v1.placeholder(...)
logits = create_model()
loss = evaluate_loss_function(...)

tf.compat.v1.global_variables_initializer().run()

loss = sess.run(
        [loss],
        feed_dict={
            input_name: data_input,
            targets_name: data_targets
        })