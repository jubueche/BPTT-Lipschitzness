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
        theta_star = [e for e in tf.compat.v1.trainable_variables() if e.name[:5] != "final"]
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
