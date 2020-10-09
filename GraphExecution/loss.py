import tensorflow as tf
import tensorflow_probability as tfp

# - From https://stackoverflow.com/questions/41863814/is-there-a-built-in-kl-divergence-loss-function-in-tensorflow
def kl(x, y):
        X = tfp.distributions.Categorical(probs=x)
        Y = tfp.distributions.Categorical(probs=y)
        return tfp.distributions.kl_divergence(X, Y)
    
def loss_lip(logits, logits_adv):
    s_logits = tf.nn.softmax(logits) # [BS,4]
    s_logits_adv = tf.nn.softmax(logits_adv)
    return kl(s_logits, s_logits_adv)

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
