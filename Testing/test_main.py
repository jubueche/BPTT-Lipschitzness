from jax.config import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import os.path as path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ) + "/GraphExecution")
import input_data_eager as input_data
import tensorflow as tf
from old.RNN import RNN
from RNN_Jax import RNN as Jax_RNN
import ujson as json
import numpy as np
from utils import get_parser, prepare_model_settings
import loss_jax as loss_class_jax
import jax.numpy as jnp
import loss_jax
from jax import grad
from jax.experimental import optimizers

if __name__ == '__main__':

    print(f"Tensorflow version {tf.__version__} Using eager evalation {tf.executing_eagerly()} should be True")

    parser = get_parser()
    FLAGS, unparsed = parser.parse_known_args()
    if(len(unparsed)>0):
        print("Received argument that cannot be passed. Exiting...")
        print(unparsed)
        sys.exit(0)

    model_settings = prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.feature_bin_count, FLAGS.preprocess,
        FLAGS.in_repeat
    )

    flags_dict = vars(FLAGS)
    for key in flags_dict.keys():
        model_settings[key] = flags_dict[key]
    
    # - Read the weights and input from the Graz output
    fn = os.path.join(os.path.dirname(__file__), "tests/graz_output.json")
    with open(fn, "r") as f:
        graz_dict = json.load(f)

    model_settings_graz = graz_dict["model_settings"]

    for key in model_settings_graz.keys():
        # print(f"{key}: Graz {model_settings_graz[key]} Ours {model_settings[key]}")
        assert(model_settings_graz[key] == model_settings[key]), "Model settings differ"

    # - Define trainable variables
    d_In = model_settings['fingerprint_width']
    d_Out = model_settings["label_count"]
    W_in = tf.Variable(initial_value=graz_dict["W_in"], trainable=True)
    W_rec = tf.Variable(initial_value=graz_dict["W_rec"], trainable=True)

    W_out = tf.Variable(initial_value=graz_dict["W_out"], trainable=True)
    b_out = tf.Variable(initial_value=graz_dict["b_out"], trainable=True)

    # - Create the model
    rnn = RNN(model_settings)

    # - Create Jax model
    rnn_jax = Jax_RNN(model_settings)

    # - Define loss function
    def loss_normal(target_output, logits):
        cross_entropy_mean = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=target_output, logits=logits)
        return cross_entropy_mean

    logits_graz = tf.cast(graz_dict["logits"],dtype=tf.float32)

    @tf.function
    def get_loss_and_gradients():
        with tf.GradientTape(persistent=False) as tape:
            logits, spikes = rnn.call(fingerprint_input=graz_dict["train_input"], W_in=W_in, W_rec=W_rec, W_out=W_out, b_out=b_out)
            loss = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits)
        gradients = tape.gradient(loss, [W_in,W_rec,W_out,b_out])
        return loss, logits, spikes, gradients

    loss, logits, spikes, gradients = get_loss_and_gradients()
    gradients = [g.numpy() for g in gradients]

    # - Jax
    print("================JAX================")

    def jax_compute_gradient(batch_id, X, y, opt_state, opt_update, get_params):
        params = get_params(opt_state)
        def training_loss(X, y, params):
            logits, spikes = rnn_jax.call(X, jnp.ones(shape=(1,rnn_jax.units)), **params)
            return loss_jax.loss_normal(y, logits, 0.0, 0.0)
        # - Differentiate w.r.t element at argnums (deault 0, so first element)
        grads = grad(training_loss, argnums=2)(X, y, params)
        diag_indices = jnp.arange(0,grads["W_rec"].shape[0],1)
        # - Remove the diagonal of W_rec from the gradient
        grads["W_rec"] = grads["W_rec"].at[diag_indices,diag_indices].set(0.0)
        # clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
        return grads

    init_params = {"W_in": jnp.array(W_in.numpy()), "W_rec": jnp.array(W_rec.numpy()), "W_out": jnp.array(W_out.numpy()), "b_out": jnp.array(b_out.numpy())}
    opt_init, opt_update, get_params = optimizers.adam(0.001, 0.9, 0.999, 1e-08)
    opt_state = opt_init(init_params)
    jax_gradients = jax_compute_gradient(1, jnp.array(graz_dict["train_input"]), jnp.array(graz_dict["train_groundtruth"], dtype=int), opt_state, opt_update, get_params)
    jax_gradients = [np.asarray(jax_gradients[k]) for k in jax_gradients.keys()]
    tmp = jax_gradients[1]
    jax_gradients[1] = jax_gradients[2]
    jax_gradients[2] = tmp

    logits_jax, spikes_jax = rnn_jax.call(np.array(graz_dict["train_input"]), jnp.ones(shape=(1,rnn_jax.units)), W_in=W_in.numpy(), W_rec=W_rec.numpy(), W_out=W_out.numpy(), b_out=b_out.numpy())
    loss_jax = loss_class_jax.loss_normal(np.array(graz_dict["train_groundtruth"], dtype=int), logits_jax, 0.0, 0.0)

    print("Checking gradients...")
    graz_gradients = [np.asarray(g) for g in graz_dict["gradients"]]
    pass_grad = True
    for i in range(len(gradients)):
        assert(gradients[i].shape == graz_gradients[i].shape == jax_gradients[i].shape)
        if (not (np.isclose(gradients[i],graz_gradients[i])).all()):
            pass_grad = False
        if (not (np.mean(np.abs(gradients[i]-jax_gradients[i])) < 1e-4)):
            pass_grad = False

    loss_graz = loss_normal(tf.cast(graz_dict["train_groundtruth"],dtype=tf.int32), logits_graz)
    spikes_graz = np.asarray(graz_dict["spikes"])

    print("Checking loss...")
    # print(f"Loss ours {loss.numpy()} and theirs {loss_graz.numpy()}")
    d = tf.reduce_sum(tf.math.abs((logits-logits_graz))).numpy()
    d_jax = tf.reduce_sum(tf.math.abs((logits_jax-logits_graz))).numpy()
    # print(f"Sum of absolute differences is {d}")
    if(abs(d) < 1e-3 and abs(d_jax) < 0.01):
        print("\033[92mPASSED\033[0m Logits test")
    else:
        print("\033[91mFAILED\033[0m Logits test")
    
    if(abs(loss_graz-float(loss_jax)) < 1e-5):
        print("\033[92mPASSED\033[0m Loss test")
    else:
        print("\033[91mFAILED\033[0m Loss test")

    if(pass_grad):
        print("\033[92mPASSED\033[0m Gradient test")
    else:
        print("\033[91mFAILED\033[0m Gradient test")    
        
    print("=========================================================")