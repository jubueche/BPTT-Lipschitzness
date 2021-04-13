from architectures import mnist_mlp as arch
from architectures import log
import MNIST.mnist_dataloader as dataloader
import os.path as path
from MLP_Jax import *

if __name__ == '__main__':
    FLAGS = arch.get_flags()
    base_path = path.dirname(path.abspath(__file__))
    model_save_path = path.join(base_path, f"Resources/Models/{FLAGS.session_id}_model.json")

    layer_sizes = [784, 1024, 1024, 10]
    param_scale = 0.1

    train_images, train_labels, test_images, test_labels = dataloader.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    def check_robustness(params, num_iterations, eps):
        test_accs = []
        for _ in range(num_iterations):
            params_rand = deepcopy(params)
            for idx in range(len(params_rand[:-1])):
                params_rand[idx] = (params_rand[idx][0] + npr.uniform(low=-eps,high=eps,size=params_rand[idx][0].shape), params_rand[idx][1])
            test_accs.append(float(accuracy(params_rand, (test_images, test_labels))))
        return jnp.sum(test_accs) / len(test_accs)

    params = init_random_params(param_scale, layer_sizes)

    mlp = MLP(params)

    for epoch in range(int(FLAGS.n_epochs)):
        
        robustness = check_robustness(mlp.params, num_iterations=FLAGS.n_iters, eps=FLAGS.eps_attack)
        test_acc = accuracy(mlp.params, (test_images, test_labels))
        wm = jnp.sum([jnp.mean(jnp.abs(p[0])) for p in mlp.params[:-1]])
        print(f"{epoch} / {FLAGS.n_epochs} Test accuracy {test_acc}")
        print(f"Weight magnitude {wm}")
        print(f"Robustness {robustness}")

        log(FLAGS.session_id,"weight_magnitude",float(wm))
        log(FLAGS.session_id,"robustness",float(robustness))
        log(FLAGS.session_id,"test_acc",float(test_acc))

        for _ in range(num_batches):
            mlp.params = update(mlp.params, next(batches), FLAGS.weight_increase, FLAGS.step_size)

    mlp.save(model_save_path)