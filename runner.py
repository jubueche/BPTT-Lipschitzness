from GraphExecution import utils
DEFAULT_N_HIDDEN = 200 #???
DEFAULT_BETA_LIPSCHITZNESS = 10 #???
BATCH_SIZE = 100 #???

default = ["--batch_size=100","--model_architecture=lsnn","--n_hidden=200", "--wanted_words=yes,no", "--use_epsilon_ball", "--epsilon_lipschitzness=0.01", "--num_steps_lipschitzness=10","--beta_lipschitzness=10.0"]
parser = utils.get_parser()
defaultflags = parser.parse_args(default)

import os

def find_model(flags):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "Jax/Resources/")
    name_end = '_{}_h{}_b{}_s{}'.format(flags.model_architecture, flags.n_hidden, flags.beta_lipschitzness, flags.seed)
    for file in os.listdir(model_path):
        if name_end in file:
            return os.path.join(name_end, file)
    return None

def run_model(flags):
    model_file = find_model(flags)
    if not model_file is None:
        print("Found Model, returning")
        return
    


def get_model(params):
    pass
    
#Experiment A:
