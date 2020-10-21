DEFAULT_N_HIDDEN = 200 #???
DEFAULT_BETA_LIPSCHITZNESS = 10 #???
BATCH_SIZE = 100 #???

default = ["--batch_size=100","--model_architecture=lsnn","--n_hidden=200", "--wanted_words=yes,no", "--use_epsilon_ball", "--epsilon_lipschitzness=0.01", "--num_steps_lipschitzness=10","--beta_lipschitzness=10.0"]

import os

def find_model(params):
    base_path = path.dirname(os.path.abspath(__file__))
    model_path = path.join(base_path, "Resources/")
    name_end = '_{}_h{}_b{}'.format(FLAGS.model_architecture, FLAGS.n_hidden,FLAGS.beta_lipschitzness)
    for file in os.listdir(model_path):
        if name_end in file:
            return os.path.join(name_end, file)
    retun None

def run_model(params):
    if not find_model(params) is None:
        print("Found Model, returning")
        return

def get_model(params):
    pass
    
#Experiment A:
