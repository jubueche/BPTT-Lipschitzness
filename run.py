from run_utils import run, make_unique
import argparse
import importlib
import Experiments

ap = argparse.ArgumentParser()
ap.add_argument("-exp", nargs="+", default=[])
ap.add_argument("-mode", type=str, default="direct", help="direct or bsub")
ap.add_argument("-n_threads", type=int, default=0)

flags=ap.parse_args()
if flags.n_threads == 0:
    if flags.mode == "direct":
        flags.n_threads =5
    else:
        flags.n_threads=1

for module in ["Experiments."+exp for exp in flags.exp]:
    importlib.import_module(module)
    
experiments = [getattr(getattr(Experiments,exp),exp) for exp in flags.exp]

models = []
for experiment in experiments:
    for grid in experiment.get_grids(flags.mode).values():
        models += grid


models = make_unique(models)
run("{architecture}",models,"{launch}",n_threads=flags.n_threads)
