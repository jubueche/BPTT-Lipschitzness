import argparse
import importlib
from datajuicer import run, configure, make_unique
from datajuicer.database import select, remove, get_tables
import os.path
import os
import re

base = os.path.join(os.path.dirname(os.path.realpath(__file__)),"Resources/")
sub_paths = [(os.path.join(base,p),ext) for p,ext in zip(["Logs","Models","TrainingResults"],[".log","_model.json",".json"])]

def remove_files(session_id,architecture_names):
    for sub_path,ext in sub_paths:
        f_name = os.path.join(sub_path,f"{session_id}{ext}")
        if os.path.isfile(f_name):
            print(f"Remove {f_name}")
            os.remove(f_name)
    remove_from_db(session_id,architecture_names)

def remove_from_db(session_id,architecture_names):
    # - Get the table names from the database
    table_names = get_tables("Sessions/sessions.db")

    for table_name in table_names:
        if table_name in architecture_names:
            key_name = "session_id"
        else:
            key_name = "{architecture}_session_id"
            # - Get the session_id that is used to store the pickled files
            pickle_session_id = select("Sessions/sessions.db", "session_id", table_name, where={key_name:session_id}, order_by=None)
            if not pickle_session_id == []:
                pisd = pickle_session_id[0]
                fname = f"{table_name}_{pisd}.pickle"
                if os.path.isfile(os.path.join(os.path.dirname(os.path.realpath(__file__)),f"Sessions/{fname}")):
                    print(f"Removed {fname}")
                    os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)),f"Sessions/{fname}"))

        remove("Sessions/sessions.db", table=table_name, key_name=key_name, primary_key=session_id)


toplevel = importlib.import_module("Experiments")
    
exps = []
for f in os.listdir("Experiments"):
    if os.path.isfile(os.path.join("Experiments",f)) and f.endswith(".py"):
        exps+=[f[0:-3]]

for module in ["Experiments."+ex for ex in exps]:
    importlib.import_module(module)

experiments = [getattr(getattr(toplevel,ex),ex) for ex in exps]

models = []

for ex in experiments:
    models += ex.train_grid()

# - Get unique architectures and loop over them
table_names = set([m['architecture'] for m in models])

for architecture in table_names:

    #! remove
    if architecture is "mnist_mlp":
        continue

    # - Get grid for that architecture
    grid_architecture = [m for m in models if m['architecture']==architecture]

    # - Load as much from the grid as possible
    loaded_models = run(grid_architecture, "train", run_mode="load_any", store_key="*")("{*}")

    # - Create dict from models
    hashed_models = {m[f"{architecture}_session_id"]:m for m in loaded_models}

    # - Get all session_ids for that table
    session_ids_table = select("Sessions/sessions.db", column="session_id", table=architecture, where={}, order_by='start_time')

    for session_id in session_ids_table:
        # - Is the session_id in any of the grid's session_id's?
        entry = hashed_models.get(session_id, None)
        if entry == None:
            # - This session_id is not needed and can be removed
            remove_files(session_id,table_names)
        else:
            epochs_needed = sum([int(s) for s in entry["n_epochs"].split(',')])
            # - Load the log and check how many epochs were trained
            log_fname = os.path.join(base, f"Logs/{session_id}.log")
            with open(log_fname, "r") as f:
                lines = f.readlines()
            
            largest_epoch = -1
            for line in lines:
                match = re.search("Epoch\s(\d+)", line)
                if match:
                    epoch = int(match.group(1))+1
                    if epoch > largest_epoch:
                        largest_epoch = epoch

            # - Is the largest epoch good enough?
            if largest_epoch < epochs_needed-2: # - Cut some slack
                print(f"Found log with {largest_epoch} of {epochs_needed} epochs ({session_id})")
                remove_files(session_id,table_names)