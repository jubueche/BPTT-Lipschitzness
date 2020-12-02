import argparse
import os
import pickle
import copy
import sqlite3
import random
import threading
import itertools
import ujson as json
import numpy as np
import errno
import time

def run_in_parallel(target,l_args,n_threads):
    if len(l_args) > 0:
        print("WARNING: No Tasks")
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    threads = []
    for args in l_args:
        t = threading.Thread(target=target,args=args)
        threads.append(t)
    for ts in grouper(threads,n_threads):
        for t in ts:
            if type(t) is None:
                continue
            t.start()
        for t in ts:
            if type(t) is None:
                continue
            t.join()

def update_saved_dict(file, key, value):
    exists = os.path.isfile(file)
    mkdir_p(os.path.dirname(file))
    if exists:
        data = open(file).read()
        try:
            d = json.loads(data)
        except:
            d = {}
    else:
        d = {}
    with open(file,'w+') as f:
        d[key]=value
        json.dump(d,f)

def is_in_saved_dict(file, key):
    if not os.path.isfile(file):
        return False
    with open(file,'r') as f:
        d = json.load(f)
        return key in d

def get_flags_and_register():
    bootstrap_parser = argparse.ArgumentParser()
    bootstrap_parser.add_argument("-arch_file")
    bootstrap_flags, _ = bootstrap_parser.parse_known_args()
    try:
        arch = Architecture.load(bootstrap_flags.arch_file)
    except:
        return None
    flags = arch.register_session()
    
    return flags

def record_training_data(flags, key, value, save_dir = None):
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/TrainingResults/")
    file = os.path.join(save_dir, f"{flags.session_id}.train")
    exists = os.path.isfile(file)
    mkdir_p(os.path.dirname(file))
    if exists:
        data = open(file).read()
        try:
            d = json.loads(data)
        except:
            d = {}
    else:
        d = {}
    with open(file,'w+') as f:
        if key in d:
            d[key] += [value]
        else:
            d[key]=[value]
        json.dump(d,f)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def flatten_dict(params):
        def flatten_lists(ll):
            flatten = lambda t: [item for sublist in t for item in sublist]
            
            if type(ll)==list and len(ll)>0 and type(ll[0])==list:
                return flatten(ll)
            return ll
        if type(params)==dict:
            for key in params:
                if type(params[key])==list:
                    ret = []
                    for value in params[key]:
                        p =  copy.copy(params)
                        p[key] = value
                        ret += [flatten_dict(p)]
                    return flatten_lists(ret)
            return [params]
        if type(params)==list:
            return flatten_lists([flatten_dict(p) for p in params])

class Architecture:
    class Hyperparameter:
        def __init__(self, name, t, default, double_dash=False, help=""):
            self.name = name
            self.t = t
            self.default = default
            self.double_dash=double_dash
            self.help = help

        def add_to_argparser(self,parser):
            dash = '-'
            if self.double_dash:
                dash="--"
            parser.add_argument(dash+self.name,type=self.t,default=self.default)

        def get_argument(self, value):
            dash = "-"
            if self.double_dash: dash = "--"
            return dash+self.name+"="+str(value)

    def __init__(self, name, code_path, loadResults=default_result_loader):
        assert(type(name)==str)
        assert(type(code_path) == str)
        self.name = name
        self.hyperparameters = []
        self.env_parameters = []
        self.code_path = os.path.normpath(code_path)
        self.loadResults = loadResults
    
    def add_hyperparameter(self, name, t, default, double_dash=False,help=""):
        assert type(name) == str
        assert type(t) == type
        assert type(default) == t
        self.hyperparameters += [Architecture.Hyperparameter(name, t, default, double_dash, help)]
    
    def add_env_parameter(self, name, t, default, double_dash=False,help=""):
        assert type(name) == str
        assert type(t) == type
        assert type(default) == t
        self.env_parameters += [Architecture.Hyperparameter(name, t, default, double_dash, help)]

    def get_argparser(self):
        parser = argparse.ArgumentParser(prog = os.path.basename(self.code_path))
        for hp in self.hyperparameters:
            hp.add_to_argparser(parser)
        for hp in self.env_parameters:
            hp.add_to_argparser(parser)
        parser.add_argument('-db_file',type=str,default="")
        parser.add_argument("-session_id",type=int,default=0)
        parser.add_argument("-arch_file",type=str,default='')
        return parser

    def register_session(self):
        parser = self.get_argparser()
        flags = parser.parse_args()
        flags.start_time = int(time.time()*1000)
        if flags.session_id == 0:
            random.seed()
            flags.session_id = random.randint(1000000000, 9999999999)
        
        fieldset = ["'session_id' INTEGER PRIMARY KEY"]
        column_names = [p.name for p in self.hyperparameters]
        column_values = [vars(flags)[key] for key in column_names]
        for key, val in zip(column_names, column_values):
            if type(val) == int:
                definition = "INTEGER"
            if type(val) == float:
                definition = "REAL"
            else:
                definition = "TEXT"
            if key in [p.name for p in self.env_parameters]:
                continue
            fieldset.append("'{0}' {1}".format(key, definition))

        create_table = "CREATE TABLE IF NOT EXISTS {0} ({1});".format(self.name, ", ".join(fieldset))
        
        def format_value(val):
            if type(val) == int:
                return str(val)
            if type(val) == float:
                return str(val)
            return "\"" + str(val) + "\""
        

        register = f"INSERT INTO {self.name} (session_id, "+ ", ".join(column_names) + f" ) VALUES({flags.session_id}, " + ", ".join(map(format_value,column_values)) + " );"
        
        
        
        try:
            print("Registering session...")
            conn = sqlite3.connect(flags.db_file, timeout=100)
            c = conn.cursor()
            c.execute(create_table)
            c.execute(register)
            conn.commit()
            c.close()
        except sqlite3.Error as error:
            print("Failed to insert data into sqlite table", error)
            raise Exception
        finally:
            if (conn):
                conn.close()
                print("Registering Complete")
        return flags

    def get_dict(self):
        d = {}
        for hp in self.hyperparameters:
            d[hp.name] = hp.default
        return d
    
    def make_args(self,values):
    
        def get_value(hp, values):
            if hp.name in values:
                return values[hp.name]
            else:
                return hp.default
        return " ".join([hp.get_argument(get_value(hp,values)) for hp in self.hyperparameters + self.env_parameters])


    def save(self, savePath=None):
        if savePath is None:
            savePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/Architectures/")
        saveFile = os.path.join(savePath, self.name + ".arch")
        mkdir_p(savePath)
        with open(saveFile, 'wb+') as output_file:
            pickle.dump(self,output_file)

    @staticmethod
    def load(loadFile):
        return pickle.load(open(loadFile, 'rb'))

class ModelLoadError(Exception):
    pass




# models can be a list or a dict
def make_grid(models):
    class Model:
        def __init__(self,arch_file,hyperparam_dict,checker=None):
            self.arch_file = arch_file
            self.hyperparams = architecture.get_dict()
            if not os.path.isfile(arch_file):
                print("Architecture File does not exist")
                raise(Exception)
            for key in hyperparam_dict:
                assert key in self.hyperparams
                self.hyperparams[key] = hyperparam_dict[key]
            self.checker=None
        
        def __eq__(self,other):
            return self.hyperparameters == other.hyperparameters and self.arch_file == other.arch_file

        def __ne__(self,other):
            return not self.__eq__(other)

        def __hash__(self,other):
            return hash((self.hyperparams, self.arch_file))

        def make_command(self,command,env_parameters,db_file,time_est="24:00", processors_est="16", memory_est="4096"):
            hyperparams = copy.copy(self.hyperparams)
            for key in env_parameters:
                hyperparams[key] = env_parameters[key]
            session_id = random.randint(1000000000, 9999999999)
            architecture = Architecture.load(self.arch_file)
            ct = command
            ct = ct.replace("$$SESSION_ID$$", str(session_id))
            ct = ct.replace("$$TIME_ESTIMATE$$", str(time_est))
            ct = ct.replace("$$PROCESSORS_ESTIMATE$$", str(processors_est))
            ct = ct.replace("$$MEMORY_ESTIMATE$$", str(memory_est))
            ct = ct.replace("$$CODE_PATH$$", str(architecture.code_path))
            ct = ct.replace("$$ARGS$$", architecture.make_args(hyperparams) + f" -db_file={db_file} -session_id={session_id} -arch_file={arch_file} ")
            
            return ct

        def get_newest_session(self, db_file, data_dir):
            
            architecture = Architecture.load(self.arch_file)

            def format_value(val):
                if type(val) == int:
                    return str(val)
                if type(val) == float:
                    return str(val)
                return "\"" + str(val) + "\""
                
            try:
                
                conn = sqlite3.connect(db_file)
                
                command = "SELECT session_id FROM " + architecture.name + " WHERE {0} ORDER BY start_time DESC;".format(" AND ".join(map("=".join,zip(self.hyperparams.keys(),map(format_value,self.hyperparams.values())))))
                cur = conn.cursor()
                cur.execute(command)
                result = cur.fetchall()
            except sqlite3.Error as error:
                print(error)
                return None
            finally:
                if (conn):
                    conn.close()

            if len(result)==0:
                return None
            session_id = result[0]
            if self.checker is None:
                return session_id
            for session_id in result:
                if(type(session_id) is tuple):
                    session_id = session_id[0]
                try:
                    self.checker(self.get_results(db_file,data_dir,session_id=session_id))
                except ModelLoadError as excep:
                    print("Incomplete Model: ", session_id,excep)
                    continue
                return session_id
            return None
        
        def get_data(self,db_file,data_dir,session_id=None):
            d = copy.copy(self.hyperparams)
            if session_id is None:
                session_id = self.get_newest_session(db_file)
            if session_id is None:
                print("Warining: No Session Found for Hyperparameters: ", self.hyperparams)
                return None
            
            d["session_id"] = session_id
            data_path = os.path.join(data_dir,f"{session_id}.data")
            if os.path.isfile(data_path):
                with open(data_path,'r') as f:
                    data = json.load(f)
                for key in data:
                    d[key] = data[key]
            return d

    ret = set([])
    for d in flatten_dict(models):
        if not "arch_file" in d:
            print("Must Specify arch_file")
            raise Exception
        if type(d["arch_file"]) is str:
            try:
                architecture = Architecture.load(d["arch_file"])
            except:
                print("Problem while Loading Architecture")
                raise Exception
            if not type(architecture) is Architecture:
                print("arch_file is not a valid Architecture")
        else:
            print("arch_file must be an Architecture File Path")
            raise(Exception)
        d.pop('arch_file', None)
        if not "checker" in d:
            print("Warning: No Checker specified, this model will accept any relevant session as completed.")
            checker = None
        elif:
            checker = d["checker"]
        d.pop('checker', None)
        ret.add(Model(arch_file,d))
    return ret

def process(grid,pipeline,cache_list, force=False):
    for model in grid:
        pass

def summarize(grid):
    pass

def run(grid,launch_settings):
    pass

def compute_metrics(metrics,grids=[],grid_dir=None,db_file=None,metric_dir=None,n_threads=1,force=False):

    def compute_metric(name, metric, session, metric_dir,force):
        if session is None:
            return
        
        session_id = session["session_id"]
        training_results = session["training_results"]
        
        if not callable(metric):
            metric_func, metric_args = metric
        filepath = os.path.join(metric_dir,f"{session_id}.metric")
        if is_in_saved_dict(filepath,name) and not force:
            return
        if not callable(metric):
            result = metric_func(training_results, metric_args)
        else:
            result = metric(training_results)
        if(type(result) is dict):
            for key in result:
                update_saved_dict(filepath,name+"_"+key,result[key])
        else:
            update_saved_dict(filepath,name,result)
    if grids==[]:
        grids = get_all_grids(grid_dir)
    if type(grids) is ModelGrid:
        grids = [grids]
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/sessions.db")
    if metric_dir is None:
        metric_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/Metrics/")

    sessions = [item for sublist in [grid.get_results(db_file) for grid in grids] for item in sublist]
    product = itertools.product(metrics.keys(),sessions)
    l_args = [(metric_name,metrics[metric_name],session,metric_dir,force) for metric_name, session in list(product)]
    run_in_parallel(target=compute_metric,l_args=set(l_args),n_threads=n_threads)

def get_results(self,db_file=None,metric_dir=None,grids=[],grid_dir=None):
    if grids==[]:
            grids = get_all_grids(grid_dir)
    if type(grids) is ModelGrid:
        grids = [grids]
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/sessions.db")
    if metric_dir is None:
        metric_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/Metrics/")
    flatten = lambda t: [item for sublist in t for item in sublist]
    pparams = []
    mmetrics = []
    groups = []
    group_num = 0
    results_hp = []
    results_m = []

    for d, arch in flatten([grid.get_results(db_file,metric_dir), grid.architecture for grid in grids]):
        filterByKey = lambda keys: {x: data[x] for x in keys}
        hyperparams = {x:d[x] for x in arch.hyperparameters.keys()}
        metrics = {x:d[x] for x in d if not x in hyperparams}
        if metrics is None:
            continue
        group = group_num
        for i, params in enumerate(pparams):
            same_group = True
            for key, _ in set(hyperparams.items()) ^ set(params.items()):
                if not key in self.average_over:
                    same_group = False
            if same_group:
                group = groups[i]
                break
        if group == group_num:
            group_num += 1
            results_hp += [{}]
            results_m += [{}]
        pparams += [hyperparams]
        mmetrics += [metrics]
        groups += [group]
        for param, value in hyperparams.items():
            if not param in average_over:
                results_hp[group][param] = value
    
    for i, metrics in enumerate(mmetrics):
        for metric in metrics:
            if metric in results_m[groups[i]]:
                results_m[groups[i]][metric] += [metrics[metric]]
            else:
                results_m[groups[i]][metric] = [metrics[metric]]

    for i, metrics in enumerate(results_m):
        metrics_copy = {}
        for name, l in metrics.items():
            metrics_copy[name] = l
            if not type(l) is list:
                continue
            if not all(map(lambda v: type(v) is float or type(v) is int,l)):
                continue
            else:
                mean = np.mean(np.array(l))
                std = np.std(np.array(l))
                metrics_copy[name] = mean
                metrics_copy[name+"_std"] = std
        results_m[i]=metrics_copy
    return [{**hp, **m} for hp, m in zip(results_hp,results_m)]



def query(experiment_results, select, where):
    ret = []
    for model in experiment_results:
        if all(where,lambda key: model[key] == where[key]):
            ret += [model[select]]
    return ret


def get_all_grids(grid_dir=None):
    if grid_dir is None:
        grid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/Grids/")
    ret = []
    for file in os.listdir(grid_dir):
        if not os.path.isfile(os.path.join(grid_dir, file)):
            continue
        if file.endswith(".grid"):
            ret += [ModelGrid.load(os.path.join(grid_dir,file))]
    
    return ret

def run_grids(launch_setting,env_parameters={},arch_dir=None,grid_dir=None,grids=[],n_threads=1, db_file=None, time_estimator=None, processors_estimator=None, memory_estimator=None, force=False):
        if db_file is None:
            db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/sessions.db")
        commands = []
        if grids==[]:
            grids = get_all_grids(grid_dir)
        if type(grids) is ModelGrid:
            grids = [grids]
        for grid in grids:
            commands += grid.make_commands(launch_setting,db_file, time_estimator, processors_estimator, memory_estimator,force=False,env_parameters=env_parameters,arch_dir=arch_dir)
        run_in_parallel(target=os.system,l_args=[(command,) for command in set(commands)],n_threads=n_threads)
