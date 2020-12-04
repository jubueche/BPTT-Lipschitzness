import argparse
import os
import pickle
import copy
import sqlite3
import random
import threading
import itertools
import ujson as json
import time
import re


class Architecture:
    @staticmethod
    def default_hyperparameters():
        pass
    
    @staticmethod
    def environment_parameters(mode):
        pass
    
    @staticmethod
    def launch_settings(mode):
        return {
            "launch":"python {code_file} {make_args}"
        }
    
    @staticmethod
    def checker(model):
        pass


    @classmethod
    def make(cls, mode):
        ret = {**cls.default_hyperparameters(), **cls.environment_parameters(mode), **cls.launch_settings(mode)}
        def make_args(model):
            keys = list({**cls.default_hyperparameters(), **cls.environment_parameters(mode)}.keys())
            return " ".join([" -" + key + "=" + str(model[key])+" " for key in keys]) + " -session_id="+str(model[cls.__name__+"_session_id"])+" "
        ret["make_args"] = make_args
        ret[cls.__name__+"_dependencies"] = list(cls.default_hyperparameters().keys())
        ret[cls.__name__ + "_checker"] = cls.checker
        ret[cls.__name__+"_function"] = lambda model, command: os.system(command)
        ret["architecture"] = cls.__name__
        return ret

    @staticmethod
    def help():
        return {}

    @classmethod
    def get_flags(cls, mode):
        help=cls.help()
        parser = argparse.ArgumentParser()
        for key, value in {**cls.default_hyperparameters(), **cls.environment_parameters(mode)}.items():
            parser.add_argument("-" + key,type=type(value),default=value,help=help.get(key,""))
        parser.add_argument("-session_id", type=int, default = 0)
        
        flags = parser.parse_args()
        if flags.session_id==0:
            flags.session_id = random.randint(1000000000, 9999999999)
        return flags

    @staticmethod
    def log(session_id, key, value, save_dir = None):
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resources/TrainingResults/")
        file = os.path.join(save_dir, str(session_id) + ".train")
        exists = os.path.isfile(file)
        directory = os.path.dirname(file)
        if not os.path.isdir(directory):
            os.makedirs(directory)
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

def run(action,grid,*args,n_threads=1,format_args=True):
    if type(grid) is dict:
        grid = [grid]
    assert len(grid) > 0
    new = [copy.copy(model) for model in grid]
    args = list(args)
    def format_and_run(model, function, args, format_args):
        if format_args:
            for i, arg in enumerate(args):
                if type(arg) == str:
                    args[i] = format_template(model,arg)
        print(*args)
        function(model, *args)

    def conditional_run(model, action, args, format_args):
        action = format_template(model,action)
        if action+"_dependencies" in model:
            load(model, action)
        if not action+"_session_id" in model:
            if action+"_dependencies" in model: register(model, action)
            if not action+"_function" in model:
                raise Exception("No function found for action "+action)
            
            format_and_run(model,model[action+"_function"],args, format_args)
        

    if type(action) is str:
        function=conditional_run
    elif callable(action):
        function=format_and_run
    __run_in_parallel(function,[(model,action,args,format_args) for model in new],n_threads)
    return new

def __run_in_parallel(target,l_args,n_threads):
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    if n_threads==1:
        for args in l_args:
            target(*args)
        return
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

def make_unique(grid):
    def try_to_serialize(value):
        try:
            return str(value)
        except:
            try:
                return json.dumps(value)
            except:
                try:
                    return pickle.dumps(value)
                except:
                    return ""
    sgrid = list(map(lambda model: "".join([try_to_serialize(pair) for pair in model.items()]), grid))
    ret = []
    for i in range(len(sgrid)):
        for j in range(i):
            if sgrid[j] == sgrid[i]:
                continue
        ret+=[copy.copy(grid[i])]
    return ret


def _get_dependencies(model, action):
    if not action+"_dependencies" in model:
        raise Exception("Cannot find " + action+"_dependencies")
    return [format_template(model,dep) for dep in model[action+"_dependencies"]]

def _get_db_file(model, action):
    if not action+"_db_file" in model:
        if not "db_file" in model:
            db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Sessions/sessions.db")
        else:
            db_file = format_template(model,model["db_file"])
    else:
        db_file = format_template(model,model[action+"_db_file"])
    directory = os.path.dirname(db_file)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return db_file

def _get_sessions_dir(model, action):
    if not action+"_sessions_dir" in model:
        if not "sessions_dir" in model:
            directory =  os.path.dirname(_get_db_file(model,action))
        else:
            directory =  format_template(model, model["sessions_dir"])
    else:
        directory = format_template(model, model[action+"_sessions_dir"])
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory

def _get_checker(model, action):
    if action+"_checker" in model:
        checker = model[action+"_checker"]
    elif "checker" in model:
        checker = model["checker"]
    else:
        checker = True
    if not callable(checker):
        checker= lambda _: checker
    return checker

def _get_file_format(model, action):
    if not action+"_file_format" in model:
        if not "file_format" in model:
            return "json"
        return model["file_format"]
    return model[action+"_file_format"]

#each dict in grid that does not have a matching session remains the same
#each dict in grid that does have a matching session gets updated with the session id and the model from that session if it exists
def load(model, action):
    def format_value(val):
            if type(val) == int:
                return str(val)
            if type(val) == float:
                return str(val)
            return "\"" + str(val) + "\""
    db_file = _get_db_file(model, action)
    dependencies = _get_dependencies(model,action)
    checker = _get_checker(model, action)
    ret = 'no session'
    #check in database to see if sessions exists
    try:
        conn = sqlite3.connect(db_file)
        
        command = "SELECT " + action + "_session_id FROM " + action + " WHERE "
        command += " AND ".join(map("=".join,[(dep, format_value(model[dep])) for dep in dependencies]))
        command += f" ORDER BY {action}_start_time DESC;"
        
        cur = conn.cursor()
        cur.execute(command)
        result = [sid[0] for sid in cur.fetchall()] 
    except sqlite3.Error as error:
        print(error)
        return 'no session'
    
    if (conn):
        conn.close()
    
    ret = 'incomplete model'
    
    #use checker to see if sessions are complete
    for sid in result:
        if checker(copy.copy(model)):
            model[action+"_session_id"] = sid
            ret = load_file(model, action)
            break  
    
    return ret

def load_file(model, action):
    sessions_dir =_get_sessions_dir(model,action)
    model_path_json = os.path.join(sessions_dir, action + str(model[action+"_session_id"])+".json")
    model_path_pickle = os.path.join(sessions_dir, action + str(model[action+"_session_id"])+".pickle")
    ret = 'no file'
    if os.path.isfile(model_path_json):
        loaded = json.load(open(model_path_json, 'r'))
        assert(type(loaded) is dict)
        for key in loaded:
            model[key] = loaded[key]
        ret = 'found file'
    if os.path.isfile(model_path_pickle):
        loaded = pickle.load(open(model_path_pickle, 'rb'))
        assert(type(loaded) is dict)
        for key in loaded:
            model[key] = loaded[key]
        ret = 'found file'
    return ret

def checker(func, correct_vals):
    def tryer(model, func, correct_vals):
        try:
            ret = func(model)
        except:
            return False
        return ret in correct_vals
    if correct_vals==():
        correct_vals = [True, None]
    if not type(correct_vals) in [list, tuple]:
        correct_vals = [correct_vals]
    return lambda model: tryer(model, func, correct_vals)

def register(model, action):
    db_file = _get_db_file(model, action)

    dependencies = _get_dependencies(model, action)
    
    if action+"_session_id" in model:
        return
    #if no complete sessions exist create a new session key
    model[action+"_session_id"] = random.randint(1000000000, 9999999999)
    fieldset = [f"'{action}_session_id' INTEGER PRIMARY KEY", action+'_start_time INTEGER']
    for dep in dependencies:
        val = model[dep]
        if type(val) == int:
            definition = "INTEGER"
        if type(val) == float:
            definition = "REAL"
        else:
            definition = "TEXT"
        
        fieldset.append("'{0}' {1}".format(dep, definition))

    create_table = "CREATE TABLE IF NOT EXISTS {0} ({1});".format(action, ", ".join(fieldset))
    #register session
    register = f"INSERT INTO {action} ({action}_session_id, "
    register += ", ".join(dependencies) + f", {action}_start_time "
    register += f") VALUES("
    register += str(model[action+"_session_id"]) + ", "
    def format_value(val):
        if type(val) == int:
            return str(val)
        if type(val) == float:
            return str(val)
        return "\"" + str(val) + "\""
    register += ", ".join([format_value(model[dep]) for dep in dependencies])
    register +=", "+ str(int(time.time()*1000))+ " );"
    try:
        print("Registering session...")
        print(db_file)
        conn = sqlite3.connect(db_file, timeout=100)
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


def save(model, action):
    sessions_dir = _get_sessions_dir(model, action)
    file_format = _get_file_format(model, action)
    session_path = os.path.join(sessions_dir, action + str(model[action+"_session_id"]) +"."+file_format)
    if os.path.isfile(session_path):
        if file_format == "json":
            with open(session_path, 'w+') as output_file:
                json.dump(model,output_file)
        if file_format == "pickle":
            with open(session_path, 'wb+') as output_file:
                pickle.dump(model,output_file)


def format_template(model, template):
    if not type(template) is str:
        raise Exception("Invalid result")
    if not "{" in template:
        return template
    
    l = []
    depth = 0
    br = [m.start() for m in re.finditer(r"[\{,\}]", template)]
    for i in br:
        if template[i] == "{":
            depth+=1
            if depth==1:
                l.append(i)
        if template[i] == "}":
            depth-=1
            if depth ==0:
                l.append(i)
        if depth<0:
            raise Exception("Invalid Template")
    l=[0] + [val for pair in zip(l, l) for val in pair] + [len(template)]
    def pairwise(iterable):
        "s -> (s0, s1), (s2, s3), (s4, s5), ..."
        a = iter(iterable)
        return zip(a, a)
    result = ""
    for i, delim in enumerate(pairwise(l)):
        start, end = delim
        start_part = start
        if template[start] in ["{","}"]:
            start_part = start+1
        if i%2==0:
            result+=template[start_part:end]
            continue
        
        if template[start]==template[end]:
            raise Exception("Invalid Template")
        
        part = template[start_part:end]
        if part in model:
            if type(model[part]) in [int,float]:
                result += str(model[part])
            elif type(model[part]) is str:
                result += format_template(model,model[part])
            elif callable(model[part]):
                result += format_template(model,model[part](model))
            else:
                result += pickle.dumps(model[part])
        else:
            result += part
    return format_template(model, result)


def split(grid, key, values):
    if type(grid) is dict:
        grid = [grid]
    out = []
    for value in values:
        cg = [copy.copy(model) for model in grid]
        for d in cg:
            d[key] = value
        out += cg
    return out

def configure(grid, dictionary):
    def update(model, d):
        for key in d:
            model[key] = d[key]
    return run(lambda model: update(model, dictionary), grid)

def copy_key(model):
    old_key, new_key = model["_args"]
    model[old_key] = model[new_key]

def query(grid, select, where):
    ret = []
    for model in grid:
        if all(where,lambda key: model[key] == where[key]):
            ret += [model[select]]
    return ret
