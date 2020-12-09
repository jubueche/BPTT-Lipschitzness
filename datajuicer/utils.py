import inspect
import copy
import concurrent.futures
import datajuicer.parser as parser
import random
import os.path
import datajuicer.in_out as in_out
import datajuicer.database as database
import time

#run_mode in normal, force, load

#cache_mode in normal, no_save

def dj(grid, func,n_threads=1, run_mode="normal", cache_mode="normal", cache_dir="Sessions/"):
    def _runner(grid, *args, **kwargs):
        def _run(data, func, args, kwargs):
            if type(func) == str:
                func = get(data,func)
                assert callable(func)
            for i, arg in enumerate(args):
                if type(arg) is str:
                    args[i] = format_template(data, arg)
            for kw, arg in kwargs.items():
                if type(arg) is str:
                    kwargs[kw]= format_template(data, arg)
            if not hasattr(func, "table_name"):
                return func(*args,**kwargs)
            
            boundargs = inspect.signature(func).bind(*args,**kwargs)
            boundargs.apply_defaults()

            dependencies = {}
            for i, dep_name in enumerate(func.dependencies):
                assert type(dep_name) is str
                if ":" in dep_name:
                    arg, key = dep_name.split(":", 1)
                    if key.isnumeric():
                        key = int(key)
                    reserved_keys = [name.split(":",1) for name in func.dependencies]
                    reserved_keys.pop(i)
                    reserved_keys = [item for sublist in reserved_keys for item in sublist]
                    if not key in reserved_keys:
                        dep_name = key
                    dependencies[dep_name] = boundargs.arguments[arg][key]
                else:
                    dependencies[dep_name] = boundargs.arguments[dep_name]

            session_ids = database.select(db_file = os.path.join(cache_dir, "sessions.db"),column = "session_id", table = func.table_name, where= dependencies, order_by="start_time")
            print(session_ids)
            sid = None
            for session_id in session_ids:
                checker = func.checker
                if checker is None:
                    checker = lambda *args: True
                if checker(session_id, func.table_name, cache_dir, data):
                    sid = session_id
                    break
            print(sid)
            print(sid is None or run_mode == "force")
            if sid is None or run_mode == "force":
                if run_mode == "load":
                    raise Exception("No Sessions Found")
                sid = random.randint(1000000000, 9999999999)
                data[func.table_name+"_session_id"] = sid
                row = copy.copy(dependencies)
                if cache_mode == "normal":
                    row["session_id"] = sid
                    row["start_time"] = int(time.time()*1000)
                    database.insert(db_file = os.path.join(cache_dir, "sessions.db"), table = func.table_name, row = row, primary_key = "session_id")
                ret = func(*args,**kwargs)
                if cache_mode == "no_save" or func.saver is None:
                    return ret
                
                return ret
            
            if not func.loader is None: return func.loader(sid, func.table_name, cache_dir, data)
            

        if type(grid) is dict:
            grid = [grid]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            param_list = [(copy.copy(data),func,list(args),kwargs) for data in grid]
            futures = [executor.submit(_run, *param) for param in param_list]

        return [f.result() for f in futures] 
    return lambda *args, **kwargs: _runner(grid, *args, **kwargs)

def djm(grid, func,n_threads=1, run_mode="normal", cache_mode="normal", cache_dir="Sessions/", store_key=None):
    runner = dj(grid, func, n_threads, run_mode,cache_mode,cache_dir)
    def _run_and_merge(*args, **kwargs):
        ret = runner(*args, **kwargs)
        if store_key is None:
            return [{**data, **retdata} for data, retdata in zip(grid,ret)]
        else:
            return [{**data, **{store_key:retdata}} for data, retdata in zip(grid,ret)]
        

def cachable(dependencies=None, saver=in_out.easy_saver, loader=in_out.easy_loader, checker=in_out.easy_checker, table_name=None):
    def decorator(func, dependencies, saver, loader, checker, table_name):
        if table_name is None:
            table_name = func.__name__
        if dependencies is None:
            dependencies = list(inspect.signature(func).parameters)
        func.dependencies = dependencies
        func.saver = saver
        func.loader = loader
        func.checker = checker
        func.table_name = table_name
        return func
    return lambda func: decorator(func, dependencies, saver, loader, checker, table_name)

def format_template(data, template):
        if template[0] == "{" and template[-1] == "}":
            return get(data, template[1:-1])
        else:
            return parser.replace(template, lambda k: str(get(data, k)))

def get(data, key):
    if key == "*":
        return data
    func_name, l_args = parser.get_arg_list(key)
    literal = False
    if func_name[0] == "!":
        literal = True
        func_name = func_name[1:]
    output = data[format_template(data, func_name)]
    while type(output) is str and "{" in output:
        output = format_template(data, output)
    for args in l_args:
        for i,arg in enumerate(args):
            args[i] = format_template(data, arg)
        output = output(*args)
        if not literal and type(output) is str:
            output = format_template(data, output)
    return output

def split(grid, key, values, where={}):
    if type(grid) is dict:
        grid = [grid]
    out = []
    for value in values:
        cg = [copy.copy(model) for model in grid]
        for d in cg:
            if all([d[kkey]==where[kkey] for kkey in where]):
                d[key] = value
        out += cg
    return out

def configure(grid, dictionary, where={}):
    cg = [copy.copy(model) for model in grid]
    for model in cg:
        if all([cg[key]==where[key] for key in where]):
            for key in dictionary:
                model[key] = dictionary[key]
    return cg


def query(grid, select, where):
    ret = []
    for model in grid:
        if all([model[key] == where[key] for key in where]):
            ret += [copy.copy(get(model, select))]
    return ret

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