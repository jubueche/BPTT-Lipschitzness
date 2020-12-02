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

#each dict in grid that does not have a matching session remains the same
#each dict in grid that does have a matching session gets updated with the session id and the model from that session if it exists
def get(action_name, grid, dependencies, db_file=None, checker = True):
    def format_value(val):
            if type(val) == int:
                return str(val)
            if type(val) == float:
                return str(val)
            return "\"" + str(val) + "\""
        
        
    if not callable(checker):
        checker= lambda _: checker
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Ressources/Sessions/sessions.db")
    sessions_dir = os.path.dirname(db_file)
    if type(grid) is dict:
        grid = [grid]
    unique = sorted(grid, key=lambda x: (id(type(x)), x))
    for model in unique:
        found = False
        #check in database to see if sessions exists
        try:
            conn = sqlite3.connect(db_file)
            
            command = "SELECT " + action_name + "_session_id FROM " + action_name + " WHERE "
            command += " AND ".join(map("=".join,[(dep, format_value(model[dep])) for dep in dependencies]))
            command += "ORDER BY start_time DESC;"
            
            cur = conn.cursor()
            cur.execute(command)
            result = [sid[0] for sid in cur.fetchall()] 
        except sqlite3.Error as error:
            print(error)
            
        finally:
            if (conn):
                conn.close()
        #use checker to see if sessions are complete
        for sid in result:
            if checker(copy.copy(model)):
                found = True
                model[action_name+"_session_id"] = sid
                break
            
        if found:
            model_path_json = os.path.join(sessions_dir, action_name + model[action_name+"_session_id"]+".json")
            model_path_pickle = os.path.join(sessions_dir, action_name + model[action_name+"_session_id"]+".pickle")
            if os.path.isfile(model_path_json):
                loaded = json.load(open(model_path_json, 'r'))
                assert(type(loaded) is dict)
                for key in loaded:
                    model[key] = loaded[key]
            if os.path.isfile(model_path_pickle):
                loaded = pickle.load(open(model_path_pickle, 'rb'))
                assert(type(loaded) is dict)
                for key in loaded:
                    model[key] = loaded[key]
    
    return unique


def __run_single(action_name,model,function,dependencies,db_file):
    if action_name+"_session_id" in model:
        return
    #if no complete sessions exist create a new session key
    model[action_name+"_session_id"] = random.randint(1000000000, 9999999999)
    fieldset = [f"'{action_name}_session_id' INTEGER PRIMARY KEY"]
    for dep in dependencies:
        val = model[dep]
        if type(val) == int:
            definition = "INTEGER"
        if type(val) == float:
            definition = "REAL"
        else:
            definition = "TEXT"
        
        fieldset.append("'{0}' {1}".format(dep, definition))

    create_table = "CREATE TABLE IF NOT EXISTS {0} ({1});".format(action_name, ", ".join(fieldset))
    #register session
    register = f"INSERT INTO {action_name} ({action_name}_session_id, "
    register += ", ".join(dependencies)
    register += f") VALUES("
    register += model[action_name+"_session_id"] + ", "
    def format_value(val):
        if type(val) == int:
            return str(val)
        if type(val) == float:
            return str(val)
        return "\"" + str(val) + "\""
    register += ", ".join([format_value(model[dep]) for dep in dependencies]) + " );"
    try:
        print("Registering session...")
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

    #call function
    function(model)

def __run_single_and_save(action_name,model,function,dependencies,db_file,file_format):
    __run_single(action_name,model,function,dependencies,db_file)
    sessions_dir = os.path.dirname(db_file)
    session_path = os.path.join(sessions_dir, action_name + model[action_name+"_session_id"]+"."+file_format)
    if os.path.isfile(session_path):
        if file_format == "json":
            with open(session_path, 'w+') as output_file:
                json.dump(model,output_file)
        if file_format == "pickle":
            with open(session_path, 'wb+') as output_file:
                pickle.dump(model,output_file)

#each dict in grid that does not have a session id gets a random session_id, is registered in the database, executes and the result is saved
def run(action_name,grid,function,dependencies,db_file=None,file_format="json",n_threads=1):
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Ressources/Sessions/sessions.db")
    if type(grid) is dict:
        grid = [grid]
    unique = sorted(grid, key=lambda x: (id(type(x)), x))
    for model in unique:
        model[action_name+"_n_threads"]=n_threads
    l_args = [(action_name,model,function,dependencies,db_file,file_format) for model in unique]
    __run_in_parallel(__run_single_and_save,l_args,n_threads)
    return unique

def run_command(action_name,grid,command,dependencies,db_file=None,n_threads=1):
    def execute_command(action_name, dependencies, model):
        def format_value(val):
            if type(val) in [int,float,str]:
                return str(val)
            else:
                return pickle.dumps(val)
        ct = model[action_name+"_command_template"]
        command = ""
        for i, part in enumerate(ct.split("$$")):
            if i%2==0:
                command+=part
            else:
                if part == "dependencies":
                    command += " " + " ".join(["-"+dep+"=" + format_value(model[dep]) for dep in dependencies]) + " "
                elif part in model:
                    command+=model[part]
                else:
                    raise Exception("Invalid Command")
        model[action_name+"_command"] = command
        os.system(command)
    
    if db_file is None:
        db_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Ressources/Sessions/sessions.db")
    if type(grid) is dict:
        grid = [grid]

    unique = sorted(grid, key=lambda x: (id(type(x)), x))
    for model in unique:
        model[action_name + "_command_template"] = command
    
    function = lambda model: execute_command(action_name, dependencies, model)
    
    for model in unique:
        model[action_name+"_n_threads"]=n_threads
    l_args = [(action_name,model,function,dependencies,db_file) for model in unique]
    __run_in_parallel(__run_single,l_args,n_threads)
    return unique

def split(grid, key, values):
    if type(grid) is dict:
        grid = [grid]
    unique = sorted(grid, key=lambda x: (id(type(x)), x))
    out = []
    for value in values:
        cg = copy.copy(unique)
        for d in cg:
            print(d)
            d[key] = value
        out += cg
    return out

def copy_key(grid,old_key,new_key):
    if type(grid) is dict:
        grid = [grid]
    unique = sorted(grid, key=lambda x: (id(type(x)), x))
    for model in unique:
        model[old_key] = model[new_key]
    return unique

def query(grid, select, where):
    ret = []
    for model in grid:
        if all(where,lambda key: model[key] == where[key]):
            ret += [model[select]]
    return ret