import itertools
import threading

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

def add(d):
    d["bla"] = "bro"

dl = [({"id":1},),({"id":2},)]
run_in_parallel(add,dl,n_threads=1)
print(dl)