# -*- coding: utf-8 -*-
"""
Common functions, adapted from https://github.com/WvanWoerden/NTRUFatigue
"""
from __future__ import absolute_import
from __future__ import print_function
from sage.all import *
import argparse
import copy
import datetime
import re
import io
from collections import OrderedDict
from multiprocessing import Pool
import six
from six.moves import range
from random import randint
from math import sqrt
import numpy as np
import pprint
import time

def is_prime(x):
    return all(x % i for i in range(2, int(sqrt(x))+1 ))

def next_prime(x):
    return min([a for a in range(x+1, 2*x) if is_prime(a)])

def sqnorm(a):
    return np.inner(a, a)

def run_all(f, params_list, trials=1, workers=1, seed=None):
    """Call ``f`` on all combination of listed parameters

    :param params_list: run ``f`` for all parameters given in ``params_list``
    :param trials: number of experiments to run per dimension
    :param workers: number of parallel experiments to run
    """
    if seed is None:
        seed = randint(0, 2**31)
    jobs, stats = [], OrderedDict()
    for params in params_list:
        stats[str(params)] = []
        for t in range(trials):
            seed += 1
            args = (params, seed)
            jobs.append(args)

    if workers == 1:
        for job in jobs:
            params, seed_ = job
            res = f(copy.deepcopy(job))
            stats[str(params)].append(res)            

    else:
        pool = Pool(workers)
        for i, res in enumerate(pool.map(f, jobs)):
            params, seed_ = jobs[i]
            stats[str(params)].append(res)            

    return stats

def parse_args(description, **kwds):
    """
    Parse command line arguments.

    The command line parser accepts the standard parameters as printed by calling it with
    ``--help``.  All other parameters are used to construct params objects.  For example.

    ./foo --workers 4 --trials 2 -S 1337 --a 1 2 - b 3 4

    would operate on dimension 80 with parameters (a: 1, b: 3), (a: 1, b: 4), (a: 2, b: 3), (a: 2,
    b: 4), i.e. the Cartesian product of all parameters.  It will run two trials each using four
    workers. Note that each worker may use several threads, too. The starting seed is `1337`.

    :param description: help message
    :param kwds: default parameters

    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--trials', type=int, dest="trials", default=1,
                        help="number of experiments to run per dimension")
    parser.add_argument('-w', '--workers', type=int, dest="workers", default=1,
                        help="number of parallel experiments to run")
    parser.add_argument('-f', '--full_data', type=bool, dest="full_data", default=0,
                        help="print out full data")
    parser.add_argument('-save', '--save_csv', type=bool, dest="save", default=0,
                        help="save csv data to timestamp.csv (0/1)")

    args, unknown = parser.parse_known_args()

    all_params = OrderedDict([("", kwds)])
    unknown_args = OrderedDict()

    # NOTE: This seems like the kind of thing the standard library can do (better)
    i = 0
    while i < len(unknown):
        k = unknown[i]
        if not (k.startswith("--") or k.startswith("-")):
            raise ValueError("Failure to parse command line argument '%s'"%k)
        k = re.match("^-+(.*)", k).groups()[0]
        k = k.replace("-", "_")
        unknown_args[k] = []
        i += 1
        for i in range(i, len(unknown)):
            v = unknown[i]
            if v.startswith("--") or v.startswith("-"):
                i -= 1
                break
            
            try:
                L = re.match("([0-9]+)~([0-9]+)p([0-9]+)?", v).groups()
                # print(range(int(L[0]) - 1, int(L[1]) - 1, int((int(L[1]) - int(L[0]))/int(L[2]))))
                if L[2] is not None:
                    v = [ next_prime(x) for x in range(int(L[0]) - 1, int(L[1]) - 1, int((int(L[1]) - int(L[0]))/int(L[2]))) ]
                else:
                    v = [x for x in range(int(L[0]), int(L[1])) if is_prime(x)]
                # print(v)
                unknown_args[k].extend(v)
                continue
            except:
                pass

            # try:
            #     L = re.match("([0-9]+)~([0-9]+)p", v).groups()
            #     v = [x for x in range(int(L[0]), int(L[1])) if is_prime(x)]
            #     unknown_args[k].extend(v)
            #     continue
            # except:
            #     pass

            try:
                L = re.match("([0-9]+)~([0-9]+)~?([0-9]+)?", v).groups()
                if L[2] is not None:
                    v = range(int(L[0]), int(L[1]), int(L[2]))
                else:
                    v = range(int(L[0]), int(L[1]))
                unknown_args[k].extend(v)
                continue
            except:
                pass
            
            try:
                unknown_args[k].append(int(v))
                continue
            except:
                pass

            try:
                unknown_args[k].append(float(v))
                continue
            except:
                unknown_args[k].append(v)


        i += 1
        if not unknown_args[k]:
            unknown_args[k] = [True]

    for k, v in six.iteritems(unknown_args):
        all_params_ = OrderedDict()
        for p in all_params:
            for v_ in v:
                p_ = copy.copy(all_params[p])
                p_[k] = v_
                all_params_[p+"'%s': %s, "%(k, v_)] = p_
        all_params = all_params_

    return args, all_params



def pretty_dict(d):
    s = ""
    for x,y in six.iteritems(dict(d)):
        if x=="float_type" or x=="full_data": continue

        if type(x)==float:
            s += "%s: %.3f \t"%(x,y)
        elif type(x)==int or type(x)==bool:
            s += "%s: %d \t"%(x,y)
        else:
            s += "%s: %s \t"%(x,str(y))


    return s


def print_save_stats(stats, full = True, save = False):
    
    # aggregate data for summary
    print("\n\n AVERAGE DATA\n\n ")
    for x,y in six.iteritems(stats):
        i = [ j for j, val in enumerate(y) if val is not None ]
        y = [ y[j] for j in i]
        print(pretty_dict(eval(x)))
        if len(y) > 0:
            avg = OrderedDict()
            for z in y:
                for k in z:
                    avg[k] = avg.get(k, 0) + z[k]/len(y)

            keys = (y[0].keys())
            for k in keys:
                print("%14s, "%k, end="")
            print()
            for k in keys:
                print("%14s, "%("%.4f"%avg[k]), end="")
            print()
        else:
            print("-- NO DATA --")

    full_data_string = io.StringIO()

    for x, y in six.iteritems(stats):
        i = [ j for j, val in enumerate(y) if val is not None ]
        y = [ y[j] for j in i]

        full_data_string.write(pretty_dict(eval(x)) + "\n")
        if len(y) > 0:
            keys = y[0].keys()
            full_data_string.write(", ".join(f"{k:14}" for k in keys) + "\n")
            for z in y:
                full_data_string.write(", ".join(f"{z[k]:14.4f}" for k in keys) + "\n")
        else:
            full_data_string.write("-- NO DATA --\n")

    full_data_string = full_data_string.getvalue()

    if full:
        print("\n\n FULL DATA (csv format)\n\n ")
        print(full_data_string)

    if save:
        fname = f'{datetime.datetime.now()}.csv'
        with open(fname, 'w') as f:
            f.write(full_data_string)

class DenseSubLatticeFound(Exception):
    def __init__(self, call_stack, lf, vcan, vgs, vloc, gso):
        self.call_stack = call_stack
        self.lf = lf
        self.vcan = vcan
        self.vgs = vgs
        self.vloc = vloc
        self.gso = gso

def run(kernel = None):

    if "ntru" in kernel.__name__:
        args, all_params = parse_args("Attempt to solve an ntru challenge.",
                                    float_type = "double",
                                    circulant = True,
                                    tours=8,
                                    sigmasq=0.667,
                                    )
    elif "mlwe" in kernel.__name__:
        args, all_params = parse_args("Attempt to solve an MLWE challenge.",
                                    float_type = "double",
                                    tours=8,
                                    sigmasq=0.667,
                                    )
    elif "ntwe" in kernel.__name__:
        args, all_params = parse_args("Attempt to solve an NTWE challenge.",
                                    float_type = "double",
                                    tours=8,
                                    sigmasq=0.667,
                                    )
    else:
        raise NotImplementedError("Please specify an implemented kernel")
    
    pprint.pprint(args)
    print(pretty_dict(all_params))

    t1 = time.perf_counter()
    stats = run_all(kernel, list(all_params.values()), # noqa
                    trials=args.trials,
                    workers=args.workers)

    t1 = time.perf_counter() - t1

    print_save_stats(stats, args.full_data, args.save)
    
    print(f"Time: {t1} s")