#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapted from experiments.py, courtesy of https://github.com/WvanWoerden/NTRUFatigue
"""

from __future__ import absolute_import
from sage.all import *
import copy
from collections import OrderedDict
from math import sqrt, log
import random

import six
from six.moves import range
import pprint

import numpy as np
from numpy import array, zeros, block, transpose
from numpy.linalg import slogdet
from scipy.stats import linregress

from fpylll import IntegerMatrix, BKZ, GSO
from fpylll.fplll.lll import LLLReduction
from bkz2_callback import BKZReduction
from common import DenseSubLatticeFound

from common import parse_args, run_all, pretty_dict, is_prime, next_prime, sqnorm, print_save_stats, run
from ntwe_gen_circulant import init_ntwe_lattice

import io
import datetime
import time

import warnings
warnings.filterwarnings(
    action='error', message='',
    category=RuntimeWarning
)

def ntwe_kernel(params, seed=None):
    if seed is None:
        params, seed = params
    
    random.seed(seed)    # If the seeding does not work, probably current time is used. This is to prevent all threads from sampling the exact same randomness.
    time.sleep(random.random()) 
    
    # Pool.map only supports a single parameter
    params = copy.copy(params)

    n = params["n"]
    q = params["q"]
    float_type = params["float_type"]
    tours = params["tours"]
    sigmasq = params["sigmasq"]

    C, D, _, _, _ = init_ntwe_lattice(n, q, np.sqrt(sigmasq), seed=seed) # C is basis, D is dense
    dual = D @ np.linalg.inv(np.transpose(D) @ D) # Compute dual of dense

    A = IntegerMatrix.from_matrix([[int(x) for x in v] for v in C.T])
    M = GSO.Mat(A, float_type=float_type)
    lll = LLLReduction(M)
    lll()
    bkz = BKZReduction(M)
    M.update_gso()

    sk_norms = [ sqnorm(D[: , i]) for i in range(n) ]
    sk_norm_min = min(sk_norms)
    sk_norm_max = max(sk_norms)
    
    DS_vol = slogdet(transpose(D).dot(D))[1]/2.
    temp = np.zeros((bkz.M.B.nrows, bkz.M.B.ncols))

    def insert_callback(call_stack, solution):
        kappa, b = call_stack[-1]
        assert b==len(solution)
        
        bkz.M.B.to_matrix(temp)
        # Write in cannonical basis
        v = solution @ temp[kappa : kappa + b, :]
        # babai-reduce it
        lift_can = bkz.M.babai(v, 0, kappa) @ temp[0 : kappa, : ]
        
        v = array(v) - array(lift_can)
    
        # Test if in dense sublattice
        # https://crypto.stackexchange.com/questions/83488/verify-that-a-point-is-inside-a-lattice # evenuellt multiplicera med q
        y = v @ dual
        if any(np.abs(np.round(y) - y) > 1e-6): return

        if sqnorm(v) < sk_norm_min: return
        # Distinguishes between SKR and DSD
        lf = sqnorm(v) / sk_norm_max

        # only run if we have dense sublattice
        vg = bkz.M.from_canonical(v, start=0, dimension=kappa+b)
        vgs = np.multiply(np.square(vg),bkz.M.r()[0 : kappa + b])

        raise DenseSubLatticeFound(call_stack, lf, v, vgs, solution, bkz.M.r())

    bkz.insert_callback = insert_callback

    for blocksize in list(range(2, n * 2 + 1)):

        if tours==None:
            tours = 8

        par = BKZ.Param(blocksize,
                              strategies=BKZ.DEFAULT_STRATEGY,
                              flags=BKZ.BOUNDED_LLL,
                              max_loops=tours)
        try:
            bkz(par)
            print(par)
        except DenseSubLatticeFound as err:
            kappa, b = err.call_stack[0]
            assert (b==blocksize) or (kappa+b == 3 * n)
            # subkappa, subb = err.call_stack[-1]
            vsz = np.sum(np.abs(err.vloc))
            logr = [log(x)/2. for x in err.gso]
            # d_s=len(err.gso)

            slope_part = min(30, n)
            l = n-slope_part
            r = n+slope_part # TODO: fix this. Slope is not computed correctly
            slope=linregress(range(l, r), logr[l:r]).slope
            byLLL = vsz<1.5

            sq_proj_sz = np.sum(err.vgs[kappa:kappa+b])/np.sum(err.vgs[:kappa+b])

            if (err.lf>1.):
                stats = {"DSD": True,   "DSD_lf": err.lf,  "kappa": kappa, "beta":blocksize, "DS_vol":DS_vol, "foundbyLLL": byLLL, "slope": slope, "sqproj_rel": sq_proj_sz}
                print("DSD vector", err.vcan)
            else:
                stats = {"DSD": False,   "DSD_lf": 1., "kappa": kappa, "beta":blocksize, "DS_vol": DS_vol, "foundbyLLL": byLLL, "slope": slope, "sqproj_rel": sq_proj_sz}

            return stats


if __name__ == "__main__":
    run(ntwe_kernel)
