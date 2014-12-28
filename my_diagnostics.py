# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 16:20:59 2014

@author: Erin
"""

import numpy as np

def gelman_rubin_trace_dict(trace_dict):
    Rhat = {}
    def calc_rhat(var_dict):
        # a ValueError that will handle the multidimensional case
        n = len(var_dict[0])

        # Calculate between-chain variance
        B = n * np.var(np.mean(var_dict, axis=1), ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(var_dict, axis=1, ddof=1))

        # Estimate of marginal posterior variance
        Vhat = W*(n - 1)/n + B/n

        return np.sqrt(Vhat/W)
        
    for var in trace_dict:
        Rhat[var] = calc_rhat(trace_dict[var])
    
    return Rhat
        