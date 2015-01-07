# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 16:20:59 2014

@author: Erin
"""

import numpy as np
import re

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
    
def create_trace_matrix(trace_dict, burnin=10000, thin=10, chain_num=0):
    trace_arr = np.zeros(((len(trace_dict[trace_dict.keys()[0]][chain_num])-burnin)/thin, len(trace_dict.keys())))
    for i, key in enumerate(trace_dict.keys()):
        trace_arr[:,i] = trace_dict[key][chain_num][burnin::thin]
    
    return trace_arr
    
def find_most_probable_vals(trace_array, trace_dict, axis=0):
    map_vals = {}
    u, indices = np.unique(trace_array, return_inverse=True)
    map_vector = u[np.argmax(np.apply_along_axis(np.bincount, axis, indices.reshape(trace_array.shape), None, np.max(indices) + 1), axis=axis)]
    for i, key in enumerate(trace_dict.keys()):
        map_vals[key] = map_vector[i]
    
    return map_vals
    
def sub_parameters(model, param_dict, log=True, KDs=True, generic_kf=1.5e4):
    generic_kf_log = np.log10(generic_kf)
    if KDs == True:
       for param, value in param_dict.items():
           if 'KD' in param:
               x = re.compile('(KD)(\S*)', re.IGNORECASE)
               y = x.search(param)
               param_name = y.groups()[1]
               kf_param_name = 'kf'+param_name
               kr_param_name = 'kr'+param_name
               if log == True:
                   model.parameters[kf_param_name].value = 10**generic_kf_log
                   print 'Changed parameter '+str(kf_param_name)+' to '+str(10**generic_kf_log)
                   model.parameters[kr_param_name].value = 10**(value+generic_kf_log)
                   print 'Changed parameter '+str(kr_param_name)+' to '+str(10**(value+generic_kf_log))
               else:
                   model.parameters[kf_param_name].value = generic_kf
                   print 'Changed parameter '+str(kf_param_name)+'to '+str(generic_kf)
                   model.parameters[kr_param_name].value = value*generic_kf
                   print 'Changed parameter '+str(kr_param_name)+' to '+str(value*generic_kf)
           else:
               if log == True:
                   model.parameters[param].value = 10**value
                   print 'Changed parameter '+str(param)+' to '+str(10**value)
               else:
                   model.parameters[param].value = value
                   print 'Changed parameter '+str(param)+' to '+str(value)
    else:
        for param, value in param_dict.items():
            if log == True:
                model.parameters[param].value = 10**value
                print 'Changed parameter '+str(param)+' to '+str(10**value)
            else:
                model.parameters[param].value = value
                print 'Changed parameter '+str(param)+' to '+str(value)
                
def check_thermoboxes(param_dict, log=True):
    thermo_dict = {}
    if log == True:
        box1 = (1/(10**param_dict['KD_AA_cat1']))*(1/(10**param_dict['KD_AA_allo2']))*(10**param_dict['KD_AA_cat3'])*(10**param_dict['KD_AA_allo1'])
        box2 = (1/(10**param_dict['KD_AA_allo1']))*(1/(10**param_dict['KD_AG_cat3']))*(10**param_dict['KD_AA_allo3'])*(10**param_dict['KD_AG_cat1'])
        box3 = (1/(10**param_dict['KD_AG_allo1']))*(1/(10**param_dict['KD_AA_cat2']))*(10**param_dict['KD_AG_allo2'])*(10**param_dict['KD_AA_cat1'])
        box4 = (1/(10**param_dict['KD_AG_cat1']))*(1/(10**param_dict['KD_AG_allo3']))*(10**param_dict['KD_AG_cat2'])*(10**param_dict['KD_AG_allo1'])
    
    else:
        box1 = (1/(param_dict['KD_AA_cat1']))*(1/(param_dict['KD_AA_allo2']))*(param_dict['KD_AA_cat3'])*(param_dict['KD_AA_allo1'])
        box2 = (1/(param_dict['KD_AA_allo1']))*(1/(param_dict['KD_AG_cat3']))*(param_dict['KD_AA_allo3'])*(param_dict['KD_AG_cat1'])
        box3 = (1/(param_dict['KD_AG_allo1']))*(1/(param_dict['KD_AA_cat2']))*(param_dict['KD_AG_allo2'])*(param_dict['KD_AA_cat1'])
        box4 = (1/(param_dict['KD_AG_cat1']))*(1/(param_dict['KD_AG_allo3']))*(param_dict['KD_AG_cat2'])*(param_dict['KD_AG_allo1'])     
        
    thermo_dict['box1'] = box1
    thermo_dict['box2'] = box2
    thermo_dict['box3'] = box3
    thermo_dict['box4'] = box4
    
    return thermo_dict



