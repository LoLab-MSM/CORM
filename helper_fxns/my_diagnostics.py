# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 16:20:59 2014

@author: Erin
"""

import numpy as np
import re
from pymc.stats import hpd
import matplotlib.pyplot as plt
import math

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
    
def create_trace_matrix(trace_dict, burnin=10000, thin=10, chain_num='all'):
    if chain_num == 'all':
        trace_arr = np.zeros(((((len(trace_dict[trace_dict.keys()[0]][0])-burnin)*len(trace_dict[trace_dict.keys()[0]]))/thin), len(trace_dict.keys())))
    else:
        trace_arr = np.zeros(((len(trace_dict[trace_dict.keys()[0]][chain_num])-burnin)/thin, len(trace_dict.keys())))
    for i, key in enumerate(trace_dict.keys()):
        if chain_num == 'all':
            chain_list = [trace_dict[key][j][burnin::thin] for j in range(len(trace_dict[key]))]
            trace_arr[:,i] = np.concatenate(chain_list)
        else:
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

def calc_credible_intervals(trace_arr, trace_dict, alpha=.05):
    hpd_dict = {}
    
    for i, key in enumerate(trace_dict.keys()):
        hpd_dict[key] = hpd(trace_arr[:,i], alpha)
        
    return hpd_dict

def credible_interval_array(hpd_dict):
    hpd_array = np.zeros((len(hpd_dict), len(hpd_dict[hpd_dict.keys()[0]])))
    i = 0
    for var in hpd_dict.keys():
        hpd_array[i] = hpd_dict[var]
        i += 1

    return hpd_array
    
def plot_histograms(trace_dict, trace_arr, bins=10, plot_original_vals=False, model=None, param_name_change_dict=None, max_yval = .5, title_fontsize=20):
    if plot_original_vals==True and model==None:
        raise Exception('Model needs to be specified')
    
    fig_list = []
    
    #fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    #fig2, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3, 2)
    
    n = 0
    fig, axarr = plt.subplots(3, 2)
    axarr = axarr.flatten()
    xlow = np.min(trace_arr)
    xhigh = np.max(trace_arr)
    for var_num, variable in enumerate(trace_dict):
        weights = np.ones_like(trace_arr[:,var_num])/len(trace_arr[:,var_num])
        axarr[n].hist(trace_arr[:,var_num], histtype='stepfilled', bins=bins, color='k', weights=weights)
        if plot_original_vals==True:
            axarr[n].vlines(np.log10(model.parameters[variable].value), 0, .5)
        if param_name_change_dict is not None:
            axarr[n].set_title(str(param_name_change_dict[variable]), fontsize=title_fontsize)
        else:
            axarr[n].set_title(str(variable), fontsize=title_fontsize)
        axarr[n].set_xlabel('Log(parameter value)')
        axarr[n].set_ylabel('Probability')
        axarr[n].set_xlim(np.ceil(xlow-1), np.ceil(xhigh))
        axarr[n].set_ylim(0, max_yval)
        n += 1
        if n == 6:
            fig.subplots_adjust(wspace=.5, hspace=1)
            fig_list.append(fig)
            fig, axarr = plt.subplots(3, 2)
            axarr = axarr.flatten()
            n = 0
        
    #fig1.subplots_adjust(wspace=.5, hspace=1)
    #fig2.subplots_adjust(wspace=.5, hspace=1)

    return fig_list
    
def sample_plots(trace_dict):
    fig_list = []
    
    n = 0
    nkeys = len(trace_dict.keys())
    fig, axarr = plt.subplots(3, 2)
    axarr = axarr.flatten()
    iterations = range(len(trace_dict[trace_dict.keys()[0]][0]))
    print len(iterations)
    print len(trace_dict[trace_dict.keys()[0]])
    print len(trace_dict[trace_dict.keys()[0]][0])

    for variable in trace_dict:
        for chain in range(len(trace_dict[trace_dict.keys()[0]])):
            axarr[n].plot(iterations, trace_dict[variable][chain])
        axarr[n].set_title(str(variable))
        axarr[n].set_xlabel('Iterations')
        axarr[n].set_ylabel('Sampled value')
        axarr[n].set_xlim(0, len(iterations))
        #axarr[n].set_ylim(0, .22)
        n += 1
        if n == 6 or n==nkeys:
            fig.subplots_adjust(wspace=.5, hspace=1)
            fig_list.append(fig)
            fig, axarr = plt.subplots(3, 2)
            axarr = axarr.flatten()
            n = 0
        
    #fig1.subplots_adjust(wspace=.5, hspace=1)
    #fig2.subplots_adjust(wspace=.5, hspace=1)

    return fig_list

def create_ratio_trace_dict(trace_dict):
    trace_ratio_dict = {}
    for param_num_1, parameter_1 in enumerate(trace_dict.keys()):
        for param_num_2, parameter_2 in enumerate(trace_dict.keys()[param_num_1+1::]):
            key_name = 'ratio_'+str(parameter_1)+'_'+str(parameter_2)
            trace_ratio_dict[key_name] = [trace_dict[parameter_1][i]/trace_dict[parameter_2][i] for i in range(len(trace_dict[parameter_1]))]
    
    return trace_ratio_dict
    
def convert_param_vec_dict_to_param_dict(param_vec_dict, pysb_parameter_list):
    param_dict = {}
    for param_num, param in enumerate(pysb_parameter_list):
        param_dict[param.name] = [param_vec_dict[param_vec_dict.keys()[0]][chain][:,param_num] for chain in range(len(param_vec_dict[param_vec_dict.keys()[0]]))]
    
    return param_dict
    
def plot_tsne_data(unique_tsne_output, original_trace_arr, histogram=True, hexbin_gridsize=None):
    if histogram == True:
        original_trace_arr = np.ascontiguousarray(original_trace_arr)
        unique_vecs, inv_idx = np.unique(original_trace_arr.view([('', original_trace_arr.dtype)]*original_trace_arr.shape[1]), return_inverse=True)
        full_tsne_output = unique_tsne_output[inv_idx]
        if hexbin_gridsize != None:
            fig = plt.hexbin(full_tsne_output[:,0], full_tsne_output[:,1], gridsize=hexbin_gridsize)
        else:
            fig = plt.hexbin(full_tsne_output[:,0], full_tsne_output[:,1])
    else:
        fig = plt.scatter(unique_tsne_output[:,0], unique_tsne_output[:,1])
    
    return fig
        