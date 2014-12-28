# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:26:46 2014

@author: Erin
"""

import pymc as pm
from pysb.integrate import Solver
import numpy as np
import theano
import theano.tensor as t
import pickle

from basic_COX2_model import model as cox2_model

model = pm.Model()

#Initialize PySB solver object for simulations
tspan = np.linspace(0,10, num=100)
solver = Solver(cox2_model, tspan)

#Add import of experimental data here
exp_data_PG = np.loadtxt('exp_data_pg.txt')
exp_data_PGG = np.loadtxt('exp_data_pgg.txt')

exp_data_sd_PG = np.loadtxt('exp_data_sd_pg.txt')
exp_data_sd_PGG = np.loadtxt('exp_data_sd_pgg.txt')

#Experimental starting values of AA and 2-AG (all in microM)
exp_cond_AA = [0, .5, 1, 2, 4, 8, 16]
exp_cond_AG = [0, .5, 1, 2, 4, 8, 16]

#Experimentally measured parameter values
KD_AA_cat1 = np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value)
kcat_AA1 = np.log10(cox2_model.parameters['kcat_AA1'].value)
KD_AG_cat1 = np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value)
kcat_AG1 = np.log10(cox2_model.parameters['kcat_AG1'].value)
KD_AG_allo2 = np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value)

#Likelihood function to generate simulated data that corresponds to experimental time points
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar] \
,otypes=[t.dmatrix, t.dmatrix])
def likelihood(KD_AA_cat2, kcat_AA2, KD_AA_cat3, kcat_AA3, \
    KD_AG_cat2, KD_AG_cat3, kcat_AG3, KD_AA_allo1, KD_AA_allo2, KD_AA_allo3, KD_AG_allo1, KD_AG_allo3):
    
#    print 'KD_AA_cat2: ',KD_AA_cat2
#    print 'kcat_AA2: ',kcat_AA2
#    print 'KD_AA_cat3: ',KD_AA_cat3
#    print 'kcat_AA3: ',kcat_AA3
#    print 'KD_AG_cat2: ',KD_AG_cat2
#    print 'KD_AG_cat3: ',KD_AG_cat3
#    print 'kcat_AG3: ',kcat_AG3
#    print 'KD_AA_allo1: ',KD_AA_allo1
#    print 'KD_AA_allo2: ',KD_AA_allo2
#    print 'KD_AA_allo3: ',KD_AA_allo3
#    print 'KD_AG_allo1: ',KD_AG_allo1
#    print 'KD_AG_allo3: ',KD_AG_allo3
    
    #generic kf in units of inverse microM*s (matches model units)
    generic_kf = np.log10(1.5e4)
    
    #Sub in parameter values at current location in parameter space
#    cox2_model.parameters['kf_AA_cat1'].value = 10**generic_kf
#    cox2_model.parameters['kr_AA_cat1'].value = 10**(KD_AA_cat1*generic_kf)
#    cox2_model.parameters['kcat_AA1'].value = 10**kcat_AA1
    cox2_model.parameters['kf_AA_cat2'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_cat2'].value = 10**(KD_AA_cat2+generic_kf)
    cox2_model.parameters['kcat_AA2'].value = 10**kcat_AA2
    cox2_model.parameters['kf_AA_cat3'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_cat3'].value = 10**(KD_AA_cat3+generic_kf)
    cox2_model.parameters['kcat_AA3'].value = 10**kcat_AA3
#    cox2_model.parameters['kf_AG_cat1'].value = 10**generic_kf
#    cox2_model.parameters['kr_AG_cat1'].value = 10**(KD_AG_cat1*generic_kf)
#    cox2_model.parameters['kcat_AG1'].value = 10**kcat_AG1
    cox2_model.parameters['kf_AG_cat2'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_cat2'].value = 10**(KD_AG_cat2+generic_kf)
    cox2_model.parameters['kf_AG_cat3'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_cat3'].value = 10**(KD_AG_cat3+generic_kf)
    cox2_model.parameters['kcat_AG3'].value = 10**kcat_AG3
    cox2_model.parameters['kf_AA_allo1'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo1'].value = 10**(KD_AA_allo1+generic_kf)
    cox2_model.parameters['kf_AA_allo2'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo2'].value = 10**(KD_AA_allo2+generic_kf)
    cox2_model.parameters['kf_AA_allo3'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo3'].value = 10**(KD_AA_allo3+generic_kf)
    cox2_model.parameters['kf_AG_allo1'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_allo1'].value = 10**(KD_AG_allo1+generic_kf)
#    cox2_model.parameters['kf_AG_allo2'].value = 10**generic_kf
#    cox2_model.parameters['kr_AG_allo2'].value = 10**(KD_AG_allo2*generic_kf)
    cox2_model.parameters['kf_AG_allo3'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_allo3'].value = 10**(KD_AG_allo3+generic_kf)
    
    PG_array = np.zeros((7,7), dtype='float64')
    PGG_array = np.zeros((7,7), dtype='float64')    
    
    arr_row = 0
    arr_col = 0
    
    #Simulate and fill in arrays
    for AA_init in exp_cond_AA:
        for AG_init in exp_cond_AA:
            cox2_model.parameters['AA_0'].value = AA_init
            cox2_model.parameters['AG_0'].value = AG_init
            solver.run()
            PG_array[arr_row, arr_col] = solver.yobs['obsPG'][-1]
            PGG_array[arr_row, arr_col] = solver.yobs['obsPGG'][-1]
            if arr_col < 6:        
                arr_col += 1
            else:
                arr_col = 0
        arr_row += 1
      
    print 'PG_error: ', np.sum(PG_array-exp_data_PG)
    print 'PGG_error: ', np.sum(PGG_array-exp_data_PGG)
    return PG_array, PGG_array

@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar], otypes=[t.dscalar])
def likelihood_thermobox1(KD_AA_cat3, KD_AA_allo1, KD_AA_allo2):
    box1 = (1/(10**KD_AA_cat1))*(1/(10**KD_AA_allo2))*(10**KD_AA_cat3)*(10**KD_AA_allo1)
    print 'box1: ',box1
    return np.array(box1, dtype='float64')
    
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar], otypes=[t.dscalar])
def likelihood_thermobox2(KD_AA_allo1, KD_AA_allo3, KD_AG_cat3):
    box2 = (1/(10**KD_AA_allo1))*(1/(10**KD_AG_cat3))*(10**KD_AA_allo3)*(10**KD_AG_cat1)
    print 'box2: ',box2
    return np.array(box2, dtype='float64')
    
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar], otypes=[t.dscalar])
def likelihood_thermobox3(KD_AA_cat2, KD_AG_allo1):
    box3 = (1/(10**KD_AG_allo1))*(1/(10**KD_AA_cat2))*(10**KD_AG_allo2)*(10**KD_AA_cat1)
    print 'box3: ',box3
    return np.array(box3, dtype='float64')

@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar], otypes=[t.dscalar])
def likelihood_thermobox4(KD_AG_cat2, KD_AG_allo1, KD_AG_allo3):
    box4 = (1/(10**KD_AG_cat1))*(1/(10**KD_AG_allo3))*(10**KD_AG_cat2)*(10**KD_AG_allo1)
    print 'box4: ',box4
    return np.array(box4, dtype='float64')

#Setting up PyMC model
with model:
    # Add PySB rate parameters as unobserved random variables to PyMC model
    
    #pm.Normal('KD_AA_cat1', mu=np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value), sd=.08, dtype='float64')
    #pm.Normal('kcat_AA1', mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=.08, dtype='float64')
    pm.Normal('KD_AA_cat2', mu=np.log10(cox2_model.parameters['kr_AA_cat2'].value/cox2_model.parameters['kf_AA_cat2'].value), sd=1.5, dtype='float64')
    pm.Normal('kcat_AA2', mu=np.log10(cox2_model.parameters['kcat_AA2'].value), sd=.66, dtype='float64')
    pm.Normal('KD_AA_cat3', mu=np.log10(cox2_model.parameters['kr_AA_cat3'].value/cox2_model.parameters['kf_AA_cat3'].value), sd=1.5, dtype='float64')
    pm.Normal('kcat_AA3', mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=.66, dtype='float64') 
    #pm.Normal('KD_AG_cat1', mu=np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value), sd=.08, dtype='float64')
    #pm.Normal('kcat_AG1', mu=np.log10(cox2_model.parameters['kcat_AG1'].value), sd=.08, dtype='float64')
    pm.Normal('KD_AG_cat2', mu=np.log10(cox2_model.parameters['kr_AG_cat2'].value/cox2_model.parameters['kf_AG_cat2'].value), sd=1.5, dtype='float64')
    pm.Normal('KD_AG_cat3', mu=np.log10(cox2_model.parameters['kr_AG_cat3'].value/cox2_model.parameters['kf_AG_cat3'].value), sd=1.5, dtype='float64')
    pm.Normal('kcat_AG3', mu=np.log10(cox2_model.parameters['kcat_AG3'].value), sd=.66, dtype='float64')
    pm.Normal('KD_AA_allo1', mu=np.log10(cox2_model.parameters['kr_AA_allo1'].value/cox2_model.parameters['kf_AA_allo1'].value), sd=1, dtype='float64')
    pm.Normal('KD_AA_allo2', mu=np.log10(cox2_model.parameters['kr_AA_allo2'].value/cox2_model.parameters['kf_AA_allo2'].value), sd=1, dtype='float64')
    pm.Normal('KD_AA_allo3', mu=np.log10(cox2_model.parameters['kr_AA_allo3'].value/cox2_model.parameters['kf_AA_allo3'].value), sd=1, dtype='float64')
    pm.Normal('KD_AG_allo1', mu=np.log10(cox2_model.parameters['kr_AG_allo1'].value/cox2_model.parameters['kf_AG_allo1'].value), sd=1, dtype='float64')
    #pm.Normal('KD_AG_allo2', mu=np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value), sd=.08, dtype='float64')
    pm.Normal('KD_AG_allo3', mu=np.log10(cox2_model.parameters['kr_AG_allo3'].value/cox2_model.parameters['kf_AG_allo3'].value), sd=1, dtype='float64')
    
    #Set starting location for sampling
    start = {#model.KD_AA_cat1: np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value), model.kcat_AA1: np.log10(cox2_model.parameters['kcat_AA1'].value),\
    model.KD_AA_cat2: np.log10(cox2_model.parameters['kr_AA_cat2'].value/cox2_model.parameters['kf_AA_cat2'].value), model.kcat_AA2: np.log10(cox2_model.parameters['kcat_AA2'].value),\
    model.KD_AA_cat3: np.log10(cox2_model.parameters['kr_AA_cat3'].value/cox2_model.parameters['kf_AA_cat3'].value), model.kcat_AA3: np.log10(cox2_model.parameters['kcat_AA1'].value), \
    #model.KD_AG_cat1: np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value), model.kcat_AG1: np.log10(cox2_model.parameters['kcat_AG1'].value), \
    model.KD_AG_cat2: np.log10(cox2_model.parameters['kr_AG_cat2'].value/cox2_model.parameters['kf_AG_cat2'].value), model.KD_AG_cat3: np.log10(cox2_model.parameters['kr_AG_cat3'].value/cox2_model.parameters['kf_AG_cat3'].value), \
    model.kcat_AG3: np.log10(cox2_model.parameters['kcat_AG3'].value), model.KD_AA_allo1: np.log10(cox2_model.parameters['kr_AA_allo1'].value/cox2_model.parameters['kf_AA_allo1'].value), \
    model.KD_AA_allo2: np.log10(cox2_model.parameters['kr_AA_allo2'].value/cox2_model.parameters['kf_AA_allo2'].value), model.KD_AA_allo3: np.log10(cox2_model.parameters['kr_AA_allo3'].value/cox2_model.parameters['kf_AA_allo3'].value), \
    model.KD_AG_allo1: np.log10(cox2_model.parameters['kr_AG_allo1'].value/cox2_model.parameters['kf_AG_allo1'].value), #model.KD_AG_allo2: np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value), \
    model.KD_AG_allo3: np.log10(cox2_model.parameters['kr_AG_allo3'].value/cox2_model.parameters['kf_AG_allo3'].value)}        
    
    #Model likelihood - compare simulated values of PGs and PGGs at various substrate concentrations
    PG_output, PGG_output = likelihood(\
    #model.KD_AA_cat1, model.kcat_AA1, 
    model.KD_AA_cat2, model.kcat_AA2, model.KD_AA_cat3, model.kcat_AA3, \
    #model.KD_AG_cat1, model.kcat_AG1, 
    model.KD_AG_cat2, model.KD_AG_cat3, model.kcat_AG3, model.KD_AA_allo1, model.KD_AA_allo2, model.KD_AA_allo3, model.KD_AG_allo1, \
    #model.KD_AG_allo2, 
    model.KD_AG_allo3)
    
    pm.Normal('PGs_observed', mu=PG_output, sd=exp_data_sd_PG, observed=exp_data_PG)
    pm.Normal('PGGs_observed', mu=PGG_output, sd=exp_data_sd_PGG, observed=exp_data_PGG)
    
    #Define likelihoods based on energy conservation (thermodynamic boxes):
    box1 = likelihood_thermobox1(model.KD_AA_cat3, model.KD_AA_allo1, model.KD_AA_allo2)
    pm.Normal('thermodynamic_box1', mu=box1, sd=1e-2, observed=1)
    
    box2 = likelihood_thermobox2(model.KD_AA_allo1, model.KD_AA_allo3, model.KD_AG_cat3)
    pm.Normal('thermodynamic_box2', mu=box2, sd=1e-2, observed=1)
    
    box3 = likelihood_thermobox3(model.KD_AA_cat2, model.KD_AG_allo1)
    pm.Normal('thermodynamic_box3', mu=box3, sd=1e-2, observed=1)
    
    box4 = likelihood_thermobox4(model.KD_AG_cat2, model.KD_AG_allo1, model.KD_AG_allo3)
    pm.Normal('thermodynamic_box4', mu=box4, sd=1e-2, observed=1)
    
    #Start from end of last trace
    #start2 = {('KD_AA_cat1',np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value)), ('kcat_AA1', np.log10(cox2_model.parameters['kcat_AA1'].value)), ('KD_AA_cat2',-2.76), ('kcat_AA2',.235), ('KD_AA_cat3',-2.265), ('kcat_AA3',.1760), ('KD_AG_cat1', np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value)), ('kcat_AG1', np.log10(cox2_model.parameters['kcat_AG1'].value)), ('KD_AG_cat2',-.7495), ('KD_AG_cat3',-2.793), ('kcat_AG3',-.5095), ('KD_AA_allo1',2.290), ('KD_AA_allo2',.1001), ('KD_AA_allo3',-.3808), ('KD_AG_allo1',2.280), ('KD_AG_allo2', np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value)), ('KD_AG_allo3',1.650)}   
    
    #Select MCMC stepping method
    step = pm.Dream(nseedchains=120, save_history=True, blocked=True)
    
    
    trace = pm.sample(500000, step, njobs=5)
    
    dictionary_to_pickle = {}
    print len(trace)
    for dictionary in trace:
        for var in dictionary:
           dictionary_to_pickle[var] = trace[var] 
    
    pickle.dump(dictionary_to_pickle, open('2014_12_20_Dream_COX2_parameter_fits.p', 'wb'))
    