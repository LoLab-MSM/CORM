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

from basic_COX2_model import model as cox2_model

model = pm.Model()

#Initialize PySB solver object for simulations
tspan = np.linspace(0,10, num=100)
solver = Solver(cox2_model, tspan)

#Add import of experimental data here


#Experimental starting values of AA and 2-AG (all in microM)
exp_cond_AA = [0, .5, 1, 2, 4, 8, 16]
exp_cond_AG = [0, .5, 1, 2, 4, 8, 16]

#Likelihood function to generate simulated data that corresponds to experimental time points
@theano.compile.ops.as_op(itypes=[t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar, t.dscalar] \
,otypes=[t.dscalar, t.dscalar])
def likelihood(KD_AA_cat1, kcat_AA1, KD_AA_cat2, kcat_AA2, KD_AA_cat3, kcat_AA3, KD_AG_cat1, \
    kcat_AG1, KD_AG_cat2, KD_AG_cat3, kcat_AG3, KD_AA_allo1, KD_AA_allo2, KD_AA_allo3, KD_AG_allo1, KD_AG_allo2, KD_AG_allo3):
    
    #generic kf in units of inverse microM*s (matches model units)
    generic_kf = np.log10(1.5e4)
    
    #Sub in parameter values at current location in parameter space
    cox2_model.parameters['kf_AA_cat1'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_cat1'].value = 10**(KD_AA_cat1*generic_kf)
    cox2_model.parameters['kcat_AA1'].value = 10**kcat_AA1
    cox2_model.parameters['kf_AA_cat2'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_cat2'].value = 10**(KD_AA_cat2*generic_kf)
    cox2_model.parameters['kcat_AA2'].value = 10**kcat_AA2
    cox2_model.parameters['kf_AA_cat3'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_cat3'].value = 10**(KD_AA_cat3*generic_kf)
    cox2_model.parameters['kcat_AA3'].value = 10**kcat_AA3
    cox2_model.parameters['kf_AG_cat1'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_cat1'].value = 10**(KD_AG_cat1*generic_kf)
    cox2_model.parameters['kcat_AG1'].value = 10**kcat_AG1
    cox2_model.parameters['kf_AG_cat2'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_cat2'].value = 10**(KD_AG_cat2*generic_kf)
    cox2_model.parameters['kf_AG_cat3'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_cat3'].value = 10**(KD_AG_cat3*generic_kf)
    cox2_model.parameters['kcat_AG3'].value = 10**kcat_AG3
    cox2_model.parameters['kf_AA_allo1'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo1'].value = 10**(KD_AA_allo1*generic_kf)
    cox2_model.parameters['kf_AA_allo2'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo2'].value = 10**(KD_AA_allo2*generic_kf)
    cox2_model.parameters['kf_AA_allo3'].value = 10**generic_kf
    cox2_model.parameters['kr_AA_allo3'].value = 10**(KD_AA_allo3*generic_kf)
    cox2_model.parameters['kf_AG_allo1'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_allo1'].value = 10**(KD_AG_allo1*generic_kf)
    cox2_model.parameters['kf_AG_allo2'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_allo2'].value = 10**(KD_AG_allo2*generic_kf)
    cox2_model.parameters['kf_AG_allo3'].value = 10**generic_kf
    cox2_model.parameters['kr_AG_allo3'].value = 10**(KD_AG_allo3*generic_kf)
    
    PG_array = np.zeros((7,7))
    PGG_array = np.zeros((7,7))    
    
    arr_row = 0
    arr_col = 0
    
    #Simulate and fill in arrays
    for AA_init in exp_cond_AA:
        for AG_init in exp_cond_AA:
            cox2_model.parameters['AA_0'].value = 10**AA_init
            cox2_model.parameters['AG_0'].value = 10**AG_init
            solver.run()
            PG_array[arr_row, arr_col] = solver.yobs['obsPG'][-1]
            PGG_array[arr_row, arr_col] = solver.yobs['obsPGG'][-1]
            if arr_col < 6:        
                arr_col += 1
            else:
                arr_col = 0
        arr_row += 1
        
    return PG_array, PGG_array

#Setting up PyMC model
with model:
    # Add PySB rate parameters as unobserved random variables to PyMC model
    
    pm.Normal('KD_AA_cat1', mu=np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value), sd=2, dtype='float64')
    pm.Normal('kcat_AA1', mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=2, dtype='float64')
    pm.Normal('KD_AA_cat2', mu=np.log10(cox2_model.parameters['kr_AA_cat2'].value/cox2_model.parameters['kf_AA_cat2'].value), sd=2, dtype='float64')
    pm.Normal('kcat_AA2', mu=np.log10(cox2_model.parameters['kcat_AA2'].value), sd=2, dtype='float64')
    pm.Normal('KD_AA_cat3', mu=np.log10(cox2_model.parameters['kr_AA_cat3'].value/cox2_model.parameters['kf_AA_cat3'].value), sd=2, dtype='float64')
    pm.Normal('kcat_AA3', mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=2, dtype='float64') 
    pm.Normal('KD_AG_cat1', mu=np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value), sd=2, dtype='float64')
    pm.Normal('kcat_AG1', mu=np.log10(cox2_model.parameters['kcat_AG1'].value), sd=2, dtype='float64')
    pm.Normal('KD_AG_cat2', mu=np.log10(cox2_model.parameters['kr_AG_cat2'].value/cox2_model.parameters['kf_AG_cat2'].value), sd=2, dtype='float64')
    pm.Normal('KD_AG_cat3', mu=np.log10(cox2_model.parameters['kr_AG_cat3'].value/cox2_model.parameters['kf_AG_cat3'].value), sd=2, dtype='float64')
    pm.Normal('kcat_AG3', mu=np.log10(cox2_model.parameters['kcat_AG3'].value), sd=2, dtype='float64')
    pm.Normal('KD_AA_allo1', mu=np.log10(cox2_model.parameters['kr_AA_allo1'].value/cox2_model.parameters['kf_AA_allo1'].value), sd=2, dtype='float64')
    pm.Normal('KD_AA_allo2', mu=np.log10(cox2_model.parameters['kr_AA_allo2'].value/cox2_model.parameters['kf_AA_allo2'].value), sd=2, dtype='float64')
    pm.Normal('KD_AA_allo3', mu=np.log10(cox2_model.parameters['kr_AA_allo3'].value/cox2_model.parameters['kf_AA_allo3'].value), sd=2, dtype='float64')
    pm.Normal('KD_AG_allo1', mu=np.log10(cox2_model.parameters['kr_AG_allo1'].value/cox2_model.parameters['kf_AG_allo1'].value), sd=2, dtype='float64')
    pm.Normal('KD_AG_allo2', mu=np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value), sd=2, dtype='float64')
    pm.Normal('KD_AG_allo3', mu=np.log10(cox2_model.parameters['kr_AG_allo3'].value/cox2_model.parameters['kf_AG_allo3'].value), sd=2, dtype='float64')
    
    #Model likelihood - compare simulated values of PGs and PGGs at various substrate concentrations
    PG_output, PGG_output = likelihood(model.KD_AA_cat1, model.kcat_AA1, model.KD_AA_cat2, model.kcat_AA2, model.KD_AA_cat3, model.kcat_AA3, model.KD_AG_cat1, \
    model.kcat_AG1, model.KD_AG_cat2, model.KD_AG_cat3, model.kcat_AG3, model.KD_AA_allo1, model.KD_AA_allo2, model.KD_AA_allo3, model.KD_AG_allo1, model.KD_AG_allo2, model.KD_AG_allo3)
    
    #pm.Normal('PGs_observed', mu=PG_output, sd=1, observed=exp_data_PG)
    #pm.Normal('PGGs_observed', mu=PGG_output, sd=1, observed=exp_data_PGG)
    
    #Select MCMC stepping method
    step = pm.Metropolis()
    
    trace = pm.sample(1e6, steps, njobs=None)
    