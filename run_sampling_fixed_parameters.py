# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:58:34 2016

@author: Erin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:26:46 2014
@author: Erin
"""

from core import run_dream
from pysb.integrate import Solver
import numpy as np
from parameters import NormalParam
from scipy.stats import norm
import pysb

from corm import model as cox2_model

#pysb.integrate.weave_inline = None

#Initialize PySB solver object for simulations
tspan = np.linspace(0,10, num=100)
solver = Solver(cox2_model, tspan)

#Add import of experimental data here
#location = '/Users/Erin/git/COX2/exp_data/'
location= 'home/shockle/git/COX2_kinetics/exp_data'
exp_data_PG = np.loadtxt(location+'exp_data_pg.txt')
exp_data_PGG = np.loadtxt(location+'exp_data_pgg.txt')

exp_data_sd_PG = np.loadtxt(location+'exp_data_sd_pg.txt')
exp_data_sd_PGG = np.loadtxt(location+'exp_data_sd_pgg.txt')

#Experimental starting values of AA and 2-AG (all in microM)
exp_cond_AA = [0, .5, 1, 2, 4, 8, 16]
exp_cond_AG = [0, .5, 1, 2, 4, 8, 16]

#Experimentally measured parameter values
KD_AA_cat1 = np.log10(cox2_model.parameters['kr_AA_cat1'].value/cox2_model.parameters['kf_AA_cat1'].value)
kcat_AA1 = np.log10(cox2_model.parameters['kcat_AA1'].value)
KD_AG_cat1 = np.log10(cox2_model.parameters['kr_AG_cat1'].value/cox2_model.parameters['kf_AG_cat1'].value)
kcat_AG1 = np.log10(cox2_model.parameters['kcat_AG1'].value)
KD_AG_allo3 = np.log10(cox2_model.parameters['kr_AG_allo3'].value/cox2_model.parameters['kf_AG_allo3'].value)

kf_idxs = [i for i, param in enumerate(cox2_model.parameters) if 'kf' in param.name]

#generic kf in units of inverse microM*s (matches model units)
generic_kf = np.log10(1.5e4)

#Frozen probability distributions for likelihoods
like_PGs = norm(loc=exp_data_PG, scale=exp_data_sd_PG)
like_PGGs = norm(loc=exp_data_PGG, scale=exp_data_sd_PGG)
like_thermobox = norm(loc=1, scale=1e-2)

pysb_sampled_parameter_names = ['kr_AA_cat2', 'kcat_AA2', 'kr_AA_cat3', 'kcat_AA3', 'kr_AG_cat2', 'kr_AG_cat3', 'kcat_AG3', 'kr_AA_allo1', 'kr_AA_allo2', 'kr_AA_allo3', 'kr_AG_allo1', 'kr_AG_allo2']
kfs_to_change = ['kf_AA_cat2', 'kf_AA_cat3', 'kf_AG_cat2', 'kf_AG_cat3', 'kf_AA_allo1', 'kf_AA_allo2', 'kf_AA_allo3', 'kf_AG_allo1', 'kf_AG_allo2']
kf_idxs = [i for i, param in enumerate(cox2_model.parameters) if param.name in kfs_to_change]
print 'kf idxs: ',kf_idxs

old_results = np.load('/Users/Erin/git/COX2/results/2015_02_02_COX2_all_traces.npy')
results_ordering = ['kcat_AA2', 'kcat_AA3', 'kr_AG_cat3', 'kr_AG_cat2', 'kr_AG_allo2', 'kr_AG_allo1', 'kr_AA_allo1', 'kr_AA_allo2', 'kr_AA_allo3', 'kcat_AG3', 'kr_AA_cat3', 'kr_AA_cat2']

new_ordering_dict = {name: idx for idx, name in enumerate(pysb_sampled_parameter_names)}
old_ordering_dict = {name: idx for idx, name in enumerate(results_ordering)}

#Likelihood function to generate simulated data that corresponds to experimental time points
def likelihood(parameter_vector):    
    #print 'model parameters before subbing: ',cox2_model.parameters
    param_dict = {pname: pvalue for pname, pvalue in zip(pysb_sampled_parameter_names, parameter_vector)}
    #print 'param dict: ',param_dict    
    for pname, pvalue in param_dict.items():   
        
        #Sub in parameter values at current location in parameter space
        
        if 'kr' in pname:
            cox2_model.parameters[pname].value = 10**(pvalue + generic_kf)
        
        elif 'kcat' in pname:
            cox2_model.parameters[pname].value = 10**pvalue
    
    #print 'model parameters after subbing: ',cox2_model.parameters
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
    
    #mBid_pdf_pt = (sim_mBid - exp_data['norm_ICRP'])/mBid_sd
#    np.log((np.exp(-(mBid_pdf_pt**2)/2)/(np.sqrt(2*np.pi)*mBid_sd)))    
    #pg_pdf_pt = (PG_array - exp_data_PG)/exp_data_sd_PG
    #logp_PG_2 = np.sum(np.log((np.exp(-(pg_pdf_pt**2)/2)/np.sqrt(2*np.pi)*exp_data_sd_PG)))
    #pgg_pdf_pt = (PGG_array - exp_data_PGG)/exp_data_sd_PGG
    #logp_PGG_2 = np.sum(np.log((np.exp(-(pgg_pdf_pt**2)/2)/np.sqrt(2*np.pi)*exp_data_sd_PGG)))
    
    logp_PG = np.sum(like_PGs.logpdf(PG_array))
    logp_PGG = np.sum(like_PGGs.logpdf(PGG_array))
    
    box1 = (1/(10**KD_AA_cat1))*(1/(10**param_dict['kr_AA_allo2']))*(10**param_dict['kr_AA_cat3'])*(10**param_dict['kr_AA_allo1'])
    box2 = (1/(10**param_dict['kr_AA_allo1']))*(1/(10**param_dict['kr_AG_cat3']))*(10**param_dict['kr_AA_allo3'])*(10**KD_AG_cat1)
    box3 = (1/(10**param_dict['kr_AG_allo1']))*(1/(10**param_dict['kr_AA_cat2']))*(10**param_dict['kr_AG_allo2'])*(10**KD_AA_cat1)
    box4 = (1/(10**KD_AG_cat1))*(1/(10**KD_AG_allo3))*(10**param_dict['kr_AG_cat2'])*(10**param_dict['kr_AG_allo1'])

    #box_pdf_pt = (box1 - 1)/1e-2
    #logp_box1_2 = np.sum(np.log((np.exp(-(box_pdf_pt**2)/2)/np.sqrt(2*np.pi)*1e-2)))    
    
    logp_box1 = like_thermobox.logpdf(box1)
    logp_box2 = like_thermobox.logpdf(box2)
    logp_box3 = like_thermobox.logpdf(box3)
    logp_box4 = like_thermobox.logpdf(box4)
    
    #print 'logps: ',logp_PG,logp_PGG,logp_box1, logp_box2, logp_box3, logp_box4 
    #print 'logps 2: ',logp_PG_2, logp_PGG_2, logp_box1_2
    
    total_logp = logp_PG + logp_PGG + logp_box1 + logp_box2 + logp_box3 + logp_box4
    if np.isnan(total_logp):
        total_logp = -np.inf
      
    return total_logp


# Add PySB rate parameters as unobserved random variables
   
kd_AA_cat2 = NormalParam('KD_AA_cat2', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_cat2'].value/cox2_model.parameters['kf_AA_cat2'].value), sd=1.5)
kcat_AA2 = NormalParam('kcat_AA2', value = 1, mu=np.log10(cox2_model.parameters['kcat_AA2'].value), sd=.66)
kd_AA_cat3 = NormalParam('KD_AA_cat3', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_cat3'].value/cox2_model.parameters['kf_AA_cat3'].value), sd=1.5)
kcat_AA3 = NormalParam('kcat_AA3', value = 1, mu=np.log10(cox2_model.parameters['kcat_AA1'].value), sd=.66) 
kd_AG_cat2 = NormalParam('KD_AG_cat2', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_cat2'].value/cox2_model.parameters['kf_AG_cat2'].value), sd=1.5)
kd_AG_cat3 = NormalParam('KD_AG_cat3', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_cat3'].value/cox2_model.parameters['kf_AG_cat3'].value), sd=1.5)
kcat_AG3 = NormalParam('kcat_AG3', value = 1, mu=np.log10(cox2_model.parameters['kcat_AG3'].value), sd=.66)
kd_AA_allo1 = NormalParam('KD_AA_allo1', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo1'].value/cox2_model.parameters['kf_AA_allo1'].value), sd=1)
kd_AA_allo2 = NormalParam('KD_AA_allo2', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo2'].value/cox2_model.parameters['kf_AA_allo2'].value), sd=1)
kd_AA_allo3 = NormalParam('KD_AA_allo3', value = 1, mu=np.log10(cox2_model.parameters['kr_AA_allo3'].value/cox2_model.parameters['kf_AA_allo3'].value), sd=1)
kd_AG_allo1 = NormalParam('KD_AG_allo1', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_allo1'].value/cox2_model.parameters['kf_AG_allo1'].value), sd=1)
kd_AG_allo2 = NormalParam('KD_AG_allo2', value = 1, mu=np.log10(cox2_model.parameters['kr_AG_allo2'].value/cox2_model.parameters['kf_AG_allo2'].value), sd=1)       

sampled_parameter_names = [kd_AA_cat2, kcat_AA2, kd_AA_cat3, kcat_AA3, kd_AG_cat2, kd_AG_cat3, kcat_AG3, kd_AA_allo1, kd_AA_allo2, kd_AA_allo3, kd_AG_allo1, kd_AG_allo2]

for param in sampled_parameter_names:
    print 'param.mu: ',param.mu,' and standard deviation: ',param.sd

nchains = 5

#starts = np.zeros((5, 12))
#for chain in range(len(old_results[0:5])):
#    for param_name in pysb_sampled_parameter_names:
#        new_dim = new_ordering_dict[param_name]
#        old_dim = old_ordering_dict[param_name]
#        starts[chain][new_dim] = old_results[chain][old_dim]

#print 'starts: ',starts

#start_val = [param.mu for param in sampled_parameter_names]
#starts = start_val + (np.random.random((nchains, len(start_val)))*.01)
#starts = [np.array([param.mu for param in sampled_parameter_names]) for i in range(5)]
for idx in kf_idxs:
    cox2_model.parameters[idx].value = 10**generic_kf

sampled_params, log_ps = run_dream(sampled_parameter_names, likelihood, niterations=20000, nchains=nchains, multitry=False, gamma_levels=4, adapt_gamma=True, history_thin=1, model_name='corm_dreamzs_5chain_redo', verbose=True)
    
for chain in range(len(sampled_params)):
    np.save('corm_dreamzs_5chain_redo_sampled_params_chain_'+str(chain), sampled_params[chain])
    np.save('corm_dreamzs_5chain_redo_logps_chain_'+str(chain), log_ps[chain])