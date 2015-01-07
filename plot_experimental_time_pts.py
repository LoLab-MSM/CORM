# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:15:16 2014

@author: Erin
"""

from basic_COX2_model import model
from pysb.integrate import Solver
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,10, num=100)
solver = Solver(model, t)

#Setting up two figures, one for PGs and one for PGGs, each with six subplots
PG_fig, ((pg_ax1, pg_ax4), (pg_ax2, pg_ax5), (pg_ax3, pg_ax6)) = plt.subplots(3, 2)
PGG_fig, ((pgg_ax1, pgg_ax4), (pgg_ax2, pgg_ax5), (pgg_ax3, pgg_ax6)) = plt.subplots(3,2)

#Creating numpy arrays to hold simulated experimental datapoints
PG_array = np.zeros((7,7))
PGG_array = np.zeros((7,7))

#Experimental starting values of AA and 2-AG (all in microM)
exp_cond_AA = [0, .5, 1, 2, 4, 8, 16]
exp_cond_AG = [0, .5, 1, 2, 4, 8, 16]

#Load experimental data
exp_data_PG = np.loadtxt('exp_data_pg.txt')
exp_data_PGG = np.loadtxt('exp_data_pgg.txt')
exp_data_sd_PG = np.loadtxt('exp_data_sd_pg.txt')
exp_data_sd_PGG = np.loadtxt('exp_data_sd_pgg.txt')

arr_row = 0
arr_col = 0
#Simulate and fill in arrays
for AA_init in exp_cond_AA:
    for AG_init in exp_cond_AA:
        print 'Simulating AA = '+str(AA_init)+' AG = '+str(AG_init)
        model.parameters['AA_0'].value = AA_init
        model.parameters['AG_0'].value = AG_init
        solver.run()
        PG_array[arr_row, arr_col] = solver.yobs['obsPG'][-1]
        PGG_array[arr_row, arr_col] = solver.yobs['obsPGG'][-1]
        print 'Filled in row '+str(arr_row)+' and column '+str(arr_col)+' of arrays'
        if arr_col < 6:        
            arr_col += 1
        else:
            arr_col = 0
        
    arr_row += 1

# Plot output (PG and PGGs)
PG_row_start = 1
for axis, condition in zip([pg_ax1, pg_ax2, pg_ax3, pg_ax4, pg_ax5, pg_ax6], exp_cond_AA[1::]):
    axis.plot(exp_cond_AG, PG_array[PG_row_start,:], label='sim')
    axis.errorbar(exp_cond_AG, exp_data_PG[PG_row_start,:], yerr=exp_data_sd_PG[PG_row_start, :], label='exp')
    axis.set_title(str(condition)+' microM AA')
    axis.set_xlabel('[2-AG] (microM)')
    axis.set_ylabel('PGs (microM)')
    axis.set_ylim(0, .24)
    axis.set_xlim(0, 16.5)
    axis.legend(loc=1, fontsize='xx-small')
    PG_row_start += 1
PG_fig.subplots_adjust(wspace=.5, hspace=1)
PG_fig.show()

PGG_col_start = 1
for axis, condition in zip([pgg_ax1, pgg_ax2, pgg_ax3, pgg_ax4, pgg_ax5, pgg_ax6], exp_cond_AG[1::]):
    axis.plot(exp_cond_AA, PGG_array[:,PGG_col_start], label='sim', )
    axis.errorbar(exp_cond_AG, exp_data_PGG[:, PGG_col_start], yerr=exp_data_sd_PGG[:, PGG_col_start], label='exp')
    axis.set_title(str(condition)+' microM AG')
    axis.set_xlabel('[AA] (microM)')
    axis.set_ylabel('PGGs (microM)')
    axis.set_ylim(0, .18)
    axis.set_xlim(0, 16.5)
    axis.legend(loc=1, fontsize='xx-small')
    PGG_col_start += 1
PGG_fig.subplots_adjust(wspace=.5, hspace=1)
PGG_fig.show()


