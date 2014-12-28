# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 23:15:19 2014

@author: Erin
"""

import numpy as np
import matplotlib.pyplot as plt

fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

fig2, ((ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(3, 2)

trace = np.load('2014_12_20_Dream_COX2_parameter_fits.p')

for axis, variable in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12], trace):
    axis.hist(trace[variable][0][40000::], bins=200)
    axis.set_title(str(variable))
    axis.set_xlabel('Log(parameter value)')
    axis.set_ylabel('Frequency')
fig1.subplots_adjust(wspace=.5, hspace=1)
fig2.subplots_adjust(wspace=.5, hspace=1)
fig1.show()
fig2.show()