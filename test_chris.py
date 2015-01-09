# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 12:09:39 2014

@author: Erin
"""

from pymc import *
import theano.tensor as t
from numpy import arange, array, ones, concatenate, empty
from numpy.random import randint

__all__ = ['disasters_data', 'switchpoint', 'early_mean', 'late_mean', 'rate',
            'disasters']

# Time series of recorded coal mining disasters in the UK from 1851 to 1962
disasters_data = array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                            3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                            2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                            1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                            0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                            3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = len(disasters_data)

#here is the trick
@theano.compile.ops.as_op(itypes=[t.lscalar, t.dscalar, t.dscalar],otypes=[t.dvector])
def rate(switchpoint,early_mean, late_mean):
    ''' Concatenate Poisson means '''
    out = empty(years)
    out[:switchpoint] = early_mean
    out[switchpoint:] = late_mean
    return out


with Model() as model:

    # Prior for distribution of switchpoint location
    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=years)
    # Priors for pre- and post-switch mean number of disasters
    early_mean = Exponential('early_mean', lam=1.)
    late_mean = Exponential('late_mean', lam=1.)

    # Allocate appropriate Poisson rates to years before and after current
    # switchpoint location
    idx = arange(years)
    #theano style:
    #rate = switch(switchpoint >= idx, early_mean, late_mean)
    #non-theano style
    rate = rate(switchpoint, early_mean, late_mean)

    # Data likelihood
    disasters = Poisson('disasters', rate, observed=disasters_data)

    # Initial values for stochastic nodes
    start = {'early_mean': 2., 'late_mean': 3.}

    # Use slice sampler for means
    step1 = Slice([early_mean, late_mean])
    # Use Metropolis for switchpoint, since it accomodates discrete variables
    step2 = Metropolis([switchpoint])

    tr = sample(1000, tune=500, start=start, step=[step1, step2])
    #traceplot(tr)