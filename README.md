COX-2 Reaction Model (CORM)
=============

CORM provides a proposed model of cyclooxygenase-2 (COX-2) allosteric and catalytic interactions with arachidonic acid (AA) and 2-arachidonoylglycerol (2-AG).
COX-2 turnover of AA or 2-AG produces prostaglandins (PGs) or prostaglandin glyceryl esters (PG-Gs), respectively.

Quickstart
----------

To facilitate easy use of CORM, we provide a virtual machine with everything you need to run CORM already installed.  You can download it [here](puma.mc.vanderbilt.edu:8000/virtual_machines/CORM_demo.ova) (Warning: 1.4G file).

Installation
------------

CORM is encoded as a Python Systems Biology (PySB) model.  To download and install PySB and its dependencies, [see the PySB documentation](http://docs.pysb.org/en/latest/installation.html#option-2-installing-the-dependencies-yourself).

If you would like to run Bayesian Markov Chain Monte Carlo (MCMC) fitting of the CORM parameters, you will also need the LoLab-VU fork of PyMC 3.

Clone the repository:
```
git clone https://github.com/LoLab-VU/pymc
```

Enter the cloned directory and switch to the branch that has the DREAM MCMC stepping method:
```
git checkout dream
```

Then install PyMC:
```
python setup.py install
```

Running CORM
------------

The CORM model file is defined in the file [corm.py](https://github.com/LoLab-VU/CORM/blob/master/corm.py).

You can import the model for exploration by typing:
```python
from corm import model
```
from an iPython session.

We also provide an iPython notebook that will walk you through how CORM is defined [here](http://nbviewer.ipython.org/urls/raw.githubusercontent.com/LoLab-VU/CORM/master/corm_tutorial.ipynb).

Fitting CORM Parameters
-----------------------

The file [run_pymc_sampling.py](https://github.com/LoLab-VU/CORM/blob/master/run_pymc_sampling.py) provides a Python script for fitting CORM parameters using PyMC.

To run the fitting, simply type:
```
python run_pymc_sampling.py
```

Parameter traces for all MCMC chains will be saved to a pickled Python dictionary as specified in run_pymc_sampling.py. 
