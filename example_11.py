"""
Based on Martin's HW 11.
"""

from __future__ import division

import numpy as np
import pandas as pd
from pandas.io import data
from ngm_discrete import NGM

# Collect the data with pandas.
"""
Real Gross Domestic Product, 1 Decimal (GDPC1)
Real Personal Consumption Expenditures (PCECC96)
Real Gross Private Domestic Investment, 3 Decimal (GPDIC96)
Federal Government: Current Expenditures (FGEXPND)
Net Exports of Goods & Services (NETEXP)
Nonfarm Business output/hour: OPHNFB
Agg hours worked index: AWHI
Civilian Noninstitutional Population (CNP16OV)
"""
want = ['GDPC1', 'PCECC96', 'GPDIC96', 'FGEXPND', 'NETEXP', 'OPHNFB', 'AWHI',
    'CNP16OV']
names = {'GDPC1': 'gdp',
        'PCECC96': 'c',
        'GPDIC96': 'x',
        'FGEXPND': 'g',
        'NETEXP': 'nx',
        'OPHNFB': 'bls_prod',
        'AWHI': 'hours',
        'CNP16OV': 'popn'}

start_date = '1947-01-01'
for s in want:
    if want.index(s) == 0:
        df = data.DataReader(s, data_source='fred', start=start_date)
        df = df.rename(columns={s: names[s]})
    else:
        df[names[s]] = data.DataReader(s, data_source='fred', start=start_date)

df['my_prod'] = df['gdp'] / df['hours']

# Parameters

def u(c, h, theta):
    return (1 - theta) * np.log(c) + theta * np.log(1 - h)


def f(k, h, alpha, gamma, t):
    """Labor augmenting technology.
    gamma > 0 is rate of growth of technology.
    """
    return k ** alpha * ((1 + gamma) ** t * h) ** (1 - alpha)

# Shocks:
s = lambda z: np.exp(z)


def lom_shocks(z, rho, var_e):
    """AR1 process to generate next period's shock.
    """
    e_prime = np.random.randn(0, var_e)
    return rho * z + e_prime
