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

# Capital Stock from BEA
# No gaurantees if this url is stable. It's the fixed assets table 1.2
url = 'http://www.bea.gov/iTable/download.cfm?ext=csv&fid=22884AE5861034A3DC4252D2C45EE9D76FA83A1B12454C208628629C8E66190A96C83B453D0230C6869F8D146433018118D60448DD5D29EFD8C540384FF5736C'
# Saved as capital_stock.csv for now


def silly_bea(f):
    """Parse the messed-up BEA formating.  l is a list of lines.
    """
    lines = f.readlines()
    # metadata = l[:4]
    ret = []
    for line in lines[4:]:
        ret.append(line.replace('\xa0', ''))
    s = []
    for line in ret:
        s.append(line.split(','))
    df = pd.DataFrame(s)
    df2 = df.ix[0:13].T
    df2 = df2[1:]
    idx = df2[0]
    idx = [x.strip('"').rstrip('"\r\n') for x in idx]
    idx[0] = 0
    idx = [int(x) for x in idx]

    df2.index = idx
    del df2[0]
    df2 = df2.applymap(lambda x: x.strip('"\r\n'))

    df2 = df2[1]
    df2.name = df2[0]
    df2 = df2[1:]
    return df2

with open('capital_stock.csv') as f:
    df_k = silly_bea(f)

# For some reason 1947 (first one) isn't included.
df_k.index = pd.date_range(str(df_k.index[0] - 1), periods=len(df_k),
    freq='AS-JAN')
df_k = df_k.asfreq('QS-JAN', method='ffill')

df['k'] = df_k
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

# Construct capital stock series.
