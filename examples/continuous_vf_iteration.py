# Based on http://johnstachurski.net/lectures/fvi_rpd.html#an-optimal-growth-model
# This works starting from macroModels.
# Need to figure out relative imports.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

from tools.StepFun import StepFun
from ngm_continuous import ngm_continuous

theta, alpha, beta = .5, .8, .9

u_params = {'U': lambda c: 1 - np.exp(-theta * c),
        'theta': theta, 'alpha': alpha, 'beta': beta}


class Utility:
    """Use U for functional form.
    """
    def __init__(self, u_params):
        self.U = u_params['U']
        self.theta = u_params['theta']
        self.alpha = u_params['alpha']
        self.beta = u_params['beta']

utility = Utility(u_params)

f = lambda k:  k ** alpha
L = lognorm(1)
G = L.cdf

gridmax, gridsize = 8, 150
grid = np.linspace(0, gridmax ** 1e-1, gridsize) ** 10

v = StepFun(grid, utility.U(grid))
tol = 0.005
params = {'utility': utility, 'distribution': L, 'cdf': G,
    'gridsize': gridsize, 'grid': grid, 'tol': tol, 'f': f}

c = ngm_continuous(v, params)
while 1:
    plt.plot(grid, v.Y, 'k--')
    new_v = c.bellman(v)
    if max(abs(new_v.Y - v.Y)) < tol:
        break
    v = new_v

plt.show()