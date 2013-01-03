
"""
http://www.compmacro.com/makoto/note/note_ngm_disc.pdf

Exogenous Parameters

* k_n : number of grid points
* k_l : lower bound on capital stock
* k_u : upper boind on capital stock
* epsilon : upper bound on error
* v_0 : initial guess.
"""

from __future__ import division

import numpy as np


def ngm(alpha=.36, delta=.08, v_0=.01, k_n=100, k_l=.01, k_u=10,
    epsilon=.00005, z=1, u=np.log, f=None):
    """
    """
    k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n)
    k_grid = np.tile(k_v, (k_n, k_n))
    e = 1

    f = lambda k: k ** alpha

    c = z * f(k_grid) + (1 - delta) * k - k_grid
    utility = u(c)

    while e > epsilon:
