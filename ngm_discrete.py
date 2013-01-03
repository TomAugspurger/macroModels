
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
import matplotlib.pyplot as plt

def ngm(alpha=.36, beta=.96, delta=.08, v_0=.01, k_n=100, k_l=.01,
    k_u=30, epsilon=.00005, z=1, u=np.log, f=None, max_iter=1000):
    """
    """
    k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n)
    k_grid = np.tile(k_v, (k_n, 1)).T

    f = lambda k: k ** alpha

    c = z * f(k_grid) + (1 - delta) * k_grid - k_grid.T
    utility = u(c)
    utility[c <= 0] = -100000

    e = 1
    iteration = 0
    value_function = np.zeros(k_n)
    new_value_function = np.zeros(k_n)
    policy_rule = np.zeros(k_n)

    while e > epsilon and iteration < max_iter:
        for i in range(k_n):
            temp = utility[i, :] + beta * value_function.T
            ind = np.argmax(temp)
            temp_vf = np.max(temp)
            policy_rule[i] = k_v[ind]
            new_value_function[i] = temp_vf

            e = np.max(np.abs(value_function - new_value_function))
            iteration += 1
            value_function = new_value_function

    return (value_function, policy_rule)

vf, pr = ngm()
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(vf)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(vf)
