"""
A collection of techniques to model standard neoclassical growth models.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


class NGM(object):
    """Calculate a neoclassical growth model using value function iteration.
    http://www.compmacro.com/makoto/note/note_ngm_disc.pdf

    Exogenous Parameters

    * k_n : number of grid points
    * k_l : lower bound on capital stock
    * k_u : upper boind on capital stock
    * v_0 : initial guess.
    * detla: depreciation rate of capital
    * u: a utility function
    * f: a production function
    * z: currently a placeholder for some stochastic shock matrix.
    * epsilon: tolerance of error
    * mat_iter: Non-economic.  In case something is diverging.

    """
    def __init__(self, alpha=.36, beta=.96, delta=.08, v_0=.01, k_n=100,
        k_l=.01, k_u=30, epsilon=.00005, z=1, u=np.log, f=None,
        max_iter=1000):
        self.params = {'alpha': alpha,
                    'beta': beta,
                    'delta': delta,
                    'v_0': v_0,
                    'k_n': k_n,
                    'k_l': k_l,
                    'k_u': k_u,
                    'epsilon': epsilon,
                    'z': z,
                    'u': u,
                    'f': f,
                    'max_iter': max_iter
                    }

    def ngm(self, **kwargs):
        """
        Call like vf, pr = NGM.ngm()

        TODO: Takes args from self.params as a dict
        TODO: check on v_0; right now assuming all 0.
        """
        k_l = self.params['k_l']
        k_u = self.params['k_u']
        k_n = self.params['k_n']
        alpha = self.params['alpha']
        z = self.params['z']
        beta = self.params['beta']
        delta = self.params['delta']
        epsilon = self.params['epsilon']
        u = self.params['u']
        max_iter = self.params['max_iter']

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

    def gen_plots(self, value_function, policy_rule):
        """Get a plot of the value function & policy rules.
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(value_function)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(policy_rule)
        return fig


if __name__ == "main":
    pass
