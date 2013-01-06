"""
A collection of techniques to model standard neoclassical growth models via
discretization.

Currently supports:

    -Value Function iteration
    -Howard's Improvement Algorithm (need to test).
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
    def __init__(self, alpha=.36, beta=.96, delta=.08, v_0=.01, k_n=1000,
        k_l=.05, k_u=30, epsilon=.00005, z=1, u=np.log, f=None,
        max_iter=1000, n_h=1):
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
                    'max_iter': max_iter,
                    'v_0': v_0,
                    'n_h': n_h}

    def ngm(self, **kwargs):
        """
        Call like vf, pr = NGM.ngm()

        If alt, calculation of c & u is done in loop. Else it is done before.
        Getting different results since the updateding of value_function occurs
        at different times. Alt fixes value_function and loops over each k.
        Non-alt updates value_function after each iteration.  Lean toward alt?

        Non-alt running a bit under 3x slower.
        TODO: Takes args from self.params as a dict
        TODO: Improve v_0 handling.  Right now just allows for singl value.
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
        v_0 = self.params['v_0']
        n_h = self.params['n_h']

        k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n)

        f = lambda k: k ** alpha

        e = 1
        rep = 1
        iteration = 0
        value_function = np.ones(k_n) * v_0
        new_value_function = np.zeros(k_n)
        policy_rule = np.zeros(k_n)

        while e > epsilon and iteration < max_iter:
            # No need for k_grid. k_v will broadcast.
            for i, v in enumerate(k_v):
                if rep == n_h or iteration == 0:
                    c = z * f(v) + (1 - delta) * v - k_v
                    utility = u(c)
                    utility[c <= 0] = -100000
                    temp = utility + beta * value_function
                    ind = np.argmax(temp)
                    policy_rule[i] = k_v[ind]
                    rep = 1
                else:
                    rep += 1
                temp_vf = temp[ind]
                new_value_function[i] = temp_vf

            e = np.max(np.abs(value_function - new_value_function))
            iteration += 1
            value_function = np.copy(new_value_function)
            print "For iteration %i, the error is %.6e" % (iteration, e)

        print e
        print iteration
        return (value_function, policy_rule)

    def gen_plots(self, value_function, policy_rule):
        """Get a plot of the value function & policy rules.
        """
        k_l = self.params['k_l']
        k_u = self.params['k_u']
        k_n = self.params['k_n']
        k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n)
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(k_v, value_function)
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(k_v, policy_rule)
        return fig


if __name__ == "main":
    pass
