"""
A collection of techniques to model standard neoclassical growth models via
discretization.

Currently supports non-stochastic versions of:

    -Value Function iteration
    -Howard's Improvement Algorithm.

See notes at http://www.compmacro.com/makoto/note/note_ngm_disc.pdf
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


class NGM(object):
    """Calculate a neoclassical growth model using value function iteration.
    http://www.compmacro.com/makoto/note/note_ngm_disc.pdf

    Exogenous Parameters

    * k_n: number of grid points
    * k_l: lower bound on capital stock
    * k_u: upper boind on capital stock
    * v_0: initial guess.
    * detla: depreciation rate of capital
    * u: a utility function
    * f: a production function
    * z: currently a placeholder for some stochastic shock matrix.
    * epsilon: tolerance of error
    * max_iter: Non-economic.  In case something is diverging.
    * n_h: Number of times to reuse current policy rule for value function
        iteration if using Howard improvement algorithm.

    Attributes
    ----------
    * ngm(): Solves problem and fills in some other attributes.
    * gen_plots(): Produces plot of value function & policy rule
        against the capital space.
    * value_function: Steady state value function. Comes from ngm().
    * policy_rule: Associated with value_function.
    * _is_stochastic: Helper for when stochasticity support is added.
    * is_monotonic: To be used for the monotonicity speed-up. More of a check.
    * boundry_warning: Robustness check.  Warns if capital space is
        restricting the optimal next period capital choice.



    """
    def __init__(self, alpha=.36, beta=.96, delta=.08, v_0=.01, k_n=1000,
        k_l=.05, k_u=30, epsilon=.00005, z=1, u=np.log, f=None,
        max_iter=1000, n_h=1):

        if not isinstance(alpha, (float, int)):
            raise Exception('alpha must be a float or int.')
        if not isinstance(beta, (float, int)) or beta < 0 or beta > 1:
            raise Exception('Beta should be a number between zero and one.')
        if not isinstance(delta, (float, int)) or delta < 0 or delta > 1:
            raise Exception('delta should be a number between zero and one.')
        if not isinstance(v_0, (float, int, list, np.ndarray)):
            raise Exception('Invalid initial value function.')
        if not isinstance(k_n, int):
            raise Exception('Invalid numper of points.  Must be an int.')
        if not isinstance(k_l, (float, int)):
            raise Exception('Invalid lower bound.')
        if not isinstance(k_u, (float, int)):
            raise Exception('Invalid upper bound.')
        if not isinstance(epsilon, (float, int)):
            raise Exception('Invalid tolerance level.')
        if not isinstance(max_iter, (float, int)):
            raise Exception('Invalid maximum iterations.')
        if not isinstance(n_h, int):
            raise Exception('Invlaid number for Howard\'s Improvement'\
             'Algorithm.')

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

        # Blank for now. Filled in when model is estimated.
        self.value_function = None
        self.policy_rule = None
        self._is_stochastic = None
        self.is_monotonic = None
        self.boundry_warning = False
        self.error = None
        self.iterations = None

    def ngm(self, alt=False, **kwargs):
        """
        Call like vf, pr = NGM.ngm()

        If alt, calculation of c & u is done in loop. Else it is done before.
        Getting different results since the updateding of value_function occurs
        at different times. Alt fixes value_function and loops over each k.
        Non-alt updates value_function after each iteration.  Lean toward alt?

        Non-alt running a bit under 3x slower.
        TODO: Takes args from self.params as a dict
        TODO: Improve v_0 handling.  Right now just allows for single value.
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

        f = lambda k: k ** alpha

        k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n, dtype='float')
        k_grid = np.tile(k_v, (k_n, 1)).T
        c = f(k_grid) + (1 - delta) * k_grid - k_grid.T
        utility = u(c)
        utility[c <= 0] = -100000

        e = 1
        rep = 1
        iteration = 0
        value_function = np.ones(k_n) * v_0
        new_value_function = np.zeros(k_n)
        policy_rule = np.zeros(k_n)

        while e > epsilon and iteration < max_iter:
            for i, v in enumerate(k_v):
                if rep == n_h or iteration == 0:
                    temp = utility[i] + beta * value_function
                    ind = np.argmax(temp)
                    policy_rule[i] = k_v[ind]
                    rep = 1
                    if ind in [0, k_n - 1] and iteration > 0:
                        print("Boundry Warning.  Chose %i on iteration %i"
                            % (ind, iteration))
                        self.boundry_warning = True
                else:
                    rep += 1
                temp_vf = temp[ind]
                new_value_function[i] = temp_vf
            e = np.max(np.abs(value_function - new_value_function))
            iteration += 1
            value_function = np.copy(new_value_function)
            if iteration % 10 == 0:
                print "For iteration %i, the error is %.6e" % (iteration, e)

        print('The final error is %.6e.' % e)
        print('Finished in %i iterations.' % iteration)
        self.error = e
        self.iterations = iteration
        self.value_function = value_function
        self.policy_rule = policy_rule
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

        ax1.grid()
        ax2.grid()

        ax1.set_title('Value Function and Policy Rule')
        ax1.set_ylabel('Value')
        ax2.set_xlabel('Capital Stock')
        ax2.set_ylabel('Capital Choice')
        return fig

    def _is_monotonic(self, vf, pr):
        """Check if the value function/policy rule is monotionic.
        """
        if sorted(vf) == vf and sorted(pr) == pr:
            self.is_monotonic = True

if __name__ == "main":
    pass
