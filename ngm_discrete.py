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

from tools.stochastic import Stochastic


class NGM(object):
    """Calculate a neoclassical growth model using value function iteration.
    http://www.compmacro.com/makoto/note/note_ngm_disc.pdf

    Exogenous Parameters

    * k_n: number of grid points
    * k_l: lower bound on capital stock
    * k_u: upper boind on capital stock
    * v_0: initial guess.
    * detla: depreciation rate of capital
    * utility: A class with utility function, args, constraints.
    * f: a production function
    * z: Class describing stochasticity (space & transition matrix) or 1.
    * T: Transition matrix (markov process) for shocks.
    * epsilon: tolerance of error
    * max_iter: Non-economic.  In case something is diverging.
    * n_h: Number of times to reuse current policy rule for value function
        iteration if using Howard improvement algorithm.
    * s is a function mapping shocks into R. e.g. s(z) = z; or s(z) = e**z.
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


    TODO: Figure out a way to intelligently handle utility, production, etc.
    functions.  inspect.getargspec() might be helpful.

    Current plan is to make utility, production a class.  Each will define a
    function, (maybe argmunts (NOT parameters) if I can't figure out the
    python way of parsing those (**kwargs?)), and a dict? list? of
    constraints.

    Ok this is going to be harder than I thought.  Need to think on it more.
    If hours worked is to be a choice variable then the space will get very
    large very quickly.
    """
    def __init__(self, beta=.96, delta=.08, v_0=.01, k_n=1000,
        k_l=.05, k_u=30, epsilon=.00005, u=(lambda x, h=1, theta=.5:
        theta * np.log(x) + (1 - theta) * np.log(h)), f=(lambda k, alpha=.36:
        k ** alpha), max_iter=1000, n_h=1, z=1, T=None, simulations=1,
        periods=1, s=lambda z: z, utility=None):

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

        self.params = {
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
                    'n_h': n_h,
                    'simulations': simulations,
                    'periods': periods,
                    'T': T,
                    's': s}

        # Blank for now. Filled in when model is estimated.
        self.value_function = {}
        self.policy_rule = {}
        if isinstance(z, Stochastic):
            self.Z = z
            self._is_stochastic = True
        else:
            self.Z = 1
            self._is_stochastic = False
        self.is_monotonic = None
        self.boundry_warning = False
        self.error = None
        self.iterations = None

    def ngm(self, attr_num=1, **kwargs):
        """
        attr_num: Crummy way to accept multiple value functions for
            stochastic case.

        Call like vf, pr = NGM.ngm()

        For multple cases (e.g. z = np.array([0.9479, 1, 1.0549]) do
            for i, v in enumerate(z):
                model.ngm(attr_num=i, z=v)

        and then vf1, vf2, vf3 = model.value_function.values()
        and      pr1, pr2, pr3 = model.policy_rule.values()
        TODO: Takes args from self.params as a dict
        TODO: Improve v_0 handling.  Right now just allows for single value.
        """
        k_l, k_u = self.params['k_l'], self.params['k_u']
        k_n, beta = self.params['k_n'], self.params['beta']
        delta, epsilon = self.params['delta'], self.params['epsilon']
        u, max_iter = self.params['u'], self.params['max_iter']
        v_0, n_h = self.params['v_0'], self.params['n_h']
        f, s = self.params['f'], self.params['s']

        k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n, dtype='float')
        self.k_v = k_v
        k_grid = np.tile(k_v, (k_n, 1)).T

        e = 1
        rep = 1
        iteration = 0

        if not self._is_stochastic:  # Deterministic case.
            c = s(self.Z) * f(k_grid) + (1 - delta) * k_grid - k_grid.T
            _utility = u(c)
            _utility[c <= 0] = -100000

            value_function = np.ones(k_n) * v_0
            new_value_function = np.zeros(k_n)
            policy_rule = np.zeros(k_n)

            while e > epsilon and iteration < max_iter:
                for i, v in enumerate(k_v):
                    if rep == n_h or iteration == 0:
                        temp = _utility[i] + beta * value_function
                        ind = np.argmax(temp)
                        policy_rule[i] = k_v[ind]
                        rep = 1
                        if ind in [0, k_n - 1] and iteration > 0:
                            print("Boundry Warning.  Chose %i on iteration %i"
                                % (ind, iteration))
                            self.boundry_warning = True
                    else:
                        rep += 1
                    temp = _utility[i] + beta * value_function
                    new_value_function[i] = temp[ind]
                e = np.max(np.abs(value_function - new_value_function))
                iteration += 1
                value_function = np.copy(new_value_function)
                if iteration % 10 == 0:
                    print "For iteration %i, the error is %.6e" % (iteration, e)
        else:
            c = np.zeros((k_n, k_n, self.Z.size))

            for i, v in enumerate(self.Z.z):
                c[:, :, i] = s(v) * f(k_grid) - k_grid.T
            _utility = u(c)
            _utility[c <= 0] = -100000

            value_function = np.tile(np.log(k_v), (self.Z.size, 1)).T
            new_value_function = np.zeros([k_n, self.Z.size])
            policy_rule = np.zeros([k_n, self.Z.size])

            while e > epsilon and iteration < max_iter:
                for j, w in enumerate(self.Z.z):
                    for i, v in enumerate(k_v):
                        if rep == n_h or iteration == 0:
                            temp = _utility[i, :, j] + beta * value_function[:, j]
                            ind = np.argmax(temp)
                            policy_rule[i, j] = k_v[ind]
                            rep = 1
                            if ind in [0, k_n - 1] and iteration > 0:
                                print("Boundry Warning.  Chose %i on iteration %i"
                                    % (ind, iteration))
                                self.boundry_warning = True
                        else:
                            rep += 1
                        new_value_function[i, j] = temp[ind]
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

    def gen_plots(self, value_function, policy_rule, fig=None):
        """Get a plot of the value function & policy rules.
        Handling for multiplots.  If you want to add to an existing
        plot (say multiple vf/pr's) pass that figure as fig.
        Defualt is to return a new figure.
        """
        k_l = self.params['k_l']
        k_u = self.params['k_u']
        k_n = self.params['k_n']
        k_v = np.arange(k_l, k_u, (k_u - k_l) / k_n)
        if fig is None:
            fig = plt.figure()
        else:
            fig = fig
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
        if vf.sort() == vf and pr == pr.sort():
            self.is_monotonic = True

if __name__ == "main":
    pass
