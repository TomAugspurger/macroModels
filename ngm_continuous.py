"""Based on http://johnstachurski.net/lectures/fvi_rpd.html
"""

from __future__ import division

import types

import numpy as np
from scipy.optimize import fminbound
from tools.StepFun import StepFun
from tools.linear_interpolation import LinInterp


class ngm_continuous:
    def __init__(self, step, params):
        """Estimate a value function and policy rule in continuous space.

        Parameters
        ----------

        * step : An instance of StepFun.

        params includes:
        * utility : A class with attributes including a utility function
            (along with any parameters), and a discount factor beta.
        * L : A distribution (e.g. lognormal)
        * G : The cdf of distribution L.
        * gridsize : Int with the number of points in the grid.
        * grid : Grid to iterate on.
        * f : A production function (e.g. k ** alpha)
        """
        self.utility = params['utility']
        try:
            self.L = params['distribution']
            self.G = params['cdf']
        except KeyError:
            self.L, self.G = None, None
        self.gridsize = params['gridsize']
        self.grid = params['grid']
        self.f = params['f']
        try:
            self.W = params['W']
        except KeyError:
            self.W = None

    def cum_dist_fctn(self, G, c):
        """Returns the cdf of c * W.
        e.g. c is a constant and W is lognormal.
        """
        if c == 0.0:
            return np.vectorize(lambda x: x if x < 0 else 1)
        return lambda x: G(x / c)

    def arg_maximum(self, h, a, b):
        """Compute argmax of a function h on [a, b]."""
        return fminbound(lambda x: -h(x), a, b)

    def bellman(self, w):
        """
        Approximate the Bellman operator.

        Parameters
        ----------

        * w: instance of StepFun where x.X = grid

        Returns
        -------

        * New instance of StepFun.
        """
        if isinstance(w, StepFun):
            Tw = np.empty(self.gridsize)
            g = np.empty(self.gridsize)
            for i, y in enumerate(self.grid):
                h = lambda k: self.utility.U(y - k) + self.utility.beta * w.expectation(
                    self.cum_dist_fctn(self.G, self.f(k)))
                g[i] = self.arg_maximum(h, 0, y)
                Tw[i] = h(g[i])

            return (StepFun(self.grid, Tw), g)
        elif isinstance(w, (LinInterp, types.FunctionType)):
                pr = []
                vals = []
                for y in self.grid:
                    h = lambda k: self.utility.U(y - k) + self.utility.beta * (
                        np.mean(w(self.f(k, self.W))))
                    temp = self.arg_maximum(h, 0, y)
                    pr.append(temp)
                    vals.append(h(temp))
                return (LinInterp(self.grid, vals), pr)
