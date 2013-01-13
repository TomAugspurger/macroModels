"""Based on http://johnstachurski.net/lectures/fvi_rpd.html
"""

from __future__ import division

import numpy as np
from scipy.optimize import fminbound
from tools.StepFun import StepFun


class ngm_continuous:
    def __init__(self, step, params):
        """
        """
        self.U = params.utility
        self.L = params.distribution
        self.G = params.cdf
        self.gridmax, self.gridsize = params.gridmax, params.gridsize
        self.grid = params.grid

    def cum_dist_fctn(self, c, G):
        """Returns the cdf of c * W.
        e.g. c is a constant and W is lognormal.
        """
        if c == 0.0:
            return np.vectorize(lambda x: x if x < 0 else 1)
        return lambda x: G(x / c)

    def maximum(self, h, a, b):
        """Compute max of h on [a, b]."""
        return h(fminbound(lambda x: -h(x), a, b))

    def bellman(self, w, gridsize):
        """
        Approximate the Bellman operator.

        Parameters
        ----------

        * w: instance of StepFun where x.X = grid

        Returns
        -------

        * New instance of StepFun.
        """
        Tw = np.empty(gridsize)
        for i, y in enumerate(self.grid):
            h = lambda k: self.U(y - k) + self.U.rho * w.expectation(
                self.cum_dist_fctn(k ** self.U.alpha))
            Tw[i] = self.maximum(h, 0, y)

        return StepFun(self.grid, Tw)
