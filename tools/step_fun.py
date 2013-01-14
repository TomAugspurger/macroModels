import numpy as np


class StepFun:
    """Based on http://johnstachurski.net/lectures/fvi_rpd.html"""
    def __init__(self, X, Y):
        """
        Parameters
        ----------

        *X: increasing array with length n.
        *Y: any array of length n.

        Returns
        -------

        *s: a step function with sum_{i=0}^{n - 1} Y[i] 1{X[i]} <= x < X[i + 1]
        with X[n] := infty
        """
        self.X, self.Y = X, Y

    def __call__(self, x):
        """Evaluate the step function at x.
        """
        if x < self.X[0]:
            return 0.0
        i = self.X.searchsorted(x, side='right') - 1
        return self.Y[i]

    def expectation(self, F):
        """Compute expection of s(Y) given F, the cdf of Y.
        """
        probs = np.append(F(self.X), 1)
        return np.dot(self.Y, probs[1:] - probs[:-1])
