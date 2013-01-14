# Based on http://johnstachurski.net/lectures/funapprox.html#linear-interpolation

from scipy import interp


class LinInterp:
    "Provides linear interpolation in one dimension."

    def __init__(self, X, Y):
        """Parameters: X and Y are sequences or arrays
        containing the (x,y) interpolation points."""
        self.X, self.Y = X, Y

    def __call__(self, z):
        return interp(z, self.X, self.Y)
