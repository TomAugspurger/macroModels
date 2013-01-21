from inspect import getargspec


class Economy(object):
    """
    Should describe

    Preferences
        * Utility Function (arguments / form)
        * Discounting
    Technology
        * Production Function (arguments / form)
        * Depreciation (goes here or elsewhere?)
    Uncertainty
        * If stochastic then some sort of markov process.

    Optionally data for simulations?
    """
    def __init__(self, preferences=None, technology=None, uncertainty=None,
                    data=None):
        self.Preferences = preferences
        self.Technology = technology
        self.Uncertainty = uncertainty
        self.Data = data
    pass


class Utility(object):
    """Use U for functional form.
    """
    def __init__(self, u_params):
        self.U = u_params['U']
        parse_args(self, self.U)
        self.discount = u_params['beta']


class Technology(object):
    def __init__(self, params):
        self.F = params['F']
        parse_args(self, self.F)
        self.depreciation = params['delta']


def parse_args(obj, d):
    r = getargspec(d)
    n = len(r.defaults)
    obj.params = {r.args[i + 1]: r.defaults[i] for i in range(n)}