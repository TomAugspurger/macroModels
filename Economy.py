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
    Stochastic
        * If stochastic then some sort of markov process.

    Optionally data for simulations?
    """
    def __init__(self, preferences=None, technology=None, stochastic=None,
                    model=None, data=None):
        self.Preferences = preferences
        self.Technology = technology
        self.Stochastic = stochastic
        self.Model = model
        self.Data = data
    pass


class Preferences(object):
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


class Model(object):
    def __init__(self, params=None):
        if params == None:
            self.k_l = .05
            self.k_u = 20
            self.k_n = 1000
            self.epsilon = 5e-5
            self.max_iter = 10000
            self.n_h = 1
            self.v_0 = 1
        else:
            self.k_l = params['k_l']
            self.k_u = params['k_u']
            self.k_n = params['k_n']
            self.epsilon = params['epsilon']
            self.max_iter = params['max_iter']
            self.n_h = params['n_h']
            self.v_0 = params['v_0']


class Stochastic:
    def __init__(self, z, T, s):
        """
        z is an array of shocks.
        T is the transition matrix.
        s is the functional form of the shock (e.g. s(x) = x).
        """
        self.shock_space = z
        self.transition = T
        self.form = s
        self.size = len(z)


def parse_args(obj, d):
    r = getargspec(d)
    n = len(r.defaults)
    obj.params = {r.args[i + 1]: r.defaults[i] for i in range(n)}
