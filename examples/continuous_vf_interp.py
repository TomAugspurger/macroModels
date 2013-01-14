import numpy as np
import matplotlib.pyplot as plt

from ngm_continuous import ngm_continuous

theta, alpha, beta = 0.5, 0.8, 0.9

u_params = {'U': lambda c: 1 - np.exp(-theta * c),
        'theta': theta, 'beta': beta}


class Utility:
    """Use U for functional form.
    """
    def __init__(self, u_params):
        self.U = u_params['U']
        self.theta = u_params['theta']
        self.beta = u_params['beta']

utility = Utility(u_params)


def f(k, z):
    return (k ** alpha) * z  # Production.
W = np.exp(np.random.randn(1000))  # Random draw of shock.

gridmax, gridsize = 8, 150
grid = np.linspace(0, gridmax ** 1e-1, gridsize) ** 10

v = utility.U  # Initial condition
tol = 0.005  # Error tolerance

params = {'utility': utility, 'distribution': None, 'cdf': None,
    'gridsize': gridsize, 'grid': grid, 'tol': tol, 'f': f, 'W': W}


c = ngm_continuous(v, params)

iteration = 1
while 1:
    plt.plot(grid, v(grid), 'k-')
    new_v = c.bellman(v)[0]
    if max(abs(new_v(grid) - v(grid))) < tol:
        break
        c.iterations = iteration
        c.error = max(abs(new_v(grid) - v(grid)))
    v = new_v
    iteration += 1

plt.show()

pr = c.bellman(v)[1]
fig = plt.figure()
plt.plot(grid, v(grid), grid, pr)
