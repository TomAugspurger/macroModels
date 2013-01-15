"""
Based on Martin's HW 9:
g) Assume that alpha=1/3, l=2, rho=0.0, epsilon=0.02, pi_1=pi_2=0.5,
and choose theta so that people work 1/3 of their time on average.
Implement the algorithm you proposed in question e). Display the value function
from question f) together with the one you obtain numerically for both shocks.
Also display the policy rule from question f) together with the one you obtain
numerically for both shocks.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from test_stochastic import NGM
from tools.stochastic import Stochastic

# Parameters
alpha, l, rho, epsilon, pi_1, pi_2 = 1 / 3, 2, 0.0, 0.02, 0.5, 0.5
beta, h = .95, 1 / 3  # h exogenously determined.

# Policy Rule for labor (analytically)
# h = (theta -  theta * alpha) / (1 - beta * alpha + beta *
#                                 alpha * theta - theta * alpha)

theta = 44 / 121
k_l, k_u, k_n = 0.02, 0.08, 200

T = np.array([[0.5, 0.5], [0.5, 0.5]])
z = np.array([np.exp(-0.02,), np.exp(0.02)])


def u(c, h=h, theta=theta):
    return theta * np.log(c) + (1 - theta) * np.log(1 - h)


def f(k, h=h, alpha=alpha):
    return k ** alpha * h ** (1 - alpha)


stoch = Stochastic(z, T)

model = NGM(k_u=1, z=stoch, k_n=k_n)
vf, pr = model.ngm()


fig = model.gen_plots(vf[:, 0], pr[:, 0])
fig = model.gen_plots(vf[:, 1], pr[:, 1], fig=fig)

plt.draw()
