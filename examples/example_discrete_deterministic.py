from __future__ import division

import numpy as np
from ngm_discrete import NGM
import economy


def u(c, h=0, theta=.5):
        return theta * np.log(c) + (1 - theta) * np.log(1 - h)
u_params = {'U': u, 'beta': .96}
preferences = economy.Preferences(u_params)


def f(x, alpha=.36):
    return x ** alpha
f_params = {'F': f, 'delta': .08}
technology = economy.Technology(f_params)

model = economy.Model()  # Defaults

economy = economy.Economy(preferences=preferences, technology=technology,
    model=model)

ngm = NGM(economy)

vf, pr = ngm.ngm()

fig = ngm.gen_plots(vf, pr)
