import numpy as np
import matplotlib.pyplot as plt

from ngm_discrete import NGM
import economy

T = np.array([[0.910507618836914, 0.089492259543859, 0.000000121619227],
    [0.028100505607270, 0.943798980953369, 0.028100513439361],
    [0.000000121619227, 0.089492259543859, 0.910507618836914]])
z = np.array([0.947938865630057, 1, 1.05492035009593])
stochastic = economy.Stochastic(z, T, lambda x: x)


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
    model=model, stochastic=stochastic)

model = NGM(economy)

for i, v in enumerate(z):
    model.ngm(attr_num=i, z=v)

fig = model.gen_plots(model.value_function[0], model.policy_rule[0])
model.gen_plots(model.value_function[1], model.policy_rule[1], fig=fig)
model.gen_plots(model.value_function[2], model.policy_rule[2], fig=fig)
plt.draw()
