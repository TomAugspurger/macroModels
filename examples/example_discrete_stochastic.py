import numpy as np
import matplotlib.pyplot as plt

from ngm_discrete import NGM


T = np.array([[0.910507618836914, 0.089492259543859, 0.000000121619227],
    [0.028100505607270, 0.943798980953369, 0.028100513439361],
    [0.000000121619227, 0.089492259543859, 0.910507618836914]])
z = np.array([0.947938865630057, 1, 1.05492035009593])
chain = np.zeros([251, 500])

model = NGM(z=z)

for i, v in enumerate(z):
    model.ngm(attr_num=i, z=v)

fig = model.gen_plots(model.value_function[0], model.policy_rule[0])
model.gen_plots(model.value_function[1], model.policy_rule[1], fig=fig)
model.gen_plots(model.value_function[2], model.policy_rule[2], fig=fig)
plt.draw()
