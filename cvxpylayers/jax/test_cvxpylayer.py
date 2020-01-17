from cvxpylayers.jax import construct_cvxpylayer
import cvxpy as cp
import numpy as np

import jax

p1 = cp.Parameter(2)
p2 = cp.Parameter(2)
x = cp.Variable(2)

prob = cp.Problem(cp.Minimize(cp.sum_squares(p1 + p2 -x)))

layer = construct_cvxpylayer(prob, [p1, p2], [x])

params = [np.ones(2), 2 * np.ones(2)]
print (layer(params))

print (jax.vjp(layer, params))