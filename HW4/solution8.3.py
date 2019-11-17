import numpy as np
from cvxopt import solvers, matrix

x = [(2, 3), (1, 2), (1, 3), (2, 2)]
y = [1, 1, -1, -1]
P = np.empty((4, 4))

for i in range(4):
    for j in range(4):
        P[i][j] = (x[i][0] * x[j][0] + x[i][1] * x[j][1]) * y[i] * y[j]

P = matrix(P)
q = matrix([-1., -1., -1., -1.])
G = matrix([
    [-1., 0., 0., 0., 1., 0., 0., 0.],
    [0., -1., 0., 0., 0., 1., 0., 0.],
    [0., 0., -1., 0., 0., 0., 1., 0.],
    [0., 0., 0., -1., 0., 0., 0., 1.]
])
h = matrix([0., 0., 0., 0., 10., 10., 10., 10.])
A = matrix([1., 1., -1., -1.], (1, 4))
b = matrix([0.])
result = solvers.qp(P, q, G, h, A, b)
print(result['x'])
