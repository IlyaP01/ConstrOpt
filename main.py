import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from zoutendijk import solve


def f0(x):
    x1, x2 = x
    return x1 ** 2 + x2 ** 2 + np.cos(2 * x1 + 3 * x2)


def grad0(x):
    sinus = np.sin(2 * x[0] + 3 * x[1])
    return np.array([2 * x[0] - 2 * sinus, 2 * x[1] - 3 * sinus])


def f1(x):
    return x[0]**2 + x[1]**2 - 0.5


def grad1(x):
    return np.array([2*x[0], 2 * x[1]])


def f2(x):
    return (x[0] - 0.6)**2 + (x[1] - 0.5)**2 - 0.5


def grad2(x):
    return np.array([2 * (x[0] - 0.6), 2 * (x[1] - 0.5)])


def f3(x):
    pass


def grad3(x):
    pass


A = np.array([[1, -1]])
b = np.array([0])

x, trace = solve([f0, f1, f2], [grad0, grad1, grad2], A, b, x0=None)
print(f'opt x = {x}')

fig, ax = plt.subplots()
X = np.arange(-0.5, 2, 0.1)
Y = np.arange(-0.5, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = f0(np.array([X, Y]))
cs = ax.contour(X, Y, Z, levels=30)
ax.clabel(cs, inline=1, fontsize=10)
ax.plot(np.linspace(-0.5, 2, 2), np.linspace(-0.5, 2, 2))
ax.plot(np.linspace(-0.5, 0.7, 100), np.sqrt(0.5 - np.linspace(-0.5, 0.7, 100)**2))
ax.plot([p[0] for p in trace], [p[1] for p in trace])
ax.plot(x[0], x[1], 'ro')
plt.show()
