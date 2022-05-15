import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
from scipy.optimize import LinearConstraint, NonlinearConstraint

from zoutendijk import solve


R1 = 1
R2 = 0.3  # 0.5 for internal, 0.3 for border optimal x


def f0(x):
    x1, x2, x3 = x
    return x1 ** 2 + x2 ** 2 + np.cos(2 * x1 + 3 * x2) + x3 ** 2


def grad0(x):
    sinus = np.sin(2 * x[0] + 3 * x[1])
    return np.array([2 * x[0] - 2 * sinus, 2 * x[1] - 3 * sinus, 2 * x[2]])


def f1(x):
    return x[0]**2 + x[1]**2 + x[2]**2 - R1


def grad1(x):
    return np.array([2*x[0], 2 * x[1], 2 * x[2]])


def f2(x):
    return (x[0] - 1)**2 + (x[1] - 1)**2 + x[2]**2 - R2


def grad2(x):
    return np.array([2 * (x[0] - 1), 2 * (x[1] - 1), 2 * x[2]])


def f3(x):
    return (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 1 - x[2]


def grad3(x):
    return np.array([2 * (x[0] - 0.5), 2 * (x[1] - 0.5), -1])


A = np.array([[1.5, -1, 0]])
b = np.array([0])

x, trace = solve([f0, f1, f2, f3], [grad0, grad1, grad2, grad3], A, b, x0=None)
print(f'opt x = {x}, f = {f0(x)}')

lc = LinearConstraint(A, b, b)
nlc1 = NonlinearConstraint(f1, -np.inf, 0)
nlc2 = NonlinearConstraint(f2, -np.inf, 0)
nlc3 = NonlinearConstraint(f3, -np.inf, 0)
sp_x = opt.minimize(f0, x0=np.array([0.1, 0.1, 0.1]), constraints=[lc, nlc1, nlc2, nlc3]).x
print(f'scipy solution: x = {sp_x}, f = {f0(sp_x)}')


'''
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
'''