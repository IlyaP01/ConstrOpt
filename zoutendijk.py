from scipy.optimize import linprog
import numpy as np


def solve_lin_problem(grads, ksis, A, x):
    n = len(A[0])

    c = np.zeros(n)
    c = np.insert(c, 0, 1)

    A_ub = []
    for ksi, grad, in zip(ksis, grads):
        Ai = [-ksi]
        for coord in grad(x):
            Ai.append(coord)
        A_ub.append(Ai)
    b_ub = np.zeros(len(A_ub))

    bounds = [(None, None)]
    for i in range(n):
        bounds.append((-1, 1))

    A_eq = np.insert(A, 0, np.zeros(len(A)), axis=1)
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=np.zeros(len(A)), bounds=bounds)
    eta = res.x[0]
    s = res.x[1:]

    return eta, s


def find_alpha(fs, x, s, ksis, eta, _lambda):
    alpha = 1
    x_next = x + alpha * s
    while not (fs[0](x_next) <= fs[0](x) + 0.5 * ksis[0] * eta * alpha and all([f(x_next) <= 0 for f in fs[1:]])):
        alpha *= _lambda
        x_next = x + alpha * s

    return alpha


def find_start_x(fs, grads, A, b):
    new_fs = [lambda x: f(x[1:]) - x[0] for f in fs[1:]]
    new_fs.insert(0, lambda x: x[0])
    new_grads = [lambda x: np.insert(grad(x[1:]), 0, -1) for grad in grads[1:]]
    grad0 = np.zeros(len(A[0]) + 1)
    grad0[0] = 1
    new_grads.insert(0, lambda x: grad0)
    A_new = np.insert(A, 0, np.zeros(len(A)), axis=1)
    A_square = A[:, 0:len(A)]
    b_for_square = np.copy(b)
    for i in range(len(b_for_square)):
        for j in range(len(A_square[0]), len(A[0]) - len(A_square[0]) + 1):
            b_for_square[i] -= A[i][j]
    x0 = np.linalg.solve(A_square, b_for_square)
    x0 = np.pad(x0, (0, len(A[0]) - len(A_square[0])), 'constant', constant_values=(1,))
    x0 = np.insert(x0, 0, max(f(x0) for f in fs[1:]))
    x, trace = solve(new_fs, new_grads, A_new, b, x0, find_x0=True)
    return x[1:]


def solve(fs, grads, A, b, x0=None, find_x0=False):
    ksis = np.ones(len(fs))
    _lambda = 0.75
    delta = 0.01
    eps = 1e-4

    if x0 is None:
        x = find_start_x(fs, grads, A, b)
    else:
        x = x0

    iter_n = 0
    trace = []
    while True:
        iter_n += 1
        trace.append(x)
        if iter_n >= 100:
            print('Max iter limit!')
            break
        aai = [i for i, f in enumerate(fs) if i == 0 or -delta < f(x)]  # almost active constraints
        eta, s = solve_lin_problem(np.take(grads, aai), np.take(ksis, aai), A, x)
        if eta < -delta:
            alpha = find_alpha(fs, x, s, ksis, eta, _lambda)
            x = x + alpha * s
        elif eta < 0:
            delta *= _lambda

        if eta < 0 and find_x0:
            break

        pass_indexes = [i for i in range(len(fs)) if i not in aai]
        delta0 = -max([f(x) for f in np.take(fs, pass_indexes)]) if pass_indexes else np.inf
        if abs(eta) < eps and delta < delta0:
            print('Found optimal x')
            break

    return x, trace
