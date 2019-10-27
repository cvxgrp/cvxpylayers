#!/usr/bin/env python3

import cvxpy as cp
import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import torch
from cvxpylayers.torch.cvxpylayer import CvxpyLayer


def simple_qp():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('simple qp')
    npr.seed(0)
    nx, ncon = 2, 3

    G = cp.Parameter((ncon, nx))
    h = cp.Parameter(ncon)
    x = cp.Variable(nx)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    cons = [G * x <= h]
    prob = cp.Problem(obj, cons)

    data, chain, inv_data = prob.get_problem_data(solver=cp.SCS)
    param_prob = data[cp.settings.PARAM_PROB]
    print(param_prob.A.A)

    x0 = npr.randn(nx)
    s0 = npr.randn(ncon)
    G.value = npr.randn(ncon, nx)
    h.value = G.value.dot(x0) + s0

    prob.solve(solver=cp.SCS)

    delC = npr.randn(param_prob.c.shape[0])[:-1]
    delA = npr.randn(param_prob.A.shape[0])
    num_con = delA.size // (param_prob.x.size + 1)
    delb = delA[-num_con:]
    delA = delA[:-num_con]
    delA = sp.csc_matrix(np.reshape(delA, (num_con, param_prob.x.size)))
    del_param_dict = param_prob.apply_param_jac(delC, delA, delb)
    print(del_param_dict)
    var_map = param_prob.split_solution(npr.randn(param_prob.x.size))
    print(var_map)
    print(param_prob.split_adjoint(var_map))

    print(x.value)


def full_qp():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('full qp')
    npr.seed(0)
    nx, ncon_eq, ncon_ineq = 5, 2, 3

    Q = cp.Parameter((nx, nx))
    p = cp.Parameter((nx, 1))
    A = cp.Parameter((ncon_eq, nx))
    b = cp.Parameter(ncon_eq)
    G = cp.Parameter((ncon_ineq, nx))
    h = cp.Parameter(ncon_ineq)
    x = cp.Variable(nx)
    # obj = cp.Minimize(0.5*cp.quad_form(x, Q) + p.T * x)
    obj = cp.Minimize(0.5 * cp.sum_squares(Q@x) + p.T * x)
    cons = [A * x == b, G * x <= h]
    prob = cp.Problem(obj, cons)

    x0 = npr.randn(nx)
    s0 = npr.randn(ncon_ineq)

    G.value = npr.randn(ncon_ineq, nx)
    h.value = G.value.dot(x0) + s0

    A.value = npr.randn(ncon_eq, nx)
    b.value = A.value.dot(x0)

    L = npr.randn(nx, nx)
    Q.value = L.T  # L.dot(L.T)
    p.value = npr.randn(nx, 1)

    prob.solve(solver=cp.SCS)
    print(x.value)


def ball_con():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('ball con')
    npr.seed(0)

    n = 2

    A = cp.Parameter((n, n))
    z = cp.Parameter(n)
    p = cp.Parameter(n)
    x = cp.Variable(n)
    t = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - p))
    # TODO automate introduction of variables.
    cons = [0.5 * cp.sum_squares(A * t) <= 1, t == (x - z)]
    prob = cp.Problem(obj, cons)

    L = npr.randn(n, n)
    A.value = L.T
    z.value = npr.randn(n)
    p.value = npr.randn(n)

    prob.solve(solver=cp.SCS)
    print(x.value)


def relu():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('relu')
    npr.seed(0)

    n = 4
    _x = cp.Parameter(n)
    _y = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(_y - _x))
    cons = [_y >= 0]
    prob = cp.Problem(obj, cons)

    _x.value = npr.randn(n)

    prob.solve(solver=cp.SCS)
    print(_y.value)


def sigmoid():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('sigmoid')
    npr.seed(0)

    n = 4
    _x = cp.Parameter((n, 1))
    _y = cp.Variable(n)
    obj = cp.Minimize(-_x.T * _y - cp.sum(cp.entr(_y) + cp.entr(1. - _y)))
    prob = cp.Problem(obj)

    _x.value = npr.randn(n, 1)

    prob.solve(solver=cp.SCS)
    print(_y.value)


def softmax():
    # print(f'--- {sys._getframe().f_code.co_name} ---')
    print('softmax')
    npr.seed(0)

    d = 4
    _x = cp.Parameter((d, 1))
    _y = cp.Variable(d)
    obj = cp.Minimize(-_x.T * _y - cp.sum(cp.entr(_y)))
    cons = [sum(_y) == 1.]
    prob = cp.Problem(obj, cons)

    _x.value = npr.randn(d, 1)

    prob.solve(solver=cp.SCS)
    print(_y.value)


def sdp():
    print('sdp')
    npr.seed(0)

    d = 2
    X = cp.Variable((d, d), PSD=True)
    Y = cp.Parameter((d, d))
    obj = cp.Minimize(cp.trace(Y * X))
    prob = cp.Problem(obj, [X >= 1])

    Y.value = np.abs(npr.randn(d, d))
    print(Y.value.sum())

    prob.solve(solver=cp.SCS, verbose=True)
    print(X.value)


def running_example():
    print("running example")
    m = 20
    n = 10
    x = cp.Variable((n, 1))
    F = cp.Parameter((m, n))
    g = cp.Parameter((m, 1))
    lambd = cp.Parameter((1, 1), nonneg=True)
    objective_fn = cp.norm(F @ x - g) + lambd * cp.norm(x)
    constraints = [x >= 0]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    assert problem.is_dcp()
    assert problem.is_dpp()
    print("is_dpp: ", problem.is_dpp())

    F_t = torch.randn(m, n, requires_grad=True)
    g_t = torch.randn(m, 1, requires_grad=True)
    lambd_t = torch.rand(1, 1, requires_grad=True)
    layer = CvxpyLayer(problem, parameters=[F, g, lambd], variables=[x])
    x_star, = layer(F_t, g_t, lambd_t)
    x_star.sum().backward()
    print("F_t grad: ", F_t.grad)
    print("g_t grad: ", g_t.grad)


if __name__ == '__main__':
    simple_qp()
    full_qp()
    ball_con()
    relu()
    sigmoid()
    softmax()
    running_example()
    # sdp()
