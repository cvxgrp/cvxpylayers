import unittest
import cvxpy as cp
import diffcp
import numpy as np
import jax


from cvxpylayers.jax import CvxpyLayer


def numerical_grad(f, params, param_values, delta=1e-6):
    size = sum(np.prod(v.shape) for v in param_values)
    values = np.zeros(size)
    offset = 0
    for param, value in zip(params, param_values):
        values[offset:offset + param.size] = value.flatten()
        param.value = values[offset:offset + param.size].reshape(param.shape)
        offset += param.size

    numgrad = np.zeros(values.shape)
    for i in range(values.size):
        old = values[i]
        values[i] = old + 0.5 * delta
        left_soln = f()

        values[i] = old - 0.5 * delta
        right_soln = f()

        numgrad[i] = (left_soln - right_soln) / delta
        values[i] = old

    numgrads = []
    offset = 0
    for param in params:
        numgrads.append(
            numgrad[offset:offset + param.size].reshape(param.shape))
        offset += param.size
    return numgrads


class TestCvxpyLayer(unittest.TestCase):

    def test_docstring_example(self):
        np.random.seed(0)
        k1 = jax.random.PRNGKey(0)

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tf = jax.random.normal(key=k1, shape=(m, n))
        b_tf = jax.random.normal(key=k1, shape=(m,))

        # solution, = cvxpylayer(A_tf, b_tf)
        # summed_solution = jax.numpy.sum(solution)
        def summed_solution(x, y):
            solution = cvxpylayer(x, y)
            return solution
        gradA = jax.grad(summed_solution)(A_tf, b_tf)
        # gradb = jax.grad(summed_solution, argnums=1)(A_tf, b_tf)

        # def f():
        #     problem.solve(solver=cp.SCS, eps=1e-10)
        #     return np.sum(x.value)

        # numgradA, numgradb = numerical_grad(f, [A, b], [A_tf, b_tf])
        # np.testing.assert_almost_equal(gradA, numgradA, decimal=4)
        # np.testing.assert_almost_equal(gradb, numgradb, decimal=4)

    # def test_simple_qp(self):
    #     np.random.seed(0)
    #     tf.random.set_seed(0)
    #     nx, ncon = 2, 3

    #     G = cp.Parameter((ncon, nx), name='G')
    #     h = cp.Parameter(ncon, name='h')
    #     x = cp.Variable(nx)
    #     obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    #     cons = [G * x <= h]
    #     problem = cp.Problem(obj, cons)

    #     cvxlayer = CvxpyLayer(problem, [G, h], [x])
    #     x0 = jax.random.normal((nx, 1))
    #     s0 = jax.random.normal((ncon, 1))
    #     G_t = jax.random.normal((ncon, nx))
    #     h_t = jax.squeeze(jax.matmul(G_t, x0) + s0)

    #     with tf.GradientTape() as tape:
    #         tape.watch(G_t)
    #         tape.watch(h_t)
    #         soln = cvxlayer(G_t, h_t, solver_args={'eps': 1e-10})
    #     soln = {x.name(): soln[0]}

    #     grads = tape.gradient(soln, [G_t, h_t])
    #     gradG = grads[0]
    #     gradh = grads[1]

    #     G.value = G_t
    #     h.value = h_t
    #     problem.solve(solver=cp.SCS)
    #     self.assertEqual(len(soln.values()), len(problem.variables()))
    #     np.testing.assert_almost_equal(
    #         x.value, list(soln.values())[0], decimal=5)

    #     def f():
    #         problem.solve(solver=cp.SCS, eps=1e-10)
    #         return np.sum(x.value)

    #     numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
    #     np.testing.assert_almost_equal(gradG, numgradG, decimal=4)
    #     np.testing.assert_almost_equal(gradh, numgradh, decimal=4)

    # def test_simple_qp_with_solver_args(self):
    #     np.random.seed(0)
    #     tf.random.set_seed(0)
    #     nx, ncon = 2, 3

    #     G = cp.Parameter((ncon, nx), name='G')
    #     h = cp.Parameter(ncon, name='h')
    #     x = cp.Variable(nx)
    #     obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    #     cons = [G * x <= h]
    #     problem = cp.Problem(obj, cons)

    #     cvxlayer = CvxpyLayer(problem, [G, h], [x])
    #     x0 = jax.random.normal((nx, 1))
    #     s0 = jax.random.normal((ncon, 1))
    #     G_t = jax.random.normal((ncon, nx))
    #     h_t = jax.squeeze(jax.matmul(G_t, x0) + s0)

    #     with tf.GradientTape() as tape:
    #         tape.watch(G_t)
    #         tape.watch(h_t)
    #         soln = cvxlayer(G_t, h_t, solver_args={'eps': 1e-10})
    #     soln = {x.name(): soln[0]}

    #     grads = tape.gradient(soln, [G_t, h_t])
    #     gradG = grads[0]
    #     gradh = grads[1]

    #     G.value = G_t.numpy()
    #     h.value = h_t.numpy()
    #     problem.solve(solver=cp.SCS)
    #     self.assertEqual(len(soln.values()), len(problem.variables()))
    #     np.testing.assert_almost_equal(
    #         x.value, list(soln.values())[0], decimal=5)

    #     def f():
    #         problem.solve(solver=cp.SCS, eps=1e-10)
    #         return np.sum(x.value)

    #     numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
    #     np.testing.assert_almost_equal(gradG, numgradG, decimal=4)
    #     np.testing.assert_almost_equal(gradh, numgradh, decimal=4)

    # def test_simple_qp_batched(self):
    #     np.random.seed(0)
    #     tf.random.set_seed(0)
    #     nbtch, nx, ncon = 4, 3, 2

    #     G = cp.Parameter((ncon, nx), name='G')
    #     h = cp.Parameter(ncon, name='h')
    #     x = cp.Variable(nx, name='x')
    #     obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    #     cons = [G * x <= h]
    #     problem = cp.Problem(obj, cons)

    #     cvxlayer = CvxpyLayer(problem, [G, h], [x])
    #     x0 = jax.random.normal((nx, 1))
    #     s0 = jax.random.normal((ncon, 1))
    #     G_t = jax.random.normal((nbtch, ncon, nx))
    #     h_t = jax.squeeze(jax.numpy.tensordot(G_t, x0, axes=1) + s0)

    #     with tf.GradientTape() as tape:
    #         tape.watch(G_t)
    #         tape.watch(h_t)
    #         soln = cvxlayer(G_t, h_t, solver_args={'eps': 1e-10})
    #     soln = {x.name(): soln[0]}

    #     grads = tape.gradient(soln, [G_t, h_t])
    #     gradG = grads[0]
    #     gradh = grads[1]

    #     solns = [jax.numpy.squeeze(t) for t in jax.numpy.split(soln['x'], nbtch)]
    #     Gs = [jax.numpy.squeeze(t) for t in jax.numpy.split(G_t, nbtch)]
    #     hs = [jax.numpy.squeeze(t) for t in jax.numpy.split(h_t, nbtch)]
    #     gradGs = [jax.numpy.squeeze(t).numpy() for t in jax.numpy.split(gradG, nbtch)]
    #     gradhs = [jax.numpy.squeeze(t).numpy() for t in jax.numpy.split(gradh, nbtch)]

    #     for soln, G_t, h_t, gG, gh in zip(solns, Gs, hs, gradGs, gradhs):
    #         G.value = G_t
    #         h.value = h_t
    #         problem.solve(solver=cp.SCS)
    #         np.testing.assert_almost_equal(x.value, soln, decimal=5)

    #         def f():
    #             problem.solve(solver=cp.SCS, eps=1e-10)
    #             return np.sum(x.value)

    #         numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
    #         np.testing.assert_almost_equal(gG, numgradG, decimal=2)
    #         np.testing.assert_almost_equal(gh, numgradh, decimal=2)

    # def test_logistic_regression(self):
    #     np.random.seed(243)
    #     N, n = 10, 2

    #     def sigmoid(z):
    #         return 1 / (1 + np.exp(-z))

    #     X_np = np.random.randn(N, n)
    #     a_true = np.random.randn(n, 1)
    #     y_np = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))

    #     X_tf = jax.numpy.array(X_np)
    #     lam_tf = jax.numpy.Variable(1.0 * jax.numpy.ones(1))

    #     a = cp.Variable((n, 1))
    #     X = cp.Parameter((N, n))
    #     lam = cp.Parameter(1, nonneg=True)
    #     y = y_np

    #     log_likelihood = cp.sum(
    #         cp.multiply(y, X @ a) -
    #         cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ a]).T, axis=0,
    #                        keepdims=True).T
    #     )
    #     prob = cp.Problem(
    #         cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))
    #     fit_logreg = CvxpyLayer(prob, [X, lam], [a])

    #     with tf.GradientTape(persistent=True) as tape:
    #         weights = fit_logreg(X_tf, lam_tf, solver_args={'eps': 1e-8})[0]
    #         summed = jax.numpy.sum(weights)
    #     grad_X_tf, grad_lam_tf = tape.gradient(summed, [X_tf, lam_tf])

    #     def f_train():
    #         prob.solve(solver=cp.SCS, eps=1e-8)
    #         return np.sum(a.value)

    #     numgrad_X_tf, numgrad_lam_tf = numerical_grad(
    #         f_train, [X, lam], [X_tf, lam_tf], delta=1e-6)
    #     np.testing.assert_allclose(grad_X_tf, numgrad_X_tf, atol=1e-2)
    #     np.testing.assert_allclose(grad_lam_tf, numgrad_lam_tf, atol=1e-2)

    def test_not_enough_parameters(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, "The layer's parameters.*"):
            CvxpyLayer(prob, [lam], [x])

    def test_non_dpp(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, 'Problem must be DPP.'):
            CvxpyLayer(prob, [lam], [x, y])

    def test_too_many_variables(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, 'Argument `variables`.*'):
            CvxpyLayer(prob, [lam], [x, y])

    def test_infeasible(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        layer = CvxpyLayer(prob, [param], [x])
        param_tf = jax.numpy.ones(1)
        with self.assertRaises(diffcp.SolverError):
            layer(param_tf)

    # def test_lml(self):
    #     tf.random.set_seed(0)
    #     k = 2
    #     x = cp.Parameter(4)
    #     y = cp.Variable(4)
    #     obj = -x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
    #     cons = [cp.sum(y) == k]
    #     problem = cp.Problem(cp.Minimize(obj), cons)
    #     lml = CvxpyLayer(problem, [x], [y])
    #     x_tf = np.array([1., -1., -1., -1.], dtype=jax.numpy.float64)

    #     with tf.GradientTape() as tape:
    #         y_opt = lml(x_tf, solver_args={'eps': 1e-10})[0]
    #         loss = -jax.numpy.log(y_opt[1])

    #     def f():
    #         problem.solve(solver=cp.SCS, eps=1e-10)
    #         return -np.log(y.value[1])

    #     grad = tape.gradient(loss, [x_tf])
    #     numgrad = numerical_grad(f, [x], [x_tf])
    #     np.testing.assert_almost_equal(grad, numgrad, decimal=3)

    # def test_sdp(self):
    #     tf.random.set_seed(5)

    #     n = 3
    #     p = 3
    #     C = cp.Parameter((n, n))
    #     A = [cp.Parameter((n, n)) for _ in range(p)]
    #     b = [cp.Parameter((1, 1)) for _ in range(p)]

    #     C_tf = jax.numpy.array(jax.random.normal((n, n), dtype=jax.numpy.float64))
    #     A_tf = [jax.numpy.array(jax.random.normal((n, n), dtype=jax.numpy.float64))
    #             for _ in range(p)]
    #     b_tf = [jax.numpy.array(jax.random.normal((1, 1), dtype=jax.numpy.float64))
    #             for _ in range(p)]

    #     X = cp.Variable((n, n), symmetric=True)
    #     constraints = [X >> 0]
    #     constraints += [
    #         cp.trace(A[i]@X) == b[i] for i in range(p)
    #     ]
    #     problem = cp.Problem(cp.Minimize(
    #         cp.trace(C @ X) - cp.log_det(X) + cp.sum_squares(X)),
    #         constraints)
    #     layer = CvxpyLayer(problem, [C] + A + b, [X])
    #     values = [C_tf] + A_tf + b_tf
    #     with tf.GradientTape() as tape:
    #         soln = layer(*values,
    #                      solver_args={'eps': 1e-10, 'max_iters': 10000})[0]
    #         summed = jax.numpy.sum(soln)
    #     grads = tape.gradient(summed, values)

    #     def f():
    #         problem.solve(cp.SCS, eps=1e-10, max_iters=10000)
    #         return np.sum(X.value)

    #     numgrads = numerical_grad(f, [C] + A + b, values, delta=1e-4)
    #     for g, ng in zip(grads, numgrads):
    #         np.testing.assert_allclose(g, ng, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
