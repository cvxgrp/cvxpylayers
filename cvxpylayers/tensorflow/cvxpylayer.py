from cvxpylayers import utils

import cvxpy as cp
import diffcp
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("Unable to import tensorflow. Please install "
                      "TensorFlow >= 2.0 (https://tensorflow.org).")

tf_major_version = int(tf.__version__.split('.')[0])
if tf_major_version < 2:
    raise ImportError("cvxpylayers requires TensorFlow >= 2.0; please "
                      "upgrade your installation of TensorFlow, which is "
                      "version %s." % tf.__version__)


class CvxpyLayer(object):
    """A differentiable convex optimization layer

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example usage

        ```
        import cvxpy as cp
        import tensorflow as tf
        from cvxpylayers.tensorflow import CvxpyLayer


        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tf = tf.Variable(tf.random.normal((m, n)))
        b_tf = tf.Variable(tf.random.normal((m,)))

        with tf.GradientTape() as tape:
          # solve the problem, setting the values of A and b to A_tf and b_tf
          solution, = cvxpylayer(A_tf, b_tf)
          summed_solution = tf.math.reduce_sum(solution)
        gradA, gradb = tape.gradient(summed_solution, [A_tf, b_tf])
        ```
    """

    def __init__(self, problem, parameters, variables):
        """Construct a CvxpyLayer

        Args:
          problem: The CVXPY problem; must be DPP.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
        """
        if not problem.is_dpp():
            raise ValueError('Problem must be DPP.')
        if set(parameters) != set(problem.parameters()):
            raise ValueError('Every parameter must be passed in argument '
                             '`parameters`')
        if not set(variables).issubset(set(problem.variables())):
            raise ValueError('Argument `variables` must be a subset of '
                             '`problem.variables()`')

        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        self.asa_maps = data[cp.settings.PARAM_PROB]
        self.cones = utils.dims_to_solver_dict(data['dims'])
        self.params = parameters
        self.param_ids = [p.id for p in self.params]
        self.vars = variables
        self.n_vars = len(self.vars)
        self.var_dict = {v.id: v for v in self.vars}

    def __call__(self, *parameters, solver_args={}):
        """Solve problem (or a batch of problems) corresponding to `parameters`

        Args:
          parameters: a sequence of tf.Tensors; the n-th Tensor specifies
                      the value for the n-th CVXPY Parameter. These Tensors
                      can be batched: if a Tensor has 3 dimensions, then its
                      first dimension is interpreted as the batch size.
          solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                       should be the names of keyword arguments.

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.
        """
        if len(parameters) != len(self.params):
            raise ValueError('A value must be provided for each CVXPY '
                             'parameter; received %d values, expected %d' % (
                                 len(parameters), len(self.params)))
        compute = tf.custom_gradient(
            lambda *parameters: self._compute(parameters, solver_args))
        return compute(*parameters)

    def _dx_from_dsoln(self, dsoln):
        dvars_numpy = list(map(lambda x: x.numpy(), dsoln))
        del_vars = {}
        for v, dv in zip(self.vars, dvars_numpy):
            del_vars[v.id] = dv
        return self.asa_maps.split_adjoint(del_vars)

    def _problem_data_from_params(self, params):
        c, _, A, b = self.asa_maps.apply_parameters(dict(zip(self.param_ids,
                                                             params)))
        A = -A
        return A, b, c

    def _restrict_DT_to_dx(self, DT, nbatch, s_shape):
        if nbatch > 0:
            zeros = [np.zeros(s_shape) for _ in range(nbatch)]
        else:
            zeros = np.zeros(s_shape)
        return lambda dxs: DT(dxs, zeros, zeros)

    def _split_solution(self, x):
        soln = self.asa_maps.split_solution(x, {v.id: v for v in self.vars})
        return tuple([tf.constant(soln[v.id]) for v in self.vars])

    def _compute(self, params, solver_args={}):
        params = [p.numpy() for p in params]
        nbatch = (0 if len(params[0].shape) == len(self.params[0].shape)
                  else params[0].shape[0])

        if nbatch > 0:
            split_params = [[np.squeeze(p) for p in np.split(param, nbatch)]
                            for param in params]
            params_per_problem = [
                [param_list[i] for param_list in split_params]
                for i in range(nbatch)]
            As, bs, cs = zip(*[
                self._problem_data_from_params(p) for p in params_per_problem])
            xs, _, ss, _, DT = diffcp.solve_and_derivative_batch(
                As=As, bs=bs, cs=cs, cone_dicts=[self.cones] * nbatch,
                **solver_args)
            DT = self._restrict_DT_to_dx(DT, nbatch, ss[0].shape)
            solns = [self._split_solution(x) for x in xs]
            # soln[i] is a tensor with first dimension equal to nbatch, holding
            # the optimal values for variable i
            solution = [
                tf.stack([s[i] for s in solns]) for i in range(self.n_vars)]
        else:
            A, b, c = self._problem_data_from_params(params)
            x, _, s, _, DT = diffcp.solve_and_derivative(
                A=A, b=b, c=c, cone_dict=self.cones, **solver_args)
            DT = self._restrict_DT_to_dx(DT, nbatch, s.shape)
            solution = self._split_solution(x)

        def gradient_function(*dsoln):
            if nbatch > 0:
                # split the batched dsoln tensors into lists, with one list
                # corresponding to each problem in the batch
                dsoln_lists = [[] for _ in range(nbatch)]
                for value in dsoln:
                    tensors = tf.split(value, nbatch)
                    for dsoln_list, t in zip(dsoln_lists, tensors):
                        dsoln_list.append(tf.squeeze(t))
                dxs = [self._dx_from_dsoln(dsoln_list)
                       for dsoln_list in dsoln_lists]
                dAs, dbs, dcs = DT(dxs)
                dparams_dict_unbatched = [
                    self.asa_maps.apply_param_jac(dc, -dA, db) for
                    (dA, db, dc) in zip(dAs, dbs, dcs)]
                dparams = []
                for p in self.params:
                    dparams.append(
                        tf.constant([d[p.id] for d in dparams_dict_unbatched]))
                return dparams
            else:
                dx = self._dx_from_dsoln(dsoln)
                dA, db, dc = DT(dx)
                dparams_dict = self.asa_maps.apply_param_jac(dc, -dA, db)
                return tuple(tf.constant(
                    dparams_dict[p.id]) for p in self.params)
        return solution, gradient_function
