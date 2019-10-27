import diffcp
import time
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("Unable to import torch. Please install at "
                      "https://pytorch.org.")

torch_major_version = int(torch.__version__.split('.')[0])
if torch_major_version < 1:
    raise ImportError("cvxpylayers requires PyTorch >= 1.0; please "
                      "upgrade your installation of PyTorch, which is "
                      "version %s." % torch.__version__)


class CvxpyLayer(torch.nn.Module):
    """A differentiable convex optimization layer

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example usage:
        ```
        import cvxpy as cp
        import torch
        from cvxpylayers.torch import CvxpyLayer

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        solution, = cvxpylayer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()
        ```
    """

    def __init__(self, problem, parameters, variables):
        """Construct a CvxpyLayer

        Args:
          problem: The CVXPY problem; must be DPP.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass. Must include
                      every parameter involved in problem.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
        """
        if not problem.is_dpp():
            raise ValueError('Problem must be DPP.')

        super(CvxpyLayer, self).__init__()

        assert set(problem.parameters()) == set(parameters), \
            "Every parameter must be passed to argument parameters"
        assert set(variables).issubset(set(problem.variables())), \
            "Argument variables must be a subset of problem.variables"
        assert hasattr(problem, "get_problem_data"), \
            "cvxpy problem does not support ASA form; please upgrade cvxpy"

        self.param_order = parameters
        self.param_ids = [p.id for p in self.param_order]
        self.variables = variables
        self.var_dict = {v.id for v in self.variables}

        # Construct compiler
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        self.compiler = data[cp.settings.PARAM_PROB]
        self.cone_dims = dims_to_solver_dict(data["dims"])

    def forward(self, *params, solver_args={}):
        """Solve problem (or a batch of problems) corresponding to `params`

        Args:
          params: a sequence of torch Tensors; the n-th Tensor specifies
                  the value for the n-th CVXPY Parameter. These Tensors
                  can be batched: if a Tensor has 3 dimensions, then its
                  first dimension is interpreted as the batch size.
          solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                       should be the names of keyword arguments.

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.
        """
        info = {}
        f = _CvxpyLayerFn(
            param_order=self.param_order,
            param_ids=self.param_ids,
            variables=self.variables,
            var_dict=self.var_dict,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
            solver_args=solver_args,
            info=info,
        )
        sol = f(*params)
        self.info = info
        return sol


def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array to torch tensor
    return torch.from_numpy(x).type(dtype).to(device)


def _CvxpyLayerFn(
        param_order,
        param_ids,
        variables,
        var_dict,
        compiler,
        cone_dims,
        solver_args,
        info):
    class _CvxpyLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            params_numpy = [to_numpy(p) for p in params]

            # infer dtype, device, and whether or not params are batched
            param_0_shape = params[0].shape
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device
            ctx.batch = len(param_0_shape) > len(param_order[0].shape)

            # check shapes of params
            if ctx.batch:
                ctx.batch_size = param_0_shape[0]
                for i, p in enumerate(params_numpy):
                    if len(p.shape) != len(param_order[i].shape) + 1:
                        raise RuntimeError(
                            "Inconsistent batch size passed in. Expected "
                            "parameter {} to have batch shape of {} dims "
                            "but got {} dims.".format(
                                i, 1 + len(
                                    param_order[i].shape), len(
                                    p.shape)))

                    if not np.all(p.shape[1:] == param_order[i].shape):
                        raise RuntimeError(
                            "Inconsistent parameter shapes passed in. "
                            "Expected parameter {} to have shape of {} "
                            "but got {}.".format(
                                 i,
                                 param_order[i].shape,
                                 p.shape[1:]))

                    if p.shape[0] != ctx.batch_size:
                        raise RuntimeError(
                            "Inconsistent batch size passed in. Expected "
                            "parameter {} to have batch size {} but "
                            "got {}.".format(
                                i, ctx.batch_size, p.shape[0]))
            else:
                ctx.batch_size = 1
                for i, p in enumerate(params_numpy):
                    if len(p.shape) != len(param_order[i].shape):
                        raise RuntimeError(
                            "Inconsistent batch size passed in. Expected "
                            "parameter {} to have batch shape of {} "
                            "dims but got {} dims.".format(
                                i, len(
                                    param_order[i].shape), len(
                                    p.shape)))

                    if not np.all(p.shape == param_order[i].shape):
                        raise RuntimeError(
                            "Inconsistent parameter shapes passed in. "
                            "Expected parameter {} to have shape of {} "
                            "but got {}.".format(
                                 i,
                                 param_order[i].shape,
                                 p.shape))

                    params_numpy[i] = np.expand_dims(p, 0)

            # canonicalize problem
            start = time.time()
            As, bs, cs, cone_dicts, ctx.shapes = [], [], [], [], []
            for i in range(ctx.batch_size):
                c, _, neg_A, b = compiler.apply_parameters(
                    dict(zip(param_ids, [p[i] for p in params_numpy])))
                A = -neg_A  # cvxpy canonicalizes -A
                As.append(A)
                bs.append(b)
                cs.append(c)
                cone_dicts.append(cone_dims)
                ctx.shapes.append(A.shape)
            info['canon_time'] = time.time() - start

            # compute solution and derivative function
            start = time.time()
            try:
                xs, _, _, _, ctx.DT_batch = diffcp.solve_and_derivative_batch(
                    As, bs, cs, cone_dicts, **solver_args)
            except diffcp.SolverError as e:
                print(
                    "Please consider re-formulating your problem so that "
                    "it is always solvable.")
                raise e
            info['solve_time'] = time.time() - start

            # extract solutions and append along batch dimension
            sol = [[] for _ in range(len(variables))]
            for i in range(ctx.batch_size):
                sltn_dict = compiler.split_solution(
                    xs[i], active_vars=var_dict)
                for j, v in enumerate(variables):
                    sol[j].append(to_torch(
                        sltn_dict[v.id], ctx.dtype, ctx.device).unsqueeze(0))
            sol = [torch.cat(s, 0) for s in sol]

            if not ctx.batch:
                sol = [s.squeeze(0) for s in sol]

            return tuple(sol)

        @staticmethod
        def backward(ctx, *dvars):
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]

            if not ctx.batch:
                dvars_numpy = [np.expand_dims(dvar, 0) for dvar in dvars_numpy]

            # differentiate from cvxpy variables to cone problem data
            dxs, dys, dss = [], [], []
            for i in range(ctx.batch_size):
                del_vars = {}
                for v, dv in zip(variables, [dv[i] for dv in dvars_numpy]):
                    del_vars[v.id] = dv
                dxs.append(compiler.split_adjoint(del_vars))
                dys.append(np.zeros(ctx.shapes[i][0]))
                dss.append(np.zeros(ctx.shapes[i][0]))

            dAs, dbs, dcs = ctx.DT_batch(dxs, dys, dss)

            # differentiate from cone problem data to cvxpy parameters
            start = time.time()
            grad = [[] for _ in range(len(param_order))]
            for i in range(ctx.batch_size):
                del_param_dict = compiler.apply_param_jac(
                    dcs[i], -dAs[i], dbs[i])
                for j, p in enumerate(param_order):
                    grad[j] += [to_torch(del_param_dict[p.id],
                                         ctx.dtype, ctx.device).unsqueeze(0)]
            grad = [torch.cat(g, 0) for g in grad]
            info['dcanon_time'] = time.time() - start

            if not ctx.batch:
                grad = [g.squeeze(0) for g in grad]

            return tuple(grad)

    return _CvxpyLayerFnFn.apply
