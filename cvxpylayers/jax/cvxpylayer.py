import diffcp
import time
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

try:
    import jax
    from jax import core
    from jax import ad
except ImportError:
    raise ImportError("Unable to import jax/jaxlib. Please follow instructions "
                      "at https://github.com/google/jax.")

def construct_cvxpylayer(problem, parameters, variables, solver_args={}):
    """Construct a differentiable convex optimization layer.

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example Usage:
    ```
    XXX
    ```

    Args:
        problem: The CVXPY problem; must be DPP.
        parameters: A list of CVXPY Parameters in the problem; the order
                    of the Parameters determines the order in which parameter
                    values must be supplied in the forward pass. Must include
                    every parameter involved in problem.
        variables: A list of CVXPY Variables in the problem; the order of the
                    Variables determines the order of the optimal variable
                    values returned from the forward pass.
        solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                    should be the names of keyword arguments

    """
    if not problem.is_dpp():
        raise ValueError('Problem must be DPP.')
    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match "
                            "problem.parameters")
    if not set(variables).issubset(set(problem.variables())):
        raise ValueError("Argument variables must be a subset of "
                            "problem.variables")

    param_order = parameters
    param_ids = [p.id for p in param_order]
    variables = variables
    var_dict = {v.id for v in variables}
    solver_args = solver_args

    # Construct compiler
    data, _, _ = problem.get_problem_data(solver=cp.SCS)
    compiler = data[cp.settings.PARAM_PROB]
    cone_dims = dims_to_solver_dict(data["dims"])

    layer_p = core.Primitive("layer")
    layer_vjp_p = core.Primitive("layer_vjp")
    layer = lambda params: layer_p.bind(params)
    layer_vjp = lambda params: layer_vjp_p.bind(params)

    def layer_impl(params):
        """Solve problem (or a batch of problems) corresponding to `params`.

        Args:
            params: a sequence of numpy arrays; the nth array specifies
                    the value for the nth CVXPY Parameter. 
        """
        if len(params) != len(param_ids):
            raise ValueError('An array must be provided for each CVXPY '
                            'parameter; received %d arrays, expected %d' % (
                                len(params), len(param_ids)))

        if len(params) != len(param_ids):
            raise ValueError('An array must be provided for each CVXPY '
                            'parameter; received %d arrays, expected %d' % (
                                len(params), len(param_ids)))

        c, _, neg_A, b = compiler.apply_parameters(dict(
            zip(param_ids, params)), keep_zeros=True
        )
        A = -neg_A

        xs, _, _, _, DT = diffcp.solve_and_derivative(A, b, c, cone_dims, **solver_args)

        sltn_dict = compiler.split_solution(xs, active_vars=var_dict)
        return [sltn_dict[v.id] for v in variables]
    
    layer_p.def_impl(layer_impl)

    def layer_vjp_impl(params):
        if len(params) != len(param_ids):
            raise ValueError('An array must be provided for each CVXPY '
                            'parameter; received %d arrays, expected %d' % (
                                len(params), len(param_ids)))

        if len(params) != len(param_ids):
            raise ValueError('An array must be provided for each CVXPY '
                            'parameter; received %d arrays, expected %d' % (
                                len(params), len(param_ids)))

        c, _, neg_A, b = compiler.apply_parameters(dict(
            zip(param_ids, params)), keep_zeros=True
        )
        A = -neg_A
        m, n = A.shape

        xs, _, _, _, DT = diffcp.solve_and_derivative(A, b, c, cone_dims, **solver_args)

        sltn_dict = compiler.split_solution(xs, active_vars=var_dict)
        sol = [sltn_dict[v.id] for v in variables]

        def vjp(dvars):
            del_vars = {}
            for v, dv in zip(variables, dvars):
                del_vars[v.id] = dv
            dx = compiler.split_adjoint(del_vars)
            dy = np.zeros(m)
            ds = np.zeros(n)
        
            dA, db, dc = DT(dx, dy, ds)
            del_param_dict = compiler.apply_param_jac(dc, -dA, db)
            return [del_param_dict[p.id] for p in parameters]

        return sol, vjp

    def _layer_vjp(params):
        return layer_vjp(params)
    
    layer_vjp_p.def_impl(layer_vjp_impl)
    ad.defjvp(layer_vjp_p, lambda x, g: [np.zeros(v.shape) for v in variables])

    ad.defvjp_all(layer_p, layer_vjp)

    return layer