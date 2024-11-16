import diffcp
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np
import time
from functools import partial
from cvxpylayers.utils import \
    ForwardContext, BackwardContext, forward_numpy, backward_numpy

try:
    import jax
except ImportError:
    raise ImportError("Unable to import jax. Please install from "
                      "https://github.com/google/jax")
from jax import core
import jax.numpy as jnp


def CvxpyLayer(problem, parameters, variables, gp=False):
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
        gp: Whether to parse the problem using DGP (True or False).

    Returns:
        A callable that solves the problem.
    """

    if gp:
        if not problem.is_dgp(dpp=True):
            raise ValueError('Problem must be DPP.')
    else:
        if not problem.is_dcp(dpp=True):
            raise ValueError('Problem must be DPP.')

    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match "
                         "problem.parameters")
    if not set(variables).issubset(set(problem.variables())):
        raise ValueError("Argument variables must be a subset of "
                         "problem.variables")
    if not isinstance(parameters, list) and \
            not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as "
                         "a list or tuple")
    if not isinstance(variables, list) and \
            not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as "
                         "a list or tuple")

    var_dict = {v.id for v in variables}

    # Construct compiler
    param_order = parameters
    if gp:
        for param in parameters:
            if param.value is None:
                raise ValueError("An initial value for each parameter is "
                                 "required when gp=True.")
        data, solving_chain, _ = problem.get_problem_data(
            solver=cp.SCS, gp=True, solver_opts={'use_quad_obj': False})
        compiler = data[cp.settings.PARAM_PROB]
        dgp2dcp = solving_chain.get(cp.reductions.Dgp2Dcp)
        param_ids = [p.id for p in compiler.parameters]
        old_params_to_new_params = (
            dgp2dcp.canon_methods._parameters
        )
    else:
        data, _, _ = problem.get_problem_data(
                solver=cp.SCS, solver_opts={'use_quad_obj': False})
        compiler = data[cp.settings.PARAM_PROB]
        param_ids = [p.id for p in param_order]
        dgp2dcp = None
    cone_dims = dims_to_solver_dict(data["dims"])

    info = {}
    CvxpyLayerFn_p = core.Primitive("CvxpyLayerFn_" + str(hash(problem)))

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def CvxpyLayerFn(solver_args, *params):
        return CvxpyLayerFn_p.bind(solver_args, *params)

    def CvxpyLayerFn_impl(solver_args, *params):
        """Solve problem (or a batch of problems) corresponding to `params`

        Args:
            solver_args: a dict of optional arguments, to send to `diffcp`.
                         Keys should be the names of keyword arguments.
            params: a sequence of JAX arrays; the n-th argument specifies
                    the value for the n-th CVXPY Parameter. These arrays
                    can be batched: if a array has 3 dimensions, then its
                    first dimension is interpreted as the batch size. These
                    arrays must all have the same dtype.

        Returns:
            a list of optimal variable values, one for each CVXPY Variable
            supplied to the constructor.
        """
        if len(params) != len(param_ids):
            raise ValueError('An array must be provided for each CVXPY '
                             'parameter; received %d arrays, expected %d' % (
                                 len(params), len(param_ids)))

        dtype, batch, batch_sizes, batch_size = batch_info(
            params, param_order)

        # convert to numpy arrays
        params_numpy = [np.array(param) for param in params]
            
        context = ForwardContext(
            gp=gp,
            solve_and_derivative=True,
            batch=batch,
            batch_size=batch_size,
            batch_sizes=batch_sizes,
            compiler=compiler,
            param_ids=param_ids,
            param_order=param_order if gp else None,
            old_params_to_new_params=old_params_to_new_params if gp else None,
            cone_dims=cone_dims,
            solver_args=solver_args,
            variables=variables,
            var_dict=var_dict
        )

        sol, info_forward = forward_numpy(params_numpy, context)
        
        # convert to jax arrays and store info
        sol = [jnp.array(s, dtype=dtype) for s in sol]
        info.update(info_forward)

        return tuple(sol)

    CvxpyLayerFn_p.def_impl(CvxpyLayerFn_impl)

    def CvxpyLayerFn_fwd_vjp(solver_args, *params):
        sol = CvxpyLayerFn(solver_args, *params)
        return sol, (params, sol)

    def CvxpyLayerFn_bwd_vjp(solver_args, res, dvars):
        params, sol = res
        dtype, batch, batch_sizes, batch_size = batch_info(
            params, param_order)

        # Use info here to retrieve this from the forward pass because
        # the residual in JAX's vjp doesn't allow non-JAX types to be
        # easily returned. This works when calling this serially,
        # but will break if this is called in parallel.

        # convert to numpy arrays
        dvars_numpy = [np.array(dvar) for dvar in dvars]
            
        context = BackwardContext(
            info=info,
            gp=gp,
            batch=batch,
            batch_size=batch_size,
            batch_sizes=batch_sizes,
            variables=variables,
            compiler=compiler,
            param_ids=param_ids,
            param_order=param_order if gp else None,
            params=params if gp else None,
            old_params_to_new_params=old_params_to_new_params if gp else None,
            sol=[np.array(s) for s in sol] if gp else None,
        )
            
        grad_numpy, info_backward = backward_numpy(dvars_numpy, context)
        
        # convert to jax arrays and store info
        grad = [jnp.array(g, dtype=dtype) for g in grad_numpy]
        info.update(info_backward)

        return tuple(grad)

    CvxpyLayerFn.defvjp(CvxpyLayerFn_fwd_vjp, CvxpyLayerFn_bwd_vjp)

    # Default solver_args to an optional empty dict
    def f(*params, **kwargs):
        solver_args = kwargs.get('solver_args', {})
        return CvxpyLayerFn(solver_args, *params)

    return f


def batch_info(params, param_order):
    # infer dtype and whether or not params are batched
    dtype = params[0].dtype

    batch_sizes = []
    for i, (p, q) in enumerate(zip(params, param_order)):
        # check dtype, device of params
        if p.dtype != dtype:
            raise ValueError(
                "Two or more parameters have different dtypes. "
                "Expected parameter %d to have dtype %s but "
                "got dtype %s." %
                (i, str(dtype), str(p.dtype))
            )

        # check and extract the batch size for the parameter
        # 0 means there is no batch dimension for this parameter
        # and we assume the batch dimension is non-zero
        if p.ndim == q.ndim:
            batch_size = 0
        elif p.ndim == q.ndim + 1:
            batch_size = p.shape[0]
            if batch_size == 0:
                raise ValueError(
                    "The batch dimension for parameter {} is zero "
                    "but should be non-zero.".format(i))
        else:
            raise ValueError(
                "Invalid parameter size passed in. Expected "
                "parameter {} to have have {} or {} dimensions "
                "but got {} dimensions".format(
                    i, q.ndim, q.ndim + 1, p.ndim))

        batch_sizes.append(batch_size)

        # validate the parameter shape
        p_shape = p.shape if batch_size == 0 else p.shape[1:]
        if not np.all(p_shape == param_order[i].shape):
            raise ValueError(
                "Inconsistent parameter shapes passed in. "
                "Expected parameter {} to have non-batched shape of "
                "{} but got {}.".format(
                        i,
                        q.shape,
                        p.shape))

    batch_sizes = np.array(batch_sizes)
    batch = np.any(batch_sizes > 0)

    if batch:
        nonzero_batch_sizes = batch_sizes[batch_sizes > 0]
        batch_size = nonzero_batch_sizes[0]
        if np.any(nonzero_batch_sizes != batch_size):
            raise ValueError(
                "Inconsistent batch sizes passed in. Expected "
                "parameters to have no batch size or all the same "
                "batch size but got sizes: {}.".format(
                    batch_sizes))
    else:
        batch_size = 1

    return dtype, batch, batch_sizes, batch_size
