import diffcp
import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict
import numpy as np
import time
from functools import partial

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
            solver=cp.SCS, gp=True)
        compiler = data[cp.settings.PARAM_PROB]
        dgp2dcp = solving_chain.get(cp.reductions.Dgp2Dcp)
        param_ids = [p.id for p in compiler.parameters]
        old_params_to_new_params = (
            dgp2dcp.canon_methods._parameters
        )
    else:
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
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

        if gp:
            param_map = {}
            # construct a list of params for the DCP problem
            for param, value in zip(param_order, params):
                if param in old_params_to_new_params:
                    new_id = old_params_to_new_params[param].id
                    param_map[new_id] = jnp.log(value)
                else:
                    new_id = param.id
                    param_map[new_id] = value
            params_numpy = [np.array(param_map[pid]) for pid in param_ids]
        else:
            params_numpy = [np.array(p) for p in params]

        # canonicalize problem
        start = time.time()
        As, bs, cs, cone_dicts, shapes = [], [], [], [], []
        for i in range(batch_size):
            params_numpy_i = [
                p if sz == 0 else p[i]
                for p, sz in zip(params_numpy, batch_sizes)]
            c, _, neg_A, b = compiler.apply_parameters(
                dict(zip(param_ids, params_numpy_i)),
                keep_zeros=True)
            A = -neg_A  # cvxpy canonicalizes -A
            As.append(A)
            bs.append(b)
            cs.append(c)
            cone_dicts.append(cone_dims)
            shapes.append(A.shape)
        info['canon_time'] = time.time() - start
        info['shapes'] = shapes

        # compute solution and derivative function
        start = time.time()
        try:
            xs, _, _, _, DT_batch = diffcp.solve_and_derivative_batch(
                As, bs, cs, cone_dicts, **solver_args)
            info['DT_batch'] = DT_batch
        except diffcp.SolverError as e:
            print(
                "Please consider re-formulating your problem so that "
                "it is always solvable or increasing the number of "
                "solver iterations.")
            raise e
        info['solve_time'] = time.time() - start

        # extract solutions and append along batch dimension
        start = time.time()
        sol = [[] for i in range(len(variables))]
        for i in range(batch_size):
            sltn_dict = compiler.split_solution(
                xs[i], active_vars=var_dict)
            for j, v in enumerate(variables):
                sol[j].append(jnp.expand_dims(jnp.array(
                    sltn_dict[v.id], dtype=dtype), axis=0))
        sol = [jnp.concatenate(s, axis=0) for s in sol]

        if not batch:
            sol = [jnp.squeeze(s, axis=0) for s in sol]

        if gp:
            sol = [jnp.exp(s) for s in sol]

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
        shapes = info['shapes']
        DT_batch = info['DT_batch']

        if gp:
            # derivative of exponential recovery transformation
            dvars = [dvar*s for dvar, s in zip(dvars, sol)]

        dvars_numpy = [np.array(dvar) for dvar in dvars]

        if not batch:
            dvars_numpy = [np.expand_dims(dvar, 0) for dvar in dvars_numpy]

        # differentiate from cvxpy variables to cone problem data
        dxs, dys, dss = [], [], []
        for i in range(batch_size):
            del_vars = {}
            for v, dv in zip(variables, [dv[i] for dv in dvars_numpy]):
                del_vars[v.id] = dv
            dxs.append(compiler.split_adjoint(del_vars))
            dys.append(np.zeros(shapes[i][0]))
            dss.append(np.zeros(shapes[i][0]))

        dAs, dbs, dcs = DT_batch(dxs, dys, dss)

        # differentiate from cone problem data to cvxpy parameters
        start = time.time()
        grad = [[] for _ in range(len(param_ids))]
        for i in range(batch_size):
            del_param_dict = compiler.apply_param_jac(
                dcs[i], -dAs[i], dbs[i])
            for j, pid in enumerate(param_ids):
                grad[j] += [jnp.expand_dims(jnp.array(
                    del_param_dict[pid], dtype=dtype), axis=0)]
        grad = [jnp.concatenate(g, axis=0) for g in grad]
        if gp:
            # differentiate through the log transformation of params
            dcp_grad = grad
            grad = []
            dparams = {pid: g for pid, g in zip(param_ids, dcp_grad)}
            for param, value in zip(param_order, params):
                v = 0.0 if param.id not in dparams else dparams[param.id]
                if param in old_params_to_new_params:
                    dcp_param_id = old_params_to_new_params[param].id
                    # new_param.value == log(param), apply chain rule
                    v += (1.0 / value) * dparams[dcp_param_id]
                grad.append(v)
        info['dcanon_time'] = time.time() - start

        if not batch:
            grad = [jnp.squeeze(g, axis=0) for g in grad]
        else:
            for i, sz in enumerate(batch_sizes):
                if sz == 0:
                    grad[i] = jnp.sum(grad[i], axis=0)

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
