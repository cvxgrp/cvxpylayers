def dims_to_solver_dict(cone_dims):
    cones = {
        "f": int(cone_dims.zero),
        "l": int(cone_dims.nonpos),
        "q": [int(v) for v in cone_dims.soc],
        "ep": int(cone_dims.exp),
        "s": [int(v) for v in cone_dims.psd]
    }
    return cones
