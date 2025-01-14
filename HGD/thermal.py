import numpy as np


def update_temperature(s, T, p):
    """
    Used for modelling the diffusion of heat into the body. Still not functional. Do not use.
    """
    T[np.isnan(s)] = p.temperature["inlet_temperature"]  # HACK
    T[p.boundary] = p.temperature["boundary_temperature"]
    T_inc = np.zeros_like(T)
    T_inc[1:-1, 1:-1] = 1e-3 * (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4 * T[1:-1, 1:-1])
    return T + T_inc
