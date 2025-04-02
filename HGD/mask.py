import numpy as np
import HGD.operators
from scipy.stats import truncnorm


def update(p, s):
    """
    Create the boundary mask.
    """

    mask = np.zeros((p.nx, p.ny), dtype=bool)
    mask_methods = [name for name, obj in globals().items() if callable(obj)]

    for method in p.masks:
        if method in mask_methods:
            mask_func = globals()[method]
            mask |= mask_func(p, s)  # bitwise OR
        else:
            raise ValueError(f"Mask method '{method}' not found.")

    p.boundary_mask = mask
    # set all masked values to NaN
    for k in range(p.nm):
        s[:, :, k][mask] = np.nan
    return p


def slope(p, s):
    # Create a mask for a slope that is p.cyclic_BC_y_offset high at on the left and 0 on the right
    x = np.arange(0, p.nx)
    y = np.arange(0, p.ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return Y < (-p.cyclic_BC_y_offset / p.nx * X + p.cyclic_BC_y_offset)
