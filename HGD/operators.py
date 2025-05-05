import warnings
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from HGD import stress


def swap(src, dst, arrays, nu, p):
    for n in range(len(arrays)):
        if arrays[n] is not None:
            arrays[n][*src], arrays[n][*dst] = arrays[n][*dst], arrays[n][*src]
    nu[src[0], src[1]] += 1.0 / p.nm
    nu[dst[0], dst[1]] -= 1.0 / p.nm
    return [arrays, nu]


def get_solid_fraction(s: np.ndarray, loc: list | None = None) -> float:
    """Calculate solid fraction of a single physical in a 3D array.

    Args:
        s: a 3D numpy array
        loc: Either None or a list of two integers.

    Returns:
        The fraction of the solid phase in s at (i, j) as a float.
    """
    # return np.mean(~np.isnan(s[i, j, :]))
    if loc is None:
        return 1.0 - np.mean(np.isnan(s), axis=2)
    else:
        return 1.0 - np.mean(np.isnan(s[loc[0], loc[1], :]))


def get_average(s, loc: list | None = None):
    """
    Calculate the mean size over the microstructural co-ordinate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if loc is None:
            s_bar = np.nanmean(s, axis=2)
        else:
            s_bar = np.nanmean(s[loc[0], loc[1], :])
    return s_bar


def get_hyperbolic_average(s: np.ndarray, loc: list | None = None) -> float:
    """
    Calculate the hyperbolic mean size over the microstructural co-ordinate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if loc is None:
            return 1.0 / np.nanmean(1.0 / s, axis=2)
        else:
            return 1.0 / np.nanmean(1.0 / s[loc[0], loc[1], :])


def get_depth(s):
    """
    Unused.
    """
    depth = np.mean(np.mean(~np.isnan(s), axis=2), axis=1)
    return depth


def empty_up(nu_here, p):
    # would this be faster with a convolution?
    # nu_up = np.roll(nu_here, -1, axis=1)
    # nu_up_left = np.roll(nu_up, -1, axis=0)
    # nu_up_right = np.roll(nu_up, 1, axis=0)
    # nu_left = np.roll(nu_here, -1, axis=0)
    # nu_right = np.roll(nu_here, 1, axis=0)
    # return (nu_up == 0.0) | (nu_up_left == 0.0) | (nu_up_right == 0.0) | (nu_left == 0.0) | (nu_right == 0.0)
    if p.mu == 0:
        return np.zeros_like(nu_here, dtype=bool)
    else:
        # HACK: RANDOMLY PICKED MAX AND MIN BELOW. NO IDEA WHAT THEY SHOULD BE. WILL BE REDUNDANT WITH THE STRESS BASED FAILURE CRITERION
        delta_limit_max = np.amax(p.delta_limit)
        # if p.nu_cs_mode == "constant":
        nu_cs_min = np.amin(p.nu_cs)
        L = np.ceil(nu_cs_min / delta_limit_max).astype(int)
        # else:
        # L = np.ceil(p.nu_1 / delta_limit).astype(int)
        # kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]).T
        kernel = np.ones((2 * L + 1, 2 * L + 1))
        kernel[L, L] = 0
        nu_min = minimum_filter(nu_here, footprint=kernel)

        # nu_min_dilated = maximu_filter(nu_min, size=3)

        return nu_min == 0  # & (nu_here > 0)


def stable_slope_stress(s, p, last_swap, debug=False):
    """
    Determines the stability of slopes based on the stress.

    Parameters:
    s (numpy.ndarray): A 3D array representing the solid fraction in the system.
    p (object): An object containing the parameter `delta_limit` which is used to determine stability.
    last_swap (numpy.ndarray): A 3D array representing the last swap in the system.

    Returns:
    numpy.ndarray: A 3D boolean array where `True` indicates stable slopes and `False` indicates unstable slopes.
    """

    sigma = stress.calculate_stress(s, last_swap, p)
    mobilised_friction_angle = stress.get_friction_angle(sigma, p, last_swap)

    stable = mobilised_friction_angle <= p.repose_angle

    if debug and np.random.rand() < 0.01:
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.figure(98)
        plt.ion()
        plt.subplot(211)
        plt.pcolormesh(p.x, p.y, mobilised_friction_angle.T)
        plt.colorbar()
        plt.subplot(212)
        plt.pcolormesh(p.x, p.y, stable.T)
        plt.colorbar()
        plt.pause(1e-5)

    Stable = np.repeat(stable[:, :, np.newaxis], s.shape[2], axis=2)
    return Stable


def stable_slope_gradient(s, dir, p, debug=False):
    """
    Determines the stability of slopes based on the solid fraction.

    Parameters:
    s (numpy.ndarray): A 3D array representing the solid fraction in the system.
    dir (int): The direction in which to roll the array (shift axis).
    p (object): An object containing the parameter `delta_limit` which is used to determine stability.

    Returns:
    numpy.ndarray: A 3D boolean array where `True` indicates stable slopes and `False` indicates unstable slopes.
    """

    nu = get_solid_fraction(s)
    dnu_dx, dnu_dy = np.gradient(nu)
    nu_mag = np.sqrt(dnu_dx**2 + dnu_dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.nan_to_num(dnu_dy / dnu_dx, posinf=0, neginf=0)
    stable = (np.abs(slope) < p.mu) & (nu_mag > 0.2)

    if debug:
        import matplotlib.pyplot as plt

        plt.ion()
        plt.figure(98)
        plt.subplot(2, 2, 1)
        plt.imshow(nu_mag)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(slope)
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.imshow(stable)
        plt.colorbar()
        plt.pause(1e-1)

    Stable = np.repeat(stable[:, :, np.newaxis], s.shape[2], axis=2)
    return Stable


def stable_slope_fast(s, d, p, chi=None, potential_free_surface=None):
    """
    Determines the stability of slopes based on the solid fraction.

    Parameters:
    s (numpy.ndarray): A 3D array representing the solid fraction in the system.
    d (int): The direction in which to roll the array (shift axis).
    p (object): An object containing the parameter `delta_limit` which is used to determine stability.

    Returns:
    numpy.ndarray: A 3D boolean array where `True` indicates stable slopes and `False` indicates unstable slopes.
    """

    nu_here = get_solid_fraction(s)
    nu_dest = np.roll(nu_here, d, axis=0)
    delta_nu = nu_dest - nu_here

    # delta_nu = -dir*np.gradient(nu_here,axis=0)
    if p.inertia:
        get_delta_limit(p, chi)
    # else:
    # delta_limit = p.delta_limit

    if potential_free_surface is None:
        stable = delta_nu <= p.delta_limit
    else:
        stable = (delta_nu <= p.delta_limit) & potential_free_surface

    Stable = np.repeat(stable[:, :, np.newaxis], s.shape[2], axis=2)
    return Stable


def get_delta_limit(p, chi):
    if chi is None:
        chi = np.zeros([p.nx, p.ny])
    mu = p.mu * (1 - chi**0.2)  # HACK - extra factor is a magic number
    mu[mu < 0] = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_mu = np.nan_to_num(1.0 / mu, nan=0.0, posinf=1e30, neginf=0.0)
    p.delta_limit = p.nu_cs / (inv_mu + 1)


def stable_slope(s, i, j, dest, p):
    """
    Determines if the slope between two points is stable based on the solid fraction difference.

    Parameters:
    - s (object): The simulation state, containing the grid and relevant data.
    - i (int): The current row index in the grid.
    - j (int): The current column index in the grid.
    - dest (int): The destination row index for comparison.
    - p (object): An object containing simulation parameters, including the delta_limit.

    Returns:
    - bool: True if the difference in solid fraction between the current point and the destination
        point is less than or equal to the delta_limit, indicating a stable slope. False otherwise.

    This function calculates the solid fraction at the current point (i, j) and at a destination point
    (dest, j), then compares the difference in solid fraction to a threshold (delta_limit) defined in
    the parameter object p. If the difference is less than or equal to the threshold, the function
    returns True, indicating the slope is stable. Otherwise, it returns False.
    """
    nu_here = get_solid_fraction(s, [i, j])
    nu_dest = get_solid_fraction(s, [dest, j])
    delta_nu = nu_dest - nu_here

    return delta_nu <= p.delta_limit


def locally_solid(s, i, j, p):
    """
    Determines if a given point in the simulation grid is locally solid based on the solid fraction threshold.

    Parameters:
    - s (object): The simulation state, containing the grid and relevant data.
    - i (int): The row index of the point in the grid.
    - j (int): The column index of the point in the grid.
    - p (object): An object containing simulation parameters, including the critical solid fraction threshold (nu_cs).

    Returns:
    - bool: True if the solid fraction at the given point is greater than or equal to the critical solid fraction threshold (nu_cs), indicating the point is locally solid. False otherwise.

    This function calculates the solid fraction at the specified point (i, j) in the simulation grid. It then compares this value to the critical solid fraction threshold (nu_cs) defined in the parameter object p. If the solid fraction at the point is greater than or equal to nu_cs, the function returns True, indicating the point is considered locally solid. Otherwise, it returns False.
    """
    nu = get_solid_fraction(s, [i, j])
    if isinstance(p.nu_cs, np.ndarray):
        return nu >= p.nu_cs[i, j]
    else:
        return nu >= p.nu_cs


def empty_nearby(nu, p):
    """
    Identifies empty spaces adjacent to each point in a grid based on a given solid fraction matrix.

    Parameters:
    - nu (numpy.ndarray): A 2D array representing the solid fraction at each point in the grid.
    - p (object): An object containing simulation parameters, not used in this function but included for consistency with the interface.

    Returns:
    - numpy.ndarray: A boolean array where True indicates an empty space adjacent to the corresponding point in the input grid.

    This function applies a maximum filter with a cross-shaped kernel to the solid fraction matrix 'nu'. The kernel is defined to consider the four cardinal directions (up, down, left, right) adjacent to each point. The maximum filter operation identifies the maximum solid fraction value in the neighborhood defined by the kernel for each point. Points where the maximum solid fraction in their neighborhood is 0 are considered adjacent to an empty space, and the function returns a boolean array marking these points as True.
    """
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    nu_max = maximum_filter(nu, footprint=kernel)  # , mode='constant', cval=0.0)

    return nu_max == 0


def get_top(nu, p, nu_lim=0):
    """
    Determine the top void in each column of a 2D array.

    Parameters:
    nu (ndarray): A 2D numpy array where each element indicates the presence (0) or absence (non-zero) of a void.
    p (object): An object with attributes `nx` (number of columns) and `ny` (number of rows).

    Returns:
    ndarray: A 1D numpy array of integers where each element represents the row index of the top void in the corresponding column.
             If a column has no voids, the value will be `p.ny - 1`.
    """
    solid = nu > nu_lim
    # now get top void in each column
    top = np.zeros(p.nx)
    for i in range(p.nx):
        for j in range(p.ny - 1, 0, -1):
            if solid[i, j]:
                top[i] = j
                break
    # print(f"Top voids: {top}")
    return top.astype(int)


def get_depth(nu, p, debug=False):
    """
    Calculate the depth array based on the top indices and y-coordinates.

    Parameters:
    nu : array-like
        An array or list of values used to determine the top indices.
    p : object
        An object containing the attributes 'nx', 'ny', and 'y'. 'nx' and 'ny' are the dimensions
        of the grid, and 'y' is an array of y-coordinates.

    Returns:
    numpy.ndarray
        A 2D array of shape (p.nx, p.ny) representing the depth values.
    """
    top = get_top(nu, p)
    depth = np.zeros([p.nx, p.ny])

    for i in range(p.nx):
        depth[i, :] = p.y[top[i]] - p.y

    if debug:
        import matplotlib.pyplot as plt

        plt.figure(77)
        plt.ion()
        plt.clf()
        plt.pcolormesh(p.x, p.y, depth.T)
        plt.colorbar()
        plt.pause(0.01)

    return depth


def get_lr(i, j, p):
    if i == 0:
        if p.cyclic_BC:
            l = p.nx - 1
        else:
            l = 0
        r = 1
    elif i == p.nx - 1:
        if p.cyclic_BC:
            r = 0
        else:
            r = p.nx - 1
        l = p.nx - 2
    else:
        l = i - 1
        r = i + 1

    # Cyclic BC in y direction
    if p.cyclic_BC_y_offset > 0:
        if i == p.nx - 1:
            j_r = j + p.cyclic_BC_y_offset
            if j_r >= p.ny - 1:
                j_r = p.ny - 1
        else:
            j_r = j
        if i == 0:
            j_l = j - p.cyclic_BC_y_offset
            if j_l < 0:
                j_l = 0
        else:
            j_l = j
    else:
        j_r = j
        j_l = j

    return l, r, j_l, j_r


# def stream(u, v, s, p):
#     s_bar = np.nanmean(s)  # global mean value
#     v_y = np.sqrt(p.g * s_bar)

#     # QUESTION TO ANSWER:
#     # 1. Is this just from advection?
#     # 2. If not, I guess we need to consider not just the expected velocity (derived from the probability) but what actually moved (otherwise P_lr is always symmetric and so no velocity). Can be done by multiplying by the solid fraction!

#     nu = get_solid_fraction(s)
#     n = 1 - nu
#     # unstable = nu < p.nu_cs

#     u_old = np.nanmean(u, axis=2)
#     v_old = np.nanmean(v, axis=2)

#     P_lr = p.alpha * v_y * s_bar * (p.dt / p.dy / p.dy)
#     v_lr_eff = P_lr * p.dy / p.dt  # magic!
#     # P_u = v_y * p.dt / p.dy

#     nu_l = np.roll(nu, -1, axis=1)
#     nu_r = np.roll(nu, 1, axis=1)
#     nu_u = np.roll(nu, -1, axis=0)
#     u_l_new = v_lr_eff * (1 - nu_l) + np.where(u_old < 0, -u_old, 0)
#     u_r_new = v_lr_eff * (1 - nu_r) + np.where(u_old > 0, u_old, 0)
#     u_new = u_l_new + u_r_new
#     v_new = v_y * (1 - nu_u) + v_old

#     N_u = np.floor(p.nm * v_new * p.dt / p.dy).astype(int)
#     N_l = np.floor(p.nm * u_l_new * p.dt / p.dx).astype(int)
#     N_r = np.floor(p.nm * u_r_new * p.dt / p.dx).astype(int)

#     for i in range(p.nx):
#         for j in range(p.ny):
#             l, r, j_l, j_r = get_lr(i, j, p)
#             dests = [[i, j - 1], [l, j], [r, j]]  # up, left, right
#             Ns = np.array([N_u[i, j], N_l[l, j_l], N_r[r, j_r]])
#             print(Ns, end=" ")
#             solid_indices = np.argwhere(~np.isnan(s[i, j, :]))
#             np.random.shuffle(solid_indices)
#             if Ns.sum() > 0:
#                 for k in range(len(solid_indices)):  # try each just once?
#                     d = np.random.choice([0, 1, 2], p=Ns / Ns.sum())
#                     dest = dests[d]
#                     if np.isnan(s[i, j, solid_indices[k]]):
#                         s[dest[0], dest[1], solid_indices[k]] = s[i, j, solid_indices[k]]
#                         s[i, j, solid_indices[k]] = np.nan
#                     Ns[d] -= 1
#                     if Ns.sum() == 0:
#                         break
#             print(Ns.sum())

#     u = np.repeat(u_new[:, :, np.newaxis], p.nm, axis=2)
#     v = np.repeat(v_new[:, :, np.newaxis], p.nm, axis=2)

#     return u, v, s


def stream(u, v, s, p):
    ### FROM THE PERPECTIVE OF THE SOLID PHASE!!!!!!
    s_bar = np.nanmean(s)  # global mean value
    v_y = np.sqrt(p.g * s_bar)

    nu = get_solid_fraction(s)
    nu_u = np.roll(nu, 1, axis=1)
    nu_l = np.roll(nu, 1, axis=0)
    nu_r = np.roll(nu, -1, axis=0)

    # VERTICAL
    v_old = np.nanmean(v, axis=2)
    v_new = v_y * (1 - nu_u / p.nu_cs) + v_old  # HACK: NOT SURE ABOUT THE DIVISION BY NU_CS

    # HORIZONTAL
    P_lr = p.alpha * v_y * s_bar * (p.dt / p.dy / p.dy)
    u_old = np.nanmean(u, axis=2)
    u_new_l = P_lr * (1 - nu_l / p.nu_cs) * p.dy / p.dt + np.where(
        u_old < 0, -u_old, 0
    )  # magic! --- MIGHT BE WRONG WAY AROUND L/R
    u_new_r = P_lr * (1 - nu_r / p.nu_cs) * p.dy / p.dt + np.where(u_old > 0, u_old, 0)  # magic!

    u_new = -u_new_l + u_new_r

    N_l = np.floor(p.nm * u_new_l * p.dt / p.dx).astype(int)
    N_r = np.floor(p.nm * u_new_r * p.dt / p.dx).astype(int)
    N_u = np.floor(p.nm * v_new * p.dt / p.dy).astype(int)

    # stable_left = stable_slope(s, 1, p)[:, :, 0]
    # stable_right = stable_slope_fast(s, -1, p)[:, :, 0]

    for i in range(0, p.nx - 1):
        for j in range(1, p.ny):
            if nu[i, j] < p.nu_cs:
                solid_here = ~np.isnan(s[i, j, :])
                dests = [[i, j - 1], [i - 1, j], [i + 1, j]]  # up, left, right
                Ns = [N_u[i, j], N_l[i, j], N_r[i, j]]
                # stable_left = stable_slope(s, i, j, i - 1, p)
                # stable_right = stable_slope(s, i, j, i + 1, p)
                # stable_slopes = [False, stable_left, stable_right]
                # print(stable_slopes)
                for d in range(3):
                    # if not stable_slopes[d]:
                    void_dest = np.isnan(s[dests[d][0], dests[d][1], :])
                    potential_swaps = np.logical_and(solid_here, void_dest)
                    potential_swap_indices = np.argwhere(potential_swaps).flatten()
                    num_available_dest_voids = int(
                        np.floor(p.nm * (1 - nu[dests[d][0], dests[d][1]] - p.nu_cs))
                    )
                    max_swaps = np.minimum(Ns[d], len(potential_swap_indices))
                    max_swaps = np.minimum(max_swaps, num_available_dest_voids)
                    if max_swaps > 0:
                        swaps = np.random.choice(potential_swap_indices, size=max_swaps, replace=False)
                        for k in range(max_swaps):
                            s[dests[d][0], dests[d][1], swaps[k]] = s[i, j, swaps[k]]
                            s[i, j, swaps[k]] = np.nan

    # HACK: this is not correct, but it works for now
    v_new[nu >= p.nu_cs] = 0
    v_new[nu == 0] = 0
    v = np.repeat(v_new[:, :, np.newaxis], p.nm, axis=2)

    u_new[nu >= p.nu_cs] = 0
    u_new[nu == 0] = 0
    u = np.repeat(u_new[:, :, np.newaxis], p.nm, axis=2)
    return u, v, s
