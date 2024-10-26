import warnings
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from void_migration import stress


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
        delta_limit = np.amax(p.delta_limit)
        L = np.ceil(p.nu_cs / delta_limit).astype(int)
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

    stable = mobilised_friction_angle < p.repose_angle

    if debug and np.random.rand() < 0.01:
        import matplotlib.pyplot as plt

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
    else:
        delta_limit = p.delta_limit

    if potential_free_surface is None:
        stable = delta_nu <= delta_limit
    else:
        stable = (delta_nu <= p.delta_limit) & potential_free_surface

    Stable = np.repeat(stable[:, :, np.newaxis], s.shape[2], axis=2)
    return Stable


def get_delta_limit(p, chi):
    if chi is None:
        chi = np.zeros([p.nx, p.ny])
    mu = p.mu * (1 - chi * 300)
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
