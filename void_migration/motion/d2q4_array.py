import numpy as np
from numpy.typing import ArrayLike
from void_migration import operators
import stress


# @njit
def move_voids(
    u: ArrayLike,
    v: ArrayLike,
    s: ArrayLike,
    p,
    diag: int = 0,
    c: None | ArrayLike = None,
    T: None | ArrayLike = None,
    chi: None | ArrayLike = None,
    last_swap: None | ArrayLike = None,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, None | ArrayLike, None | ArrayLike]:
    """
    Function to move voids each timestep.

    Args:
        u: Storage container for counting how many voids moved horizontally
        v: Storage container for counting how many voids moved vertically
        s: 3D array containing the local sizes everywhere. `NaN`s represent voids. Other values represent the grain size. The first two dimensions represent real space, the third dimension represents the micro-structural coordinate.
        diag: Should the voids swap horizontally (von neumnann neighbourhood, `diag=0`) or diagonally upwards (moore neighbourhood, `diag=1`). Default value `0`.
        c: If ArrayLike, a storage container for tracking motion of differently labelled particles. If `None`, do nothing.
        T: If ArrayLike, the temperature field. If `None`, do nothing.
        boundary: If ArrayLike, a descriptor of cells which voids cannot move into (i.e. boundaries). If `internal_boundary` is defined in the params file, allow for reduced movement rates rather than zero. If `None`, do nothing.

    Returns:
        u: The updated horizontal velocity
        v: The updated vertical velocity
        s: The new locations of the grains
        c: The updated concentration field
        T: The updated temperature field
    """
    options = np.array([(1, -1), (0, -1), (0, 1)])  # up, left, right
    np.random.shuffle(options)  # oh boy, this is a massive hack
    N_swap = np.zeros([p.nx, p.ny], dtype=int)

    for axis, d in options:
        nu = operators.get_solid_fraction(s)

        solid = nu >= p.nu_cs
        Solid = np.repeat(solid[:, :, np.newaxis], p.nm, axis=2)

        unstable = np.isnan(s) * ~Solid  # & ~Skip

        dest = np.roll(s, d, axis=axis)
        s_bar = operators.get_average(s)
        S_bar = np.repeat(s_bar[:, :, np.newaxis], p.nm, axis=2)
        S_bar_dest = np.roll(S_bar, d, axis=axis)

        # potential_free_surface = operators.empty_up(nu)

        if p.advection_model == "average_size":
            U_dest = np.sqrt(p.g * S_bar_dest)
        elif p.advection_model == "freefall":
            U_dest = np.sqrt(2 * p.g * p.dy)
        elif p.advection_model == "stress":
            sigma = stress.calculate_stress(s, last_swap, p)
            pressure = stress.get_pressure(sigma, p)
            u = np.sqrt(2 * pressure / p.solid_density)
            U = np.repeat(u[:, :, np.newaxis], p.nm, axis=2)
            U_dest = np.roll(
                U, d, axis=axis
            )  # NEED TO TAKE DESTINATION VALUE BECAUSE PRESSURE IS ZERO AT OUTLET!!!

        if axis == 1:  # vertical
            s_inv_bar = operators.get_hyperbolic_average(s)
            S_inv_bar = np.repeat(s_inv_bar[:, :, np.newaxis], p.nm, axis=2)
            S_inv_bar_dest = np.roll(S_inv_bar, d, axis=axis)
            # P = p.P_u_ref * (S_inv_bar_dest / dest)
            P = (p.dt / p.dy) * U_dest * (S_inv_bar_dest / dest)
            # print(P[~np.isnan(P)].max())

            P[:, -1, :] = 0  # no swapping up from top row
        elif axis == 0:  # horizontal
            if p.advection_model == "average_size":
                D = p.alpha * U_dest * S_bar_dest
            elif p.advection_model == "freefall":
                D = p.alpha * np.sqrt(2 * p.g * p.dy**3)
            P = D * (p.dt / p.dy**2) * (dest / S_bar_dest)

            # P = p.alpha * U_dest * (p.dt / p.dy**2) * (dest / S_bar_dest) # HACK: Changed by Benjy because this is how it _SHOULD_ be, we just dont know why yet

            if d == 1:  # left
                P[0, :, :] = 0  # no swapping left from leftmost column
            elif d == -1:  # right
                P[-1, :, :] = 0  # no swapping right from rightmost column

            slope_stable = operators.stable_slope_fast(s, d, p)  # , potential_free_surface)
            P[slope_stable] = 0

        swap_possible = unstable * ~np.isnan(dest)
        P = np.where(swap_possible, P, 0)
        swap = np.random.rand(*P.shape) < P

        total_swap = np.sum(swap, axis=2, dtype=int)
        max_swap = ((p.nu_cs - nu) * p.nm).astype(int)

        if axis == 0:
            nu_dest = np.roll(nu, d, axis=axis)
            delta_nu = nu_dest - nu
            # max_swap = np.where(
            #     potential_free_surface, ((delta_nu - p.delta_limit) * p.nm).astype(int), max_swap
            # )
            max_swap_2 = ((delta_nu - p.delta_limit) * p.nm).astype(int)  # check free surface overfilling
            max_swap = np.maximum(max_swap_2, 0)  #
            max_swap = np.minimum(max_swap, max_swap_2)  # dont overfill either condition

        overfilled = total_swap - max_swap
        overfilled = np.maximum(overfilled, 0)

        for i in range(p.nx):
            for j in range(p.ny):
                if overfilled[i, j] > 0:
                    swap_args = np.argwhere(swap[i, j, :]).flatten()
                    if len(swap_args) >= overfilled[i, j]:
                        over_indices = np.random.choice(swap_args, size=overfilled[i, j], replace=False)
                        swap[i, j, over_indices] = False

        swap_indices = np.argwhere(swap)
        dest_indices = swap_indices.copy()
        dest_indices[:, axis] -= d

        if axis == 1:
            v[swap_indices[:, 0], swap_indices[:, 1]] += d
        elif axis == 0:
            u[swap_indices[:, 0], swap_indices[:, 1]] += d

        # N_swap[:, :, axis] += np.sum(swap, axis=2)
        N_swap += np.sum(swap, axis=2)

        (
            s[swap_indices[:, 0], swap_indices[:, 1], swap_indices[:, 2]],
            s[dest_indices[:, 0], dest_indices[:, 1], dest_indices[:, 2]],
        ) = (
            s[dest_indices[:, 0], dest_indices[:, 1], dest_indices[:, 2]],
            s[swap_indices[:, 0], swap_indices[:, 1], swap_indices[:, 2]],
        )

        last_swap[swap_indices[:, 0], swap_indices[:, 1], swap_indices[:, 2]] = (
            2 * axis - 1
        )  # 1 for up, -1 for left or right

    last_swap[np.isnan(s)] = np.nan
    chi = N_swap / p.nm

    return u, v, s, c, T, chi, last_swap
