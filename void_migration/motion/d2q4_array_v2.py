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

    options = [(1, -1)]  # , (0, -1), (0, 1)]  # up, left, right
    for d in range(1, p.max_diff_swap_length + 1):
        options.extend([(0, -d), (0, d)])

    P_diff_weighting = (
        p.max_diff_swap_length * (p.max_diff_swap_length + 1) * (2 * p.max_diff_swap_length + 1) / 6
    )
    # np.random.shuffle(options)  # oh boy, this is a massive hack
    N_swap = np.zeros([p.nx, p.ny], dtype=int)

    nu = operators.get_solid_fraction(s)

    solid = nu >= p.nu_cs
    Solid = np.repeat(solid[:, :, np.newaxis], p.nm, axis=2)

    unstable = np.isnan(s) * ~Solid  # & ~swapped  # & ~Skip

    s_bar = operators.get_average(s)
    S_bar = np.repeat(s_bar[:, :, np.newaxis], p.nm, axis=2)

    potential_free_surface = operators.empty_up(nu, p)
    # potential_free_surface = None

    swap_indices = []
    dest_indices = []

    for axis, d in options:
        dest = np.roll(s, d, axis=axis)
        S_bar_dest = np.roll(S_bar, d, axis=axis)

        if p.advection_model == "average_size":
            U_dest = np.sqrt(p.g * S_bar_dest)
        elif p.advection_model == "freefall":
            U_dest = np.sqrt(p.g * p.dy)
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
            # if p.advection_model == "average_size":
            #     D = p.alpha * U_dest * S_bar_dest
            #     P = D * (p.dt / p.dy**2) * (dest / S_bar_dest)
            # elif p.advection_model == "freefall":
            #     P = (
            #         p.alpha * (p.dt / p.dy / p.dy) * U_dest * (dest / S_bar_dest)
            #     )  # alpha has units of m, and is equal to the the variance (std dev^2) of the plume in the silo at y=0.5m
            P = p.alpha * U_dest * dest * (p.dt / p.dy / p.dy) * P_diff_weighting

            if d > 0:  # left
                P[:d, :, :] = 0  # no swapping left from leftmost column
            else:  # right
                P[d:, :, :] = 0  # no swapping right from rightmost column

            # slope_stable = operators.stable_slope_fast(s, d, p, potential_free_surface)
            # slope_stable = operators.stable_slope_gradient(s, d, p)
            slope_stable = operators.stable_slope_stress(s, p, last_swap)
            P[slope_stable] = 0

        swap_possible = unstable * ~np.isnan(dest)
        P = np.where(swap_possible, P, 0)
        swap = np.random.rand(*P.shape) < P

        this_swap_indices = np.argwhere(swap)
        this_dest_indices = this_swap_indices.copy()
        this_dest_indices[:, axis] -= d

        swap_indices.extend(this_swap_indices)
        dest_indices.extend(this_dest_indices)

    # Prevent conflicts by filtering out swaps that would cause conflicts
    swap_indices_conflict_free, dest_indices_conflict_free = prevent_conflicts(swap_indices, dest_indices)

    if len(swap_indices_conflict_free) > 0:
        # Prevent overfilling by limiting swaps into cells based on maximum allowed nu
        swap_indices_filtered, dest_indices_filtered = prevent_overfilling(
            swap_indices_conflict_free, dest_indices_conflict_free, nu, potential_free_surface, p
        )

        (
            s[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1], swap_indices_filtered[:, 2]],
            s[dest_indices_filtered[:, 0], dest_indices_filtered[:, 1], dest_indices_filtered[:, 2]],
        ) = (
            s[dest_indices_filtered[:, 0], dest_indices_filtered[:, 1], dest_indices_filtered[:, 2]],
            s[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1], swap_indices_filtered[:, 2]],
        )

        last_swap[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1], swap_indices_filtered[:, 2]] = (
            2 * axis - 1
        )  # 1 for up, -1 for left or right

        N_swap[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1]] += 1

        delta = dest_indices_filtered - swap_indices_filtered
        u[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1]] += delta[:, 1] * p.dx / p.dt
        v[swap_indices_filtered[:, 0], swap_indices_filtered[:, 1]] += delta[:, 0] * p.dx / p.dt

    last_swap[np.isnan(s)] = np.nan
    chi = N_swap / p.nm

    return u, v, s, c, T, chi, last_swap


def prevent_conflicts(swap_indices, dest_indices):
    """
    Identifies and removes conflicts in swap and destination indices.
    This function takes two lists or arrays of indices, `swap_indices` and
    `dest_indices`, and identifies any conflicts where the same index appears
    more than once in either list. It then filters out any swaps involved in
    these conflicts, returning only the conflict-free swaps.
    Parameters:
    swap_indices (list or np.ndarray): Indices representing the source positions
                                       for swaps.
    dest_indices (list or np.ndarray): Indices representing the destination
                                       positions for swaps.
    Returns:
    tuple: A tuple containing two numpy arrays:
        - conflict-free swap indices
        - conflict-free destination indices
    """

    swap_indices = np.array(swap_indices)
    dest_indices = np.array(dest_indices)

    # Identify duplicates in swap_indices (reading conflicts)
    _, swap_inv, swap_counts = np.unique(swap_indices, axis=0, return_inverse=True, return_counts=True)
    swap_conflict_mask = swap_counts[swap_inv] > 1

    # Identify duplicates in dest_indices (writing conflicts)
    _, dest_inv, dest_counts = np.unique(dest_indices, axis=0, return_inverse=True, return_counts=True)
    dest_conflict_mask = dest_counts[dest_inv] > 1

    # Combine masks to filter out any swaps involved in conflicts
    conflict_mask = swap_conflict_mask | dest_conflict_mask
    conflict_free_mask = ~conflict_mask

    if len(conflict_free_mask) > 0:
        # Extract conflict-free swaps
        swap_indices = swap_indices[conflict_free_mask]
        dest_indices = dest_indices[conflict_free_mask]

    return swap_indices, dest_indices


def prevent_overfilling(swap_indices, dest_indices, nu, potential_free_surface, p):
    # Randomly permute the swaps
    perm = np.random.permutation(len(swap_indices))
    swap_indices = swap_indices[perm]
    dest_indices = dest_indices[perm]

    # Compute the change in nu for each swap
    nu_source = nu[swap_indices[:, 0], swap_indices[:, 1]]
    nu_dest = nu[dest_indices[:, 0], dest_indices[:, 1]]

    # swap_sideways = swap_indices[:, 0] != dest_indices[:, 0]
    swap_vertical = swap_indices[:, 1] != dest_indices[:, 1]
    delta_nu = nu_dest - nu_source

    # Allow for vertical swaps to exceed the delta limit
    delta_nu[swap_vertical] = 1

    # Count the number of swaps into each source location
    unique_locs, inverse_indices = np.unique(swap_indices[:, :2], axis=0, return_inverse=True)
    # counts = np.bincount(inverse_indices)

    # For each unique location, compute the allowed number of swaps
    max_swaps_bulk = (p.nm * (p.nu_cs - nu[unique_locs[:, 0], unique_locs[:, 1]])).astype(int)

    if potential_free_surface is None:
        max_swaps = max_swaps_bulk[inverse_indices]
    else:
        max_swaps_slope = ((delta_nu - p.delta_limit) * p.nm).astype(int)
        max_swaps_slope = np.maximum(max_swaps_slope, 0)

        # Check for potential free surface points
        unique_potential = potential_free_surface[unique_locs[:, 0], unique_locs[:, 1]]

        max_swaps = np.where(
            unique_potential[inverse_indices], max_swaps_slope, max_swaps_bulk[inverse_indices]
        )

        max_swaps = np.minimum(max_swaps, max_swaps_bulk[inverse_indices])

    # Find the first occurrence of each unique source location
    _, first_indices = np.unique(inverse_indices, return_index=True)

    # Compute position in group
    position_in_group = np.arange(len(swap_indices)) - first_indices[inverse_indices]

    # Build keep mask
    keep_mask = position_in_group < max_swaps

    # Apply the mask to swap_indices and dest_indices
    swap_indices = swap_indices[keep_mask]
    dest_indices = dest_indices[keep_mask]

    return swap_indices, dest_indices
