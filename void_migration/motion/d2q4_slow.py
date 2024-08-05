import numpy as np
from numpy.typing import ArrayLike
from void_migration import operators


def move_voids(
    u: ArrayLike,
    v: ArrayLike,
    s: ArrayLike,
    p,
    diag: int = 0,
    c: None | ArrayLike = None,
    T: None | ArrayLike = None,
    N_swap: None | ArrayLike = None,
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

    # if p.swap_rate == "constant":
    #     swap_rate = np.ones_like(s[:, :, 0])
    # if N_swap is None:
    #     swap_rate = np.ones_like(s[:, :, 0])
    # else:
    #     e = 0.8
    #     s_bar = get_hyperbolic_average(s)
    #     nu = 1.0 - np.mean(np.isnan(s[:, :, :]), axis=2)
    #     nu_RCP = 0.64
    #     nu_F = p.nu_cs
    #     Tg = N_swap/p.nm*p.dy*p.dy/(p.dt*p.dt)
    #     D_KT = np.sqrt(np.pi)/(8*(1+e))*s_bar/(nu*5.6916*(nu_RCP-nu_F)/(nu_RCP-nu))*np.sqrt(Tg)
    #     swap_rate = D_KT*2*p.dt/(p.dy*p.dy)
    #     # print(np.nanmin(swap_rate),np.nanmax(swap_rate))
    #     # import matplotlib.pyplot as plt
    #     # plt.figure(99)
    #     # plt.clf()
    #     # plt.ion()
    #     # plt.imshow(N_swap)
    #     # plt.colorbar()
    #     # plt.pause(1)

    # N_swap = np.ones_like(s[:, :, 0]) # HACK - SET NON-ZERO Tg EVERYWHERE FOR TESTING

    nu = 1.0 - np.mean(np.isnan(s), axis=2)

    skip = operators.empty_nearby(nu, p)

    s_bar = operators.get_average(s)
    s_inv_bar = operators.get_hyperbolic_average(s)

    for index in p.indices:
        i, j, k = np.unravel_index(index, [p.nx, p.ny - 1, p.nm])

        if not skip[i, j]:
            if np.isnan(s[i, j, k]):
                if not operators.locally_solid(s, i, j, p):
                    # UP
                    if np.isnan(s[i, j + 1, k]):
                        P_u = 0
                    else:
                        P_u = p.P_u_ref * (s_inv_bar[i, j + 1] / s[i, j + 1, k])

                    # LEFT
                    if i == 0:
                        if p.cyclic_BC:
                            l = -1
                        else:
                            l = i  # will force P_l to be zero at boundary
                    else:
                        l = i - 1

                    if np.isnan(s[l, j + diag, k]) or operators.stable_slope(s, i, j, l, p):
                        P_l = 0  # P_r + P_l = 1 at s=1
                    else:
                        # P_l = (0.5 + 0.5 * np.sin(np.radians(p.theta))) / (s[l, j + diag, k]/s_inv_bar[i,j])
                        # P_l = p.P_lr_ref * (s_inv_bar[i, j] / s[l, j + diag, k])
                        P_l = p.P_lr_ref * (s[l, j + diag, k] / s_bar[l, j + diag])

                    # if hasattr(p, "internal_geometry"):
                    #     if p.boundary[l, j + diag]:
                    #         P_l *= p.internal_geometry["perf_rate"]
                    # if perf_plate and i-1==perf_pts[0]: P_l *= perf_rate
                    # if perf_plate and i-1==perf_pts[1]: P_l *= perf_rate

                    # RIGHT
                    if i == p.nx - 1:
                        if p.cyclic_BC:
                            r = 0
                        else:
                            r = i  # will force P_r to be zero at boundary
                    else:
                        r = i + 1

                    if np.isnan(s[r, j + diag, k]) or operators.stable_slope(s, i, j, r, p):
                        P_r = 0
                    else:
                        # P_r = (0.5 - 0.5 * np.sin(np.radians(p.theta))) / (s[r, j + diag, k]/s_inv_bar[i,j])
                        # P_r = p.P_lr_ref * (s_inv_bar[i, j] / s[r, j + diag, k])
                        P_r = p.P_lr_ref * (s[r, j + diag, k] / s_bar[r, j + diag])

                    # if p.internal_geometry:
                    #     if p.boundary[r, j + diag]:
                    #         P_r *= p.internal_geometry["perf_rate"]
                    # if perf_plate and i+1==perf_pts[0]: P_r *= perf_rate
                    # if perf_plate and i+1==perf_pts[1]: P_r *= perf_rate

                    P_tot = P_u + P_l + P_r

                    if P_tot > 1:
                        print(f"Error: P_tot > 1, P_u = {P_u}, P_l = {P_l}, P_r = {P_r}")

                    dest = None
                    if P_tot > 0:
                        P = np.random.rand()
                        if P < P_u and P_u > 0:  # go up
                            dest = [i, j + 1, k]
                            if not np.isnan(s[i, j + 1, k]):
                                v[i, j] += 1
                        elif P < (P_l + P_u):  # go left
                            dest = [l, j + diag, k]

                            if diag == 0:
                                u[i, j] += 1  # LEFT
                                v[i, j] += 1
                            else:
                                u[i, j] += np.sqrt(2)  # UP LEFT
                                v[i, j] += np.sqrt(2)
                        elif P < P_tot:  # go right
                            dest = [r, j + diag, k]

                            if diag == 0:
                                u[i, j] -= 1  # RIGHT
                                v[i, j] += 1
                            else:
                                u[i, j] -= np.sqrt(2)  # UP RIGHT
                                v[i, j] += np.sqrt(2)
                        else:
                            pass

                        if dest is not None:
                            [s, c, T], nu = operators.swap([i, j, k], dest, [s, c, T], nu, p)

                        # N_swap[i, j] += 1
                        # N_swap[dest[0],dest[1]] += 1

    return u, v, s, c, T, N_swap
