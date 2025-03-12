import numpy as np
import HGD.operators
from scipy.stats import truncnorm


def update(p, state):
    """
    Add voids to the system. This function is called at each time step.
    Loop through all functions defined here and call them if they are in the list of boundary methods.
    """

    boundary_methods = [name for name, obj in globals().items() if callable(obj)]

    for method in p.boundaries:
        if method in boundary_methods:
            state = globals()[method](p, *state)

    if p.close_voids:
        state = close_voids(*state)

    if p.wall_motion:
        if p.t % p.save_wall == 0:
            s = state[0]
            s_mean = np.nanmean(s, axis=2)

            start_sim = np.min(np.argwhere(s_mean > 0), axis=0)[
                0
            ]  # gives the start position of column in x-direction
            end_sim = np.max(np.argwhere(s_mean > 0), axis=0)[
                0
            ]  # gives the end position of column in x-direction

            if start_sim > 1 and end_sim + 1 < p.nx - 1:
                # s[start_sim-2:start_sim-1,:,:] = np.nan
                s[end_sim + 2 : end_sim + 3, :, :] = np.nan

            state[0] = s

    return state


def charge(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    fill_mass = p.dx * p.dy / p.nm * p.solid_density
    to_fill = []
    for i in range(p.nx):
        gaussian = (
            p.charge_rate
            / p.dx
            * np.exp(-((i - (p.nx - 1) / 2) ** 2) / (2 * (p.nx * p.sigma_charge) ** 2))
            / (p.nx * p.sigma_charge * np.sqrt(2 * np.pi))
        )  # Gaussian distribution, more likely to fill in the middle

        nu_up = np.roll(1 - np.mean(np.isnan(s[i, :, :]), axis=1), -1)
        liquid_up = nu_up + 1 / p.nm <= p.nu_cs
        for k in range(p.nm):
            if np.random.rand() < gaussian:
                solid = ~np.isnan(s[i, :, k])
                solid_indices = np.nonzero(solid & liquid_up)[0]
                if len(solid_indices) > 0:
                    topmost_solid = solid_indices[-1]
                    if topmost_solid < p.ny - 1:
                        to_fill.append((i, topmost_solid + 1, k))
                        p.inlet += fill_mass

    if len(to_fill) > 0:
        fill_sizes = np.random.choice(p.size_choices, size=len(to_fill), p=p.size_weights)
        i, j, k = np.array(to_fill).T

        if p.elutriation:
            # Sort the array in ascending order
            fill_sizes = np.sort(fill_sizes)

            # Split the sorted array into two parts
            first_half = fill_sizes[::2]  # Pick every second element starting from index 0 (smallest values)
            second_half = fill_sizes[1::2][
                ::-1
            ]  # Pick every second element starting from index 1 (largest values, reversed)

            # Create an empty array to hold the result
            fill_sizes_sorted = np.empty_like(fill_sizes)

            # Place smaller values at both ends
            fill_sizes_sorted[: len(first_half)] = first_half
            fill_sizes_sorted[len(first_half) :] = second_half

            s[i, j, k] = fill_sizes_sorted
        else:
            s[i, j, k] = fill_sizes

    return s, u, v, c, T, last_swap, chi, sigma, outlet


def top_left_inlet(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    fill_mass = p.dx * p.dy / p.nm * p.solid_density
    for i in range(p.half_width):
        for k in range(p.nm):
            if np.random.rand() < p.inlet_rate:
                s[i, -1, k] = np.random.choice([p.s_m, p.s_M])
                p.inlet += fill_mass
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def central_outlet(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    fill_mass = p.dx * p.dy / p.nm * p.solid_density

    for i in range(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1):
        this_nu = 1 - np.mean(np.isnan(s[i, 0, :]))
        num_to_empty = int((this_nu - p.outlet_nu) * p.nm)
        if num_to_empty > 0:
            to_empty = np.random.choice(np.nonzero(~np.isnan(s[i, 0, :]))[0], num_to_empty, replace=False)
            for k in to_empty:
                if p.refill:
                    target_column = np.random.choice(p.nx)
                    nu_up = np.roll(1 - np.mean(np.isnan(s[target_column, :, :]), axis=1), -1)
                    solid = ~np.isnan(s[target_column, :, k])
                    liquid_up = nu_up + 1 / p.nm <= p.nu_cs

                    solid_indices = np.nonzero(solid & liquid_up)[0]
                    if len(solid_indices) > 0:
                        topmost_solid = solid_indices[-1]
                        if topmost_solid < p.ny - 1:
                            available_voids = np.nonzero(np.isnan(s[target_column, topmost_solid + 1, :]))[0]
                            this_void = np.random.choice(available_voids)
                            print(s[target_column, topmost_solid + 1, this_void], s[i, 0, k])
                            s[target_column, topmost_solid + 1, this_void], s[i, 0, k] = (
                                s[i, 0, k],
                                s[target_column, topmost_solid + 1, this_void],
                            )
                        else:
                            print("WARNING: No room to refill")
                else:
                    s[i, 0, k] = np.nan
                p.outlet += fill_mass

    return s, u, v, c, T, last_swap, chi, sigma, outlet


def right_outlet(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for i in range(p.nx - p.half_width * 2, p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if np.random.rand() < p.outlet_rate:
                    if p.refill:
                        target_column = np.random.choice(p.half_width * 2)
                        nu_up = np.roll(1 - np.mean(np.isnan(s[target_column, :, :]), axis=1), -1)
                        solid = ~np.isnan(s[target_column, :, k])
                        liquid_up = nu_up + 1 / p.nm <= p.nu_cs

                        solid_indices = np.nonzero(solid & liquid_up)[0]
                        if len(solid_indices) > 0:
                            topmost_solid = solid_indices[-1]
                            if topmost_solid < p.ny - 1:
                                s[target_column, topmost_solid + 1, k], s[i, 0, k] = (
                                    s[i, 0, k],
                                    s[target_column, topmost_solid + 1, k],
                                )
                    else:
                        s[i, 0, k] = np.nan
                    outlet[-1] += 1
    return s, u, v, c, T, last_swap, chi, sigma, outlet

    # elif temp_mode == "temperature":  # Remove at central outlet
    #     for i in range(nx // 2 - half_width, nx // 2 + half_width + 1):
    #         for k in range(nm):
    #             # if np.random.rand() < Tg:
    #             if not np.isnan(s[i, 0, k]):
    #                 if refill:
    #                     if np.sum(np.isnan(s[nx // 2 - half_width : nx // 2 + half_width + 1, -1, k])) > 0:
    #                         if internal_geometry:
    #                             target = (
    #                                 nx // 2
    #                                 - half_width
    #                                 + np.random.choice(
    #                                     np.nonzero(
    #                                         np.isnan(
    #                                             s[nx // 2 - half_width : nx // 2 + half_width + 1, -1, k]
    #                                         )
    #                                     )[0]
    #                                 )
    #                             )  # HACK
    #                         else:
    #                             target = np.random.choice(np.nonzero(np.isnan(s[:, -1, k]))[0])
    #                         s[target, -1, k], s[i, 0, k] = s[i, 0, k], s[target, -1, k]
    #                         T[target, -1, k] = inlet_temperature
    #                         outlet_T.append(T[i, 0, k])
    #                 else:
    #                     s[i, 0, k] = np.nan
    #                 outlet[-1] += 1


def multiple_outlets(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for l, source_pt in enumerate(p.source_pts):
        for i in range(source_pt - p.half_width, source_pt + p.half_width + 1):
            for k in range(p.nm):
                if np.random.rand() < p.Tg[l]:
                    target = np.random.choice(np.nonzero(np.isnan(s[:, -1, k]))[0])
                    s[target, -1, k] = s[i, 0, k]
                    if target <= p.internal_geometry.perf_pts[0]:
                        c[target, -1, k] = 0
                    elif target <= p.internal_geometry.perf_pts[1]:
                        c[target, -1, k] = 1
                    else:
                        c[target, -1, k] = 2
                    s[i, 0, k] = np.nan
    return u, v, s, c, outlet


def slope(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                # MOVE UP TO FIRST VOID --- THIS GENERATES SHEARING WHEN INCLINED!
                if (
                    np.random.rand() < (p.Tg * p.H) / (p.free_fall_velocity * p.dt)
                    and np.sum(np.isnan(s[i, :, k]))
                ) > 0:  # Tg is relative height (out of the maximum depth) that voids should rise to before being filled
                    first_void = np.isnan(s[i, :, k]).nonzero()[0][0]
                    v[i, : first_void + 1] += np.isnan(s[i, : first_void + 1, k])
                    s[i, : first_void + 1, k] = np.roll(s[i, : first_void + 1, k], 1)
                # MOVE EVERYTHING UP
                # if (np.random.rand() < p.Tg * p.dt / p.dy and np.sum(np.isnan(s[i, :, k]))) > 0:
                #     if np.isnan(s[i, -1, k]):
                #         v[i, :] += 1  # np.isnan(s[i,:,k])
                #         s[i, :, k] = np.roll(s[i, :, k], 1)
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def vibrate(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if (
                    np.random.rand() < p.void_production_rate * p.dt / p.dy
                    and np.sum(np.isnan(s[i, :, k])) > 0
                ):
                    # possible_sites = np.isnan(s[i, :, k]) * (nu[i, :] < p.nu_cs)
                    nu = 1.0 - np.mean(np.isnan(s), axis=2)
                    possible_sites = nu[i, :] < p.nu_cs
                    print(possible_sites)
                    if np.sum(possible_sites) > 0:
                        # if sum(np.isnan(s[i, : first_void + 1, k]))
                        first_void = possible_sites.nonzero()[0][0]
                        v[i, : first_void + 1] += np.isnan(s[i, : first_void + 1, k])
                        s[i, : first_void + 1, k] = np.roll(s[i, : first_void + 1, k], 1)
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def vibrate_first(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    # for i in range(5,nx-5):

    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if (
                    np.random.rand() < p.void_production_rate * p.dt / p.dy
                    and np.sum(np.isnan(s[i, :, k])) > 0
                ):
                    # possible_sites = np.isnan(s[i, :, k]) * (nu[i, :] < p.nu_cs)
                    nu = 1.0 - np.mean(np.isnan(s), axis=2)
                    possible_sites = nu[i, :] < p.nu_cs
                    print(possible_sites)
                    if np.sum(possible_sites) > 0:
                        # if sum(np.isnan(s[i, : first_void + 1, k]))
                        first_void = possible_sites.nonzero()[0][0]
                        v[i, : first_void + 1] += np.isnan(s[i, : first_void + 1, k])
                        s[i, : first_void + 1, k] = np.roll(s[i, : first_void + 1, k], 1)
    return u, v, s, c, outlet


def vibrate_random(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    # for i in range(5,nx-5):
    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if (
                    np.random.rand() < p.void_production_rate * p.dt / p.dy
                    and np.sum(np.isnan(s[i, :, k])) > 0
                ):
                    nan_indices = np.where(np.isnan(s[i, :, k]))[0]
                    target_void = np.random.choice(nan_indices)
                    v[i, : target_void + 1] += np.isnan(s[i, : target_void + 1, k])
                    s[i, : target_void + 1, k] = np.roll(s[i, : target_void + 1, k], 1)
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def vibro_top(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if (
                    np.random.rand() < p.void_production_rate * p.dt / p.dy
                    and np.sum(np.isnan(s[i, :, k])) > 0
                ):
                    v[i, :] += np.isnan(s[i, :, k])
                    s[i, :, k] = np.roll(s[i, :, k], 1)
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def pour(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    s[p.nx // 2 - p.half_width : p.nx // 2 + p.half_width + 1, -1, :] = 1.0
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def wall(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    for i in range(0, p.half_width):
        for k in range(p.nm):
            # if np.random.rand() < 0.1:
            if not np.isnan(s[i, 0, k]):
                if p.refill:
                    if np.sum(np.isnan(s[0 : p.half_width, -1, k])) > 0:
                        target = np.random.choice(np.nonzero(np.isnan(s[0 : p.half_width, -1, k]))[0])
                        s[target, -1, k], s[i, 0, k] = s[i, 0, k], s[target, -1, k]

                else:
                    s[i, 0, k] = np.nan
                    if hasattr(p, "charge_discharge"):
                        c[i, 0, k] = np.nan
                outlet[-1] += 1


def place_on_top(
    p, s, u, v, c, T, last_swap, chi, sigma, outlet
):  # place cells on top, centre starting at base
    if p.gsd_mode == "bi":  # bidisperse
        if p.silo_width == "half":
            x_points = np.arange(0, p.half_width)
            req = np.random.choice([p.s_m, p.s_M], size=[p.half_width, p.nm])  # create an array of grainsizes

            mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
            req[mask] = np.nan

        elif p.silo_width == "full":
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width)

            req = np.random.choice(
                [p.s_m, p.s_M], size=[(p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm]
            )  # create an array of grainsizes
            mask = (
                np.random.rand((p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm) > p.fill_ratio
            )  # create how much to fill
            req[mask] = np.nan  # convert some cells to np.nan

    if p.gsd_mode == "fbi":  # bidisperse
        if p.silo_width == "half":
            x_points = np.arange(0, p.half_width)
            #     req = np.random.choice(
            #     [p.s_m, p.Fr*p.s_m, p.s_M, p.Fr*p.s_M], size=[p.half_width, p.nm]
            # )  # create an array of grainsizes
            f_1 = p.half_width - int(p.half_width / 2)
            f_2 = p.half_width - f_1
            req1 = np.random.uniform(p.s_m, p.Fr * p.s_m, size=[p.nm, f_1])
            req2 = np.random.uniform(p.s_M, p.Fr * p.s_M, size=[p.nm, f_2])
            req3 = np.concatenate((req1, req2), axis=1)
            req = req3.reshape(p.half_width, p.nm)
            mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
            req[mask] = np.nan

        elif p.silo_width == "full":
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width)

            f_1 = int(len(x_points) - int(len(x_points) / 2))
            f_2 = int(len(x_points) - f_1)

            req1 = np.random.uniform(p.s_m, p.Fr * p.s_m, size=[p.nm, f_1])
            req2 = np.random.uniform(p.s_M, p.Fr * p.s_M, size=[p.nm, f_2])

            req3 = np.concatenate((req1, req2), axis=1)

            req = req3.reshape(int(len(x_points)), p.nm)

            mask = (
                np.random.rand((p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm) > p.fill_ratio
            )  # create how much to fill
            req[mask] = np.nan  # convert some cells to np.nan

    if p.gsd_mode == "mono":
        if p.silo_width == "half":
            x_points = np.arange(0, p.half_width)
            req = p.s_m * np.ones([p.half_width, p.nm])  # monodisperse

            p.s_M = p.s_m
            mask = np.random.rand(p.half_width, p.nm) > p.fill_ratio
            req[mask] = np.nan

        # elif p.silo_width == "full":
        #     x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)
        #     req = p.s_m * np.ones(
        #         [(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
        #     )  # monodisperse

        #     p.s_M = p.s_m
        #     mask = (
        #         np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
        #         > p.fill_ratio
        #     )
        #     req[mask] = np.nan

        elif p.silo_width == "full":
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width)
            req = p.s_m * np.ones(
                [(p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm]
            )  # monodisperse

            p.s_M = p.s_m
            mask = (
                np.random.rand((p.nx // 2 + p.half_width) - (p.nx // 2 - p.half_width), p.nm) > p.fill_ratio
            )
            req[mask] = np.nan

    den = 1 - np.mean(np.isnan(s), axis=2)
    if np.mean(den) == 0.0:
        for i in range(len(x_points)):
            for k in range(p.nm):
                s[x_points[i], 0, k] = req[i, k]
                if ~np.isnan(s[x_points[i], 0, k]):
                    c[x_points[i], 0, k] = p.current_cycle
    else:
        for i in range(len(x_points)):
            for k in range(p.nm):
                if (
                    np.isnan(s[x_points[i], 0, k])
                    and np.count_nonzero(np.isnan(s[x_points[i], :, k])) == p.ny
                ):
                    s[x_points[i], 0, k] = req[i, k]
                    if ~np.isnan(s[x_points[i], 0, k]):
                        c[x_points[i], 0, k] = p.current_cycle
                else:
                    a = np.max(np.argwhere(~np.isnan(s[x_points[i], :, k])))  # choose the max ht
                    if a >= p.ny - 3:
                        pass
                    else:
                        if den[x_points[i], a + 1] < p.nu_cs:
                            # print("TTTTTTTTTTTTTTTTTTTTTTTTT",den[x_points[i], a + 1])
                            s[x_points[i], a + 1, k] = req[i, k]  # place a cell on the topmost cell "a+1"
                            if ~np.isnan(s[x_points[i], a + 1, k]):
                                c[x_points[i], a + 1, k] = p.current_cycle
                        else:
                            a = np.min(np.argwhere(den[x_points[i], :] < p.nu_cs))
                            s[x_points[i], a + 1, k] = req[i, k]  # place a cell on the topmost cell "a+1"
                            if ~np.isnan(s[x_points[i], a + 1, k]):
                                c[x_points[i], a + 1, k] = p.current_cycle

    return s, u, v, c, T, last_swap, chi, sigma, outlet


# def generate_voids(u, v, s):  # Moving voids create voids
#     U = np.sqrt(u**2 + v**2)
#     for i in range(nx):
#         for j in range(ny):
#             for k in range(nm):
#                 if not np.isnan(s[i, j, k]):
#                     if np.random.rand() < 1 * U[i, j] / nm * dt / dy:  # FIXME
#                         last_void = (
#                             np.isfinite(s[i, :, k]).nonzero()[0][-1] + 1
#                         )  # get first void above top filled site
#                         # FIXME: THIS WILL DIE IF TOP HAS A VOID IN IT
#                         v[i, j : last_void + 1] += 1  # np.isnan(s[i,j:last_void+1,k])
#                         s[i, j : last_void + 1, k] = np.roll(s[i, j : last_void + 1, k], 1)
#     return u, v, s


def close_voids(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    """
    Not implemented. Do not use.
    """
    for i in range(p.nx):
        for j in np.arange(p.ny - 1, -1, -1):  # go from top to bottom
            for k in range(p.nm):
                if np.isnan(s[i, j, k]):
                    pass
                    # if np.random.rand() < 5e-2 * dt / dy:  # FIXME
                    #     v[i, j:] -= 1
                    #     s[i, j:, k] = np.roll(s[i, j:, k], -1)
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def silo_fluid(p, s, u, v, c, T, last_swap, chi, sigma, outlet):
    nu = HGD.operators.get_solid_fraction(s)
    top = HGD.operators.get_top(nu, p, nu_lim=p.nu_cs)

    # Particles fall a height H in t_f = A/s^2, A = (18\mu H)/(g \Delta rho)
    # If a total amount of material M is discharged into the system over a time period T, then the average mass flow rate is M/T
    # The mass flow rate of a particular size \dot{m}(s,t) = M/T \times f(s,t) \times \mathcal{H}(t-Hs^2/A)\mathcal{H}(T+Hs^2/A-t), where f(s,t) is the fraction of material of size s at time t, and the two heaviside functions ensure that the material is only discharged between t_f and T+t_f.

    # s_bins = np.linspace(p.s_m, p.s_M, p.nm + 1) # should be defined in initial.py for each case
    # f = initial.gsd(p, s_bins)
    s_bins = np.array([p.s_m, p.s_M])  # HACK: JUST BIDISPERSE
    f = np.array([0.5, 0.5])  # HACK: ASSUME EQUAL WEIGHTS

    A = (18 * p.dynamic_viscosity * p.H) / (p.g * p.delta_rho)
    t_f = A / s_bins**2
    fill_rate = p.fill_fraction / p.charge_duration * p.nu_cs[0, 0] * p.nx * p.ny * p.nm
    m = fill_rate * f * np.heaviside(p.t - t_f, 1) * np.heaviside(p.charge_duration + t_f - p.t, 1)

    # Now for each s calculate the distance from the centre it is spread to
    u_y = s_bins**2 * p.delta_rho * p.g / (18 * p.dynamic_viscosity)
    tau_p = s_bins**2 * p.solid_density / (18 * p.dynamic_viscosity)
    tau_f = p.H / u_y
    Stk = tau_p / tau_f
    u_x = p.aspect_ratio_y * u_y
    W_crit = u_x * t_f.mean()  # distance from centre of silo to where particles are spread to

    A = 1  # fitting parameter for mu
    B = 0.5  # fitting parameter for sigma

    if p.tstep == 0:
        print(f"\nTime: {p.t}, t_f: {t_f}, m: {m}, W_crit: {W_crit}")

    for n in range(len(s_bins)):
        # N = int(round(m[n] * p.dt))  # target value we want to fill across whole row
        N = np.random.poisson(m[n] * p.dt)  # target value we want to fill across whole row
        # mu = W_crit[n] / (Stk[n] ** A + 1)
        mu = 0  # i never want PILES at the corners and nothing in the middle??

        sigma = (p.H - p.fill_opening_width) / (Stk[n] ** B + 1) + p.fill_opening_width
        a, b = (-p.W / 2.0 - mu) / sigma, (p.W / 2.0 - mu) / sigma

        # Compute the truncated normal PDF
        pdf_values = truncnorm.pdf(p.x, a, b, loc=mu, scale=sigma)

        # Normalize to get discrete probabilities
        discrete_probs = pdf_values / np.sum(pdf_values)
        to_fill = np.random.multinomial(N, discrete_probs)

        # for i in range(p.nx):
        #     j = top[i]

        #     sigma = (p.H - p.fill_opening_width) / (Stk[n] ** B + 1) + p.fill_opening_width
        #     a1, b1 = (-p.W / 2.0 - mu) / sigma, (p.W / 2.0 - mu) / sigma
        #     a2, b2 = (-p.W / 2.0 + mu) / sigma, (p.W / 2.0 + mu) / sigma

        #     to_fill = int(
        #         round(
        #             N
        #             # / 2.0
        #             * p.dx
        #             * (
        #                 truncnorm(a=a1, b=b1, loc=mu, scale=sigma).pdf(p.x[i])
        #                 # + truncnorm(a=a2, b=b2, loc=-mu, scale=sigma).pdf(p.x[i])
        #             )
        #         )
        #     )
        #     to_fill_total += to_fill

        # debug = False
        # if debug:

        #     import matplotlib.pyplot as plt

        #     plt.figure(32)
        #     plt.clf()
        #     plt.title(n)
        #     plt.ion()
        #     plt.plot(
        #         p.x,
        #         truncnorm.pdf(p.x, a1, b1, loc=mu, scale=sigma),
        #         "b-",
        #     )
        #     plt.plot(
        #         p.x,
        #         truncnorm.pdf(p.x, a2, b2, loc=-mu, scale=sigma),
        #         "r-",
        #     )
        #     plt.pause(0.01)

        for i in range(p.nx):
            j = top[i]
            s = place_at(s, i, j, p, to_fill[i], s_bins, n)
        # if N > 0:
        #     print(
        #         f"Filled {to_fill_total} voids with {s_bins[n]}. Target was {N} ({to_fill_total/N*100:.2f}%)"
        #     )
    return s, u, v, c, T, last_swap, chi, sigma, outlet


def place_at(s, i, j, p, to_fill, s_bins, n):
    # Now actually fill the voids
    while to_fill > 0:
        remaining_voids = p.nu_cs[i, j] - HGD.operators.get_solid_fraction(s, [i, j])
        if remaining_voids > 0:
            available_k = np.nonzero(np.isnan(s[i, j, :]))[0]
            if len(available_k) > 0:
                num = min(to_fill, len(available_k), int(remaining_voids * p.nm))
                chosen_k = np.random.choice(available_k, size=num, replace=False)
                s[i, j, chosen_k] = s_bins[n]
                to_fill -= len(chosen_k)
        else:
            j += 1
            if j >= p.ny:
                raise ValueError("Not enough space to fill")
    return s
