import numpy as np


def update(u, v, s, p, c, outlet, t):
    """
    Add voids to the system. This function is called at each time step.
    Loop through all functions defined here and call them if they are in the list of boundary methods.
    """

    boundary_methods = [name for name, obj in globals().items() if callable(obj)]

    for method in p.boundaries:
        if method in boundary_methods:
            u, v, s, c, outlet = globals()[method](u, v, s, p, c, outlet)

    if p.close_voids:
        u, v, s = close_voids(u, v, s)

    if p.wall_motion:
        if t % p.save_wall == 0:
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

    return u, v, s, c, outlet


def charge(u, v, s, p, c, outlet):
    fill_mass = p.dx * p.dy / p.nm * p.solid_density
    to_fill = []
    for i in range(p.nx):
        for k in range(p.nm):
            gaussian = (
                p.charge_rate
                * np.exp(-((i - p.nx // 2) ** 2) / (2 * (p.nx * p.sigma_charge) ** 2))
                / (p.nx * p.sigma_charge * np.sqrt(2 * np.pi))
            )  # Gaussian distribution, more likely to fill in the middle
            if np.random.rand() < gaussian:
                nu_up = np.roll(1 - np.mean(np.isnan(s[i, :, :]), axis=1), -1)
                solid = ~np.isnan(s[i, :, k])
                liquid_up = nu_up + 1 / p.nm <= p.nu_cs

                solid_indices = np.nonzero(solid & liquid_up)[0]
                if len(solid_indices) > 0:
                    topmost_solid = solid_indices[-1]
                    if topmost_solid < p.ny - 1:
                        to_fill.append((i, topmost_solid + 1, k))
                        p.inlet += fill_mass

    if len(to_fill) > 0:
        fill_sizes = np.random.choice(p.size_choices, size=len(to_fill), p=p.size_weights)
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

        i, j, k = np.array(to_fill).T
        s[i, j, k] = fill_sizes_sorted

    return u, v, s, c, outlet


def central_outlet(u, v, s, p, c, outlet):
    fill_mass = p.dx * p.dy / p.nm * p.solid_density
    for i in range(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1):
        for k in range(p.nm):
            if np.random.rand() < p.outlet_rate:
                if not np.isnan(s[i, 0, k]):
                    if p.refill:
                        target_column = np.random.choice(p.nx)
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
                    p.outlet += fill_mass

    return u, v, s, c, outlet


def right_outlet(u, v, s, p, c, outlet):
    for i in range(p.nx - p.half_width * 2, p.nx):
        for k in range(p.nm):
            if np.random.rand() < p.outlet_rate:
                if not np.isnan(s[i, 0, k]):
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
    return u, v, s, c, outlet

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


def multiple_outlets(u, v, s, p, c, outlet):
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


def slope(u, v, s, p, c, outlet):
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
    return u, v, s, c, outlet


def vibrate_first(u, v, s, p, c, outlet):
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


def vibrate_random(u, v, s, p, c, outlet):
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
    return u, v, s, c, outlet


def vibro_top(u, v, s, p, c, outlet):
    for i in range(p.nx):
        for k in range(p.nm):
            if not np.isnan(s[i, 0, k]):
                if (
                    np.random.rand() < p.void_production_rate * p.dt / p.dy
                    and np.sum(np.isnan(s[i, :, k])) > 0
                ):
                    v[i, :] += np.isnan(s[i, :, k])
                    s[i, :, k] = np.roll(s[i, :, k], 1)
    return u, v, s, c, outlet


def pour(u, v, s, p, c, outlet):
    s[p.nx // 2 - p.half_width : p.nx // 2 + p.half_width + 1, -1, :] = 1.0
    return u, v, s, c, outlet


def wall(u, v, s, p, c, outlet):  # Remove at central outlet - use this one
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


def place_on_top(u, v, s, p, c, outlet):  # place cells on top, centre starting at base
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

    return u, v, s, c, outlet


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


def close_voids(u, v, s, p):
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
    return u, v, s
