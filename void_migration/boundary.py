import numpy as np


def add_voids(u, v, s, p, c, outlet):
    if p.add_voids == "central_outlet":  # Remove at central outlet - use this one
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
                        outlet[-1] += 1
    elif p.add_voids == "right_outlet":  # Remove at central outlet - use this one
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
    elif p.add_voids == "multiple_outlets":  # Remove at multiple points in base
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
    elif p.add_voids == "slope":  # Add voids at base
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
    elif p.add_voids == "vibro_first":  # Add voids at base
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
    elif p.add_voids == "vibro_random":  # Add voids at base
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
    elif p.add_voids == "vibro_top":  # Add voids at base
        # for i in range(5,nx-5):
        for i in range(p.nx):
            for k in range(p.nm):
                if not np.isnan(s[i, 0, k]):
                    if (
                        np.random.rand() < p.void_production_rate * p.dt / p.dy
                        and np.sum(np.isnan(s[i, :, k])) > 0
                    ):
                        v[i, :] += np.isnan(s[i, :, k])
                        s[i, :, k] = np.roll(s[i, :, k], 1)
    elif p.add_voids == "pour":  # pour in centre at top
        s[p.nx // 2 - p.half_width : p.nx // 2 + p.half_width + 1, -1, :] = 1.0

    elif p.add_voids == "place_on_top":  # pour in centre starting at base
        if p.gsd_mode == "bi":  # bidisperse
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)

            req = np.random.choice(
                [p.s_m, p.s_M], size=[(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
            )  # create an array of grainsizes
            mask = (
                np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                > p.fill_ratio
            )  # create how much to fill
            req[mask] = np.nan  # convert some cells to np.nan
        if p.gsd_mode == "mono":
            x_points = np.arange(p.nx // 2 - p.half_width, p.nx // 2 + p.half_width + 1)
            req = p.s_m * np.ones(
                [(p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm]
            )  # monodisperse

            p.s_M = p.s_m
            mask = (
                np.random.rand((p.nx // 2 + p.half_width + 1) - (p.nx // 2 - p.half_width), p.nm)
                > p.fill_ratio
            )
            req[mask] = np.nan

        den = 1 - np.mean(np.isnan(s), axis=2)
        if np.mean(den) == 0.0:
            for i in range(len(x_points)):
                for k in range(p.nm):
                    s[x_points[i], 0, k] = req[i, k]
                    if ~np.isnan(req[i, k]):
                        c[x_points[i], 0, k] = p.current_cycle
        else:
            for i in range(len(x_points)):
                for k in range(p.nm):
                    if (
                        np.isnan(s[x_points[i], 0, k])
                        and np.count_nonzero(np.isnan(s[x_points[i], :, k])) == p.ny
                    ):
                        s[x_points[i], 0, k] = req[i, k]
                        if ~np.isnan(req[i, k]):
                            c[x_points[i], 0, k] = p.current_cycle
                    else:
                        a = np.max(np.argwhere(~np.isnan(s[x_points[i], :, k])))  # choose the max ht
                        if a >= p.ny - 2:
                            pass
                        else:
                            s[x_points[i], a + 1, k] = req[i, k]  # place a cell on the topmost cell "a+1"
                            if ~np.isnan(req[i, k]):
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
