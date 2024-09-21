import numpy as np


def IC(p):
    """
    Sets up the initial value of the grain size and/or void distribution everywhere.

    Args:
        p: Parameters class. In particular, the `gsd_mode` and `IC_mode` should be set to determine the grain size distribution (gsd) and the initial condition (IC).

    Returns:
        The array of grain sizes. Values of `NaN` are voids.
    """
    rng = np.random.default_rng()
    pre_masked = False

    # pick a grain size distribution
    if p.gsd_mode == "mono":
        s = np.nan * np.ones([p.nx, p.ny, p.nm])  # monodisperse
        for i in range(p.nx):
            for j in range(p.ny):
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, j, fill] = p.s_m
        p.s_M = p.s_m

    if p.gsd_mode == "half_half":
        in_ny = int(np.ceil(p.ny * 0.4))  # the height can be controlled here if required
        s = np.nan * np.ones([p.nx, p.ny, p.nm])  # monodisperse
        for i in range(p.nx):
            for j in range(in_ny // 2):
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, j, fill] = p.s_M
            for k in range(in_ny // 2, in_ny):
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, k, fill] = p.s_m
        if len(p.cycles) > 0:  # p.charge_discharge:
            pre_masked = False
        else:
            pre_masked = True

    if p.gsd_mode == "four_layers":
        s = np.nan * np.ones([p.nx, p.ny, p.nm])  # monodisperse
        layers = 4
        d_pts = p.ny * 0.6
        lay_breaks = p.ny / layers
        lay_breaks = d_pts / layers
        y_ordinates = []
        for i in range(layers):
            y_ordinates.append(np.arange(0 + lay_breaks * i, lay_breaks * (i + 1), 1))
        y_ordinates = np.array(y_ordinates).astype(int)

        for i in range(p.nx):
            for j in y_ordinates[0]:
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, j, fill] = p.s_M
            for k in y_ordinates[1]:
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, k, fill] = p.s_m
            for l in y_ordinates[2]:
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, l, fill] = p.s_M
            for m in y_ordinates[3]:
                fill = rng.choice(p.nm, size=int(p.nm * p.nu_fill), replace=False)
                s[i, m, fill] = p.s_m
        if len(p.cycles) > 0:  # p.charge_discharge:
            pre_masked = False
        else:
            pre_masked = True

    if p.gsd_mode == "bi" or p.gsd_mode == "fbi":  # bidisperse
        if (p.nm * p.large_concentration * p.nu_fill) < 2:
            s = np.random.choice([p.s_m, p.s_M], size=[p.nx, p.ny, p.nm])
        else:
            s = np.nan * np.ones([p.nx, p.ny, p.nm])
            for i in range(p.nx):
                for j in range(p.ny):
                    large = rng.choice(
                        p.nm, size=int(p.nm * p.large_concentration * p.nu_fill), replace=False
                    )
                    s[i, j, large] = p.s_M
                    remaining = np.where(np.isnan(s[i, j, :]))[0]
                    small = rng.choice(
                        remaining, size=int(p.nm * (1 - p.large_concentration) * p.nu_fill), replace=False
                    )
                    s[i, j, small] = p.s_m
        if len(p.cycles) > 0:  # p.charge_discharge:
            pre_masked = False
        else:
            pre_masked = True

    elif p.gsd_mode == "poly":  # polydisperse
        """
        # s_0 = p.s_m / (1.0 - p.s_m)  # intermediate calculation
        s_non_dim = np.random.rand(p.nm)
        # s = (s + s_0) / (s_0 + 1.0)  # now between s_m and 1
        this_s = (p.s_M - p.s_m) * s_non_dim + p.s_m
        s = np.nan * np.ones([p.nx, p.ny, p.nm])
        # HACK: gsd least uniform in space, still need to account for voids
        for i in range(p.nx):
            for j in range(p.ny):
                np.random.shuffle(this_s)
                s[i, j, :] = this_s
        """

        s = np.random.uniform(p.s_m, p.s_M, size=[p.nx, p.ny, p.nm])

        mask = np.random.rand(p.nx, p.ny, p.nm) > p.nu_fill
        s[mask] = np.nan
        if len(p.cycles) > 0:  # p.charge_discharge:
            pre_masked = False
        else:
            pre_masked = True

    # where particles are in space
    if not pre_masked:
        if p.IC_mode == "random":  # voids everywhere randomly
            mask = np.random.rand(p.nx, p.ny, p.nm) > p.nu_fill
        elif p.IC_mode == "top":  # voids at the top
            mask = np.zeros([p.nx, p.ny, p.nm], dtype=bool)
            mask[:, int(p.fill_ratio * p.ny) :, :] = True
        elif p.IC_mode == "full":  # completely full
            mask = np.zeros_like(s, dtype=bool)
        elif p.IC_mode == "column":  # just middle full to top
            mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)
            mask[
                p.nx // 2 - int(p.fill_ratio / 4 * p.nx) : p.nx // 2 + int(p.fill_ratio / 4 * p.nx), :, :
            ] = False

            mask[
                :, -1, :
            ] = True  # top row can't be filled for algorithmic reasons - could solve this if we need to

        elif p.IC_mode == "empty":  # completely empty
            mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)

        elif p.IC_mode == "left_column":  # just middle full to top
            mask = np.ones([p.nx, p.ny, p.nm], dtype=bool)
            mask[: int(p.fill_ratio * p.nx), :, :] = False

            mask[
                :, -1, :
            ] = True  # top row can't be filled for algorithmic reasons - could solve this if we need to

        s[mask] = np.nan

        if p.wall_motion:
            nu = 1.0 - np.mean(np.isnan(s[:, :, :]), axis=2)
            start_sim = np.min(np.argwhere(nu > 0), axis=0)[
                0
            ]  # gives the start position of column in x-direction
            end_sim = np.max(np.argwhere(nu > 0), axis=0)[
                0
            ]  # gives the end position of column in x-direction

            s[0 : start_sim - 1, :, :] = 0  # Depending upon these values, make changes in void_migration.py
            s[end_sim + 2 :, :, :] = 0

    return s


def set_boundary(s, X, Y, p):
    if p.internal_geometry:
        p.boundary = np.zeros([p.nx, p.ny], dtype=bool)
        # boundary[4:-4:5,:] = 1
        p.boundary[np.cos(500 * 2 * np.pi * X) > 0] = 1
        p.boundary[:, : p.nx // 2] = 0
        p.boundary[:, -p.nx // 2 :] = 0
        p.boundary[:, p.ny // 2 - 5 : p.ny // 2 + 5] = 0
        p.boundary[np.abs(X) - 2 * p.half_width * p.dy > Y] = 1
        p.boundary[np.abs(X) - 2 * p.half_width * p.dy > p.H - Y] = 1
        boundary_tile = np.tile(p.boundary.T, [p.nm, 1, 1]).T
        s[boundary_tile] = np.nan
    else:
        p.boundary = np.zeros([p.nx, p.ny], dtype=bool)


def inclination(p, s):
    """
    This function is used when the bottom of silo is to be constructed with
    certain angle with horizontal. The variable used is form_angle which
    is by default set to 0 in defaults.json5
    """
    if p.form_angle == 0:
        pass
    elif p.form_angle > 0:
        req_r = np.arange(p.nx // 2 + p.half_width + 1, p.nx, 1) - (p.nx // 2 + p.half_width + 1)
        req_r_y = []
        for i in req_r:
            if (
                round(np.tan(np.radians(p.form_angle)) * i) < 2
            ):  # to avoid larger outlet when angle becomes less
                req_r_y.append(2)
            else:
                req_r_y.append(round(np.tan(np.radians(p.form_angle)) * i))

        req_l = np.arange(0, p.nx // 2 - p.half_width, 1)
        req_l_y = []
        for i in req_l:
            if round(np.tan(np.radians(p.form_angle)) * i) < 2:
                req_l_y.append(2)
            else:
                req_l_y.append(round(np.tan(np.radians(p.form_angle)) * i))

        x_vals = np.concatenate((req_l, (p.nx // 2 + p.half_width + 1) + req_r))
        y_vals = np.concatenate((req_l_y[::-1], req_r_y))
        print(x_vals)

        for i in range(len(x_vals)):
            for j in range(y_vals[i]):
                s[x_vals[i], j, :] = 0

        s = np.ma.masked_where(s == 0, s)

    return s


def set_concentration(s, X, Y, p):
    # if hasattr(p, "temperature"):
    #     c = np.zeros_like(s)  # original bin that particles started in
    #     c[int(p.internal_geometry.perf_pts[0] * p.nx) : int(p.internal_geometry.perf_pts[1] * p.nx)] = 1
    #     c[int(p.internal_geometry.perf_pts[1] * p.nx) :] = 2
    #     c[np.isnan(s)] = np.nan
    if len(p.cycles) > 0:  # p.charge_discharge:
        if p.IC_mode == "full":
            c = np.ones_like(s)
        else:
            c = np.zeros_like(s)  # original bin that particles started in
            c[np.isnan(s)] = np.nan
    else:
        c = None

    return c
