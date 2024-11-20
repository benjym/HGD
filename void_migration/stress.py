import numpy as np
import warnings
from void_migration import operators

# Implementing Eq 36 and 37 from:
# Models of stress fluctuations in granular media
# P. Claudin and J.-P. Bouchaud, M. E. Cates and J. P. Wittmer
# NOTE: in our geometry tan(psi) = 1 (because dx=dy)

# Since tan(psi) = 1, we have that c_0^2 = 1 - stress_fraction

# K = sigma_xx / sigma_yy

# K_0 STRESS
# c_0^2 = lateral earth pressure coefficient, K
# Using Jaky's formula, K = 1 - sin(phi), where phi is the repose angle (actually effective angle of internal friction)
# stress_fraction = sin(phi)
# or if we have a different value for K, we have that
# stress_fraction = 1 - K

# ISOTROPIC STRESS
# c_0^2 = 1 - stress_fraction = K
# K = 1
# stress_fraction = 0

# ANISOTROPIC STRESS
# Rothenburg and Bathurst: mu approx = a/2 <--- does this help??
# Can take a = beta x magnitude of 1/M sum_k P^{last}_k
# Where P_{last}^k = -1 for vertical and 1 for horizontal? (So that a = 0 for homogenous, 1 for fully ordered

# Now we want to map the following:
# a = -1 -> K_p = (1+sin(phi))/(1-sin(phi))
# a = 0 -> K = 1
# a = 1 -> K_a = (1-sin(phi))/(1+sin(phi))

# This can be satisfied with the following logarithmic scaling:
# K = K_p*(K_a/K_p)^((a+1)/2)


def calculate_stress_fraction(last_swap, p):
    if p.stress_mode == "K_0":
        stress_fraction = np.sin(np.radians(p.repose_angle))
        stress_fraction = np.full([p.nx, p.ny], stress_fraction)
    elif p.stress_mode == "no_lateral":
        print("WARNING: Using no lateral stress")
        stress_fraction = np.ones([p.nx, p.ny])
    elif p.stress_mode == "isotropic":
        stress_fraction = np.zeros([p.nx, p.ny])
    elif p.stress_mode == "active":
        K_a = (1 - np.sin(np.radians(p.repose_angle))) / (1 + np.sin(np.radians(p.repose_angle)))
        stress_fraction = np.full([p.nx, p.ny], 1 - K_a)
    elif p.stress_mode == "passive":
        K_p = (1 + np.sin(np.radians(p.repose_angle))) / (1 - np.sin(np.radians(p.repose_angle)))
        stress_fraction = np.full([p.nx, p.ny], 1 - K_p)
    elif p.stress_mode == "sandpile":
        K = 1.0 / (1.0 + 2.0 * (np.tan(np.radians(p.repose_angle)) ** 2))
        stress_fraction = np.full([p.nx, p.ny], 1 - K)
    elif p.stress_mode == "anisotropic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.nanmean(
                np.abs(last_swap), axis=2
            )  # between 0 and 1, 0 for isotropic, 1 for fully anisotropic

            K_a = (1 - np.sin(np.radians(p.repose_angle))) / (1 + np.sin(np.radians(p.repose_angle)))
            K_iso = 1.0
            # K = K_iso * (K_a / K_iso) ** a_scaled
            K = (K_a - K_iso) * a + K_iso
            stress_fraction = 1 - K
            # import matplotlib.pyplot as plt
            # plt.figure(77)
            # plt.ion()
            # plt.clf()
            # plt.subplot(121)
            # plt.imshow(a.T)
            # plt.colorbar()

            # plt.subplot(122)
            # plt.imshow(K.T)
            # plt.colorbar()
            # plt.pause(0.01)

    else:
        raise ValueError("Unknown stress mode")

    return stress_fraction


def calculate_stress(s, last_swap, p):
    # return calculate_stress_NEW(s, last_swap, p)
    # return calculate_stress_OLD(s, last_swap, p)
    return harr_substep(s, last_swap, p)
    # return harr_implicit(s, last_swap, p)


def harr_substep(s, last_swap, p):
    stress_fraction = calculate_stress_fraction(last_swap, p)
    K = 1 - stress_fraction

    sigma = np.zeros([p.nx, p.ny, 2])  # sigma_xy, sigma_yy
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.dx * p.g  # * p.dy

    nu = operators.get_solid_fraction(s)

    if hasattr(p, "point_load"):
        empty_at_middle = nu[p.nx // 2, :] == 0
        top = np.where(empty_at_middle)[0]
        if len(top) > 0:
            top = top[0]
        else:
            top = p.ny

    for j in range(p.ny - 2, -1, -1):
        this_weight = nu[:, j] * weight_of_one_cell

        # F_{x, z} = F_{x, z+\Delta z} + w \Delta z
        # + \frac{\Delta z}{\Delta x^2} D \left( F_{x+\Delta x, z+\Delta z} - 2F_{x, z+\Delta z} + F_{x-\Delta x, z+\Delta z} \right).
        # K = 1
        depth = p.dy * (top - j + 1.5)  # half a cell below the cell center
        D = K[:, j] * depth

        nsubsteps = np.ceil(np.amax(D) * p.dy / (0.5 * p.dx**2)).astype(int)  # CFL=0.5
        nsubsteps = max(nsubsteps, 1)  # at least one substep

        for m in range(nsubsteps):
            if m == 0:
                up = sigma[:, j + 1, 1]
            else:
                up = sigma_inc
            right_up = np.roll(up, 1, axis=0)
            left_up = np.roll(up, -1, axis=0)

            sigma_inc = (
                this_weight / nsubsteps
                + up
                + (p.dy / nsubsteps) / (p.dx**2) * D * (left_up - 2 * up + right_up)
            )
            sigma_inc[nu[:, j] == 0] = 0

        sigma[:, j, 1] = sigma_inc

        if j == top:
            half_pad_width = int(p.pad_width / p.dx) // 2
            sigma[p.nx // 2 - half_pad_width : p.nx // 2 + half_pad_width + 1, top, 1] = p.point_load * p.t

    dsigma_dx, dsigma_dy = np.gradient(sigma[:, :, 1], p.dx, p.dy)
    Depth = (top + 2) * p.dy - p.Y
    sigma[:, :, 0] = -K * Depth * dsigma_dx  # HACK: THIS IS MISSING THE SECOND DERIVATIVE TERM

    return sigma


def harr_implicit(s, last_swap, p):
    sigma = np.zeros((p.nx, p.ny, 2))
    weight_of_one_cell = p.solid_density * p.dx * p.g  # * p.dy
    nu = operators.get_solid_fraction(s)
    w = nu * weight_of_one_cell

    if hasattr(p, "point_load"):
        sigma[p.nx // 2, p.ny - 1, 1] = p.point_load

    # Loop over rows (z-direction, top to bottom)
    for j in range(p.ny - 2, -1, -1):
        K = 1
        depth = p.dy * (p.ny - j - 1)
        D = K * depth

        # Tridiagonal coefficients
        alpha = -D * p.dy / p.dx**2
        beta = 1 + 2 * D * p.dy / p.dx**2

        # Extract known values from the row above (z+1)
        rhs = sigma[:, j + 1, 1] + w[:, j]  # * p.dy

        # Tridiagonal matrix
        a = np.full(p.nx, alpha)  # Subdiagonal
        b = np.full(p.nx, beta)  # Main diagonal
        c = np.full(p.nx, alpha)  # Superdiagonal

        # Periodic boundary conditions
        a[0] = c[-1] = alpha
        c[0] = a[-1] = alpha

        # Solve the tridiagonal system
        F_new = solve_tridiagonal(a, b, c, rhs)  # Replace with your solver

        # Update sigma for the current row
        sigma[:, j, 1] = F_new

    return sigma


def solve_tridiagonal(a, b, c, d):
    """
    Solves a tridiagonal system using the Thomas algorithm.
    a: subdiagonal (length n-1)
    b: main diagonal (length n)
    c: superdiagonal (length n-1)
    d: right-hand side (length n)
    """
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i - 1] = c[i - 1] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    # Back substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def calculate_stress_OLD(s, last_swap, p):
    stress_fraction = calculate_stress_fraction(last_swap, p)

    sigma = np.zeros([p.nx, p.ny, 2])  # sigma_xy, sigma_yy
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.dx * p.g  # * p.dy

    nu = operators.get_solid_fraction(s)

    for j in range(p.ny - 2, -1, -1):
        for i in range(p.nx):
            if nu[i, j] > 0:
                this_weight = nu[i, j] * weight_of_one_cell
                up = sigma[i, j + 1]
                if i == 0:
                    right_up = sigma[i + 1, j + 1]
                    if p.wall_friction_angle == 0 or p.repose_angle == 0:
                        left_up = [0, 0]
                    else:
                        left_up = (
                            p.wall_friction_angle / p.repose_angle * right_up
                        )  # FIXME: somehow scale the friction. should at least be with mu and not angle, even if this somehow miraculously works

                elif i == p.nx - 1:
                    left_up = sigma[i - 1, j + 1]
                    if p.wall_friction_angle == 0 or p.repose_angle == 0:
                        right_up = [0, 0]
                    else:
                        right_up = p.wall_friction_angle / p.repose_angle * left_up  # FIXME
                else:
                    left_up = sigma[i - 1, j + 1]
                    right_up = sigma[i + 1, j + 1]
                # TODO: ADD CHECK TO REDIRECT STRESS IF VOID IS PRESENT
                sigma[i, j, 0] = 0.5 * (left_up[0] + right_up[0]) + 0.5 * (1 - stress_fraction[i, j]) * (
                    left_up[1] - right_up[1]
                )
                sigma[i, j, 1] = (
                    this_weight
                    + stress_fraction[i, j] * up[1]
                    + 0.5 * (1 - stress_fraction[i, j]) * (left_up[1] + right_up[1])
                    + 0.5 * (left_up[0] - right_up[0])
                )

    return sigma


def calculate_stress_NEW(s, last_swap, p):
    stress_fraction = calculate_stress_fraction(last_swap, p)

    sigma = np.zeros([p.nx, p.ny, 2])  # sigma_xy, sigma_yy
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.g * p.dx  # * p.dy

    nu = operators.get_solid_fraction(s)

    for j in range(p.ny - 1, 0, -1):
        for i in range(1, p.nx - 1):  # HACK - ignoring boundaries for now
            if nu[i, j] > 0:
                this_weight = nu[i, j] * weight_of_one_cell
                sigma_here = sigma[i, j]
                sigma_here[1] += this_weight
                down = [i, j - 1]
                down_left = [i - 1, j - 1]
                down_right = [i + 1, j - 1]
                down_has_mass = nu[down[0], down[1]] > 0
                down_left_has_mass = nu[down_left[0], down_left[1]] > 0
                down_right_has_mass = nu[down_right[0], down_right[1]] > 0
                p_total = (
                    stress_fraction[i, j] * down_has_mass
                    + (1 - stress_fraction[i, j]) / 2.0 * down_left_has_mass
                    + (1 - stress_fraction[i, j]) / 2.0 * down_right_has_mass
                )

                num_lr = down_left_has_mass + down_right_has_mass
                if num_lr == 0:
                    lr_frac = 0.0
                else:
                    lr_frac = 1.0 / num_lr  # 0.5 if both, 1 if one

                # lr_frac = 0.5
                # p_total = 1.0

                if p_total > 0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        p_here = (
                            stress_fraction[i, j] / p_total
                        )  # stress fraction going down the vertical leg
                        p_there = (1 - p_here) / 2.0  # stress fraction going down the diagonal legs

                    if down_has_mass:
                        # only vertical stress passed straight down
                        sigma[down[0], down[1], 1] += p_here * sigma_here[1]
                    if down_left_has_mass:
                        sigma[down_left[0], down_left[1], 0] += (
                            lr_frac * sigma_here[0]  # half of the shear force from the current cell
                            - p_there * sigma_here[1]  # redirected part of the vertical force
                        )
                        sigma[down_left[0], down_left[1], 1] += (
                            -lr_frac * sigma_here[0]  # redirected part of the shear stress
                            + p_there * sigma_here[1]
                        )
                    if down_right_has_mass:
                        sigma[down_right[0], down_right[1], 0] += (
                            lr_frac * sigma_here[0]  # half of the shear stress from the current cell
                            + p_there * sigma_here[1]  # redirected part of the vertical force
                        )
                        sigma[down_right[0], down_right[1], 1] += (
                            lr_frac * sigma_here[0]  # half of the shear stress from the current cell
                            + p_there * sigma_here[1]  # redirected part of the vertical force
                        )

    return sigma


def calculate_stress_array(s, last_swap, p):
    stress_fraction = calculate_stress_fraction(last_swap, p)

    sigma = np.zeros((p.nx, p.ny, 2))  # sigma_xy, sigma_yy
    weight_of_one_cell = p.solid_density * p.dx * p.dy * p.g

    # Compute n_cells and has_cell
    n_cells = np.sum(~np.isnan(s[:, :-1, :]), axis=2)
    has_cell = n_cells > 0
    this_weight = n_cells * weight_of_one_cell

    # Prepare arrays
    up = sigma[:, 1:, :]  # sigma[i, j+1, :]
    left_up = np.zeros((p.nx, p.ny - 1, 2))
    right_up = np.zeros((p.nx, p.ny - 1, 2))

    # Compute wall_scale
    if p.wall_friction_angle == 0 or p.repose_angle == 0:
        wall_scale = 0
    else:
        wall_scale = p.wall_friction_angle / p.repose_angle

    # Compute left_up and right_up
    left_up[1:, :, :] = sigma[:-1, 1:, :]
    right_up[:-1, :, :] = sigma[1:, 1:, :]

    # Left boundary (i=0)
    if wall_scale == 0:
        left_up[0, :, :] = 0
    else:
        left_up[0, :, :] = wall_scale * right_up[0, :, :]

    # Right boundary (i = p.nx -1)
    if wall_scale == 0:
        right_up[-1, :, :] = 0
    else:
        right_up[-1, :, :] = wall_scale * left_up[-1, :, :]

    # Compute stress_fraction for valid indices
    stress_fraction_slice = stress_fraction[:, :-1]

    # Compute sigma_x and sigma_y
    sigma_x = np.zeros((p.nx, p.ny - 1))
    sigma_y = np.zeros((p.nx, p.ny - 1))

    # Apply mask
    mask = has_cell

    # Compute sigma_x and sigma_y only where mask is True
    sf = stress_fraction_slice[mask]
    lw = this_weight[mask]
    up1 = up[:, :, 1][mask]
    lup0 = left_up[:, :, 0][mask]
    rup0 = right_up[:, :, 0][mask]
    lup1 = left_up[:, :, 1][mask]
    rup1 = right_up[:, :, 1][mask]

    sigma_x[mask] = 0.5 * (lup0 + rup0) + 0.5 * (1 - sf) * (lup1 - rup1)
    sigma_y[mask] = lw + sf * up1 + 0.5 * (1 - sf) * (lup1 + rup1) + 0.5 * (lup0 - rup0)

    # Update sigma
    sigma[:, :-1, 0][mask] = sigma_x[mask]
    sigma[:, :-1, 1][mask] = sigma_y[mask]

    return sigma


def get_mu(sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.nan_to_num(np.abs(sigma[:, :, 0]) / sigma[:, :, 1], nan=0.0, posinf=1e30, neginf=0.0)
    return mu


def get_sigma_xx(sigma, p, last_swap=None):
    stress_fraction = calculate_stress_fraction(last_swap, p)
    K = 1.0 - stress_fraction
    sigma_xx = K * sigma[:, :, 1]
    return sigma_xx


def get_pressure(sigma, p, last_swap=None):
    sigma_xx = get_sigma_xx(sigma, p, last_swap)
    pressure = 0.5 * (sigma_xx + sigma[:, :, 1])
    return pressure


def get_deviatoric(sigma, p, last_swap=None):
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]
    sigma_xx = get_sigma_xx(sigma, p, last_swap)

    return np.sqrt(((sigma_yy - sigma_xx) / 2) ** 2 + sigma_xy**2)


def get_friction_angle(sigma, p, last_swap=None):
    # pressure = get_pressure(sigma, p, last_swap)
    # deviatoric = get_deviatoric(sigma, p, last_swap)
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]
    sigma_xx = get_sigma_xx(sigma, p, last_swap)

    sigma_m = (sigma_xx + sigma_yy) / 2.0
    # sigma_d = np.sqrt(((sigma_yy - sigma_xx) / 2) ** 2 + sigma_xy**2)
    radius = np.sqrt(sigma_xy**2 + (sigma_xx - sigma_m) ** 2)  # radius of Mohr's circle

    with np.errstate(divide="ignore", invalid="ignore"):
        friction_angle = np.degrees(np.arcsin(radius / sigma_m))

    return friction_angle
