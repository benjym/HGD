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
    elif p.stress_mode == "isotropic":
        stress_fraction = np.zeros([p.nx, p.ny])
    elif p.stress_mode == "active":
        K_a = (1 - np.sin(np.radians(p.repose_angle))) / (1 + np.sin(np.radians(p.repose_angle)))
        stress_fraction = np.full([p.nx, p.ny], 1 - K_a)
    elif p.stress_mode == "passive":
        K_p = (1 + np.sin(np.radians(p.repose_angle))) / (1 - np.sin(np.radians(p.repose_angle)))
        stress_fraction = np.full([p.nx, p.ny], 1 - K_p)
    elif p.stress_mode == "anisotropic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.nanmean(last_swap, axis=2)  # 1 for up, -1 for left or right, 0 for isotropic
            a_scaled = np.abs((a + 1) / 2)  # between 0 and 1, 0 for isotropic, 1 for fully anisotropic

            K_a = (1 - np.sin(np.radians(p.repose_angle))) / (1 + np.sin(np.radians(p.repose_angle)))
            # K = K_p * (K_a / K_p) ** ((a + 1) / 2)
            K_iso = 1
            # K = K_iso * (K_a / K_iso) ** a_scaled
            K = (K_a - K_iso) * a_scaled + 1
            stress_fraction = np.full([p.nx, p.ny], 1 - K)
    else:
        raise ValueError("Unknown stress mode")

    return stress_fraction


def calculate_stress(s, last_swap, p):
    return calculate_stress_NEW(s, last_swap, p)
    # return calculate_stress_OLD(s, last_swap, p)


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
                down_has_mass = nu[*down] > 0
                down_left_has_mass = nu[*down_left] > 0
                down_right_has_mass = nu[*down_right] > 0
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

                lr_frac = 0.5
                p_total = 1.0

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
    pressure = get_pressure(sigma, p, last_swap)
    deviatoric = get_deviatoric(sigma, p, last_swap)

    with np.errstate(divide="ignore", invalid="ignore"):
        friction_angle = np.degrees(np.arcsin(deviatoric / pressure))

    return friction_angle
