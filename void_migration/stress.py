import numpy as np
import warnings


# Implementing Eq 36 and 37 from:
# Models of stress fluctuations in granular media
# P. Claudin and J.-P. Bouchaud, M. E. Cates and J. P. Wittmer
# NOTE: in our geometry tan(psi) = 1 (because dx=dy)

# Since tan(psi) = 1, we have that c_0^2 = 1 - stress_fraction

# ISOTROPIC STRESS
# c_0^2 = lateral earth pressure coefficient, K
# Using Jaky's formula, K = 1 - sin(phi), where phi is the repose angle (actually effective angle of internal friction)
# stress_fraction = sin(phi)
# or if we have a different value for K, we have that
# stress_fraction = 1 - K

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


def calculate_stress(s, last_swap, p):
    if p.stress_mode == "isotropic":
        stress_fraction = np.sin(np.radians(p.repose_angle))
        stress_fraction = np.full([p.nx, p.ny], stress_fraction)
    elif p.stress_mode == "anisotropic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.abs(
                np.nanmean(last_swap, axis=2)
            )  # between 0 and 1, 0 for isotropic, 1 for fully anisotropic
            K_p = (1 + np.sin(np.radians(p.repose_angle))) / (1 - np.sin(np.radians(p.repose_angle)))
            K_a = 1 / K_p
            K = K_p * (K_a / K_p) ** ((a + 1) / 2)
            stress_fraction = 1 - K

    sigma = np.zeros([p.nx, p.ny, 2])  # sigma_xy, sigma_yy
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.dx * p.dy * p.g
    for j in range(p.ny - 2, -1, -1):
        for i in range(p.nx):
            if np.sum(~np.isnan(s[i, j, :])) > 0:
                this_weight = np.sum(~np.isnan(s[i, j, :])) * weight_of_one_cell
                up = sigma[i, j + 1]
                if i == 0:
                    right_up = sigma[i + 1, j + 1]
                    # left_up = [0, 0] # walls carry no load!!
                    # left_up = right_up  # FIXME: no gradient at boundary???
                    if p.wall_friction_angle == 0 or p.repose_angle == 0:
                        left_up = [0, 0]
                    else:
                        left_up = (
                            p.wall_friction_angle / p.repose_angle * right_up
                        )  # FIXME: somehow scale the friction. should at least be with mu and not angle, even if this somehow miraculously works

                elif i == p.nx - 1:
                    left_up = sigma[i - 1, j + 1]
                    # right_up = [0, 0]
                    # right_up = left_up  # FIXME: no gradient at boundary???
                    if p.wall_friction_angle == 0 or p.repose_angle == 0:
                        right_up = [0, 0]
                    else:
                        right_up = p.wall_friction_angle / p.repose_angle * left_up
                else:
                    left_up = sigma[i - 1, j + 1]
                    right_up = sigma[i + 1, j + 1]
                sigma[i, j, 0] = 0.5 * (left_up[0] + right_up[0]) + 0.5 * (1 - stress_fraction[i, j]) * (
                    left_up[1] - right_up[1]
                )
                sigma[i, j, 1] = (
                    this_weight
                    + stress_fraction[i, j] * up[1]
                    + 0.5 * (1 - stress_fraction[i, j]) * (left_up[1] + right_up[1])
                    + 0.5 * (left_up[0] - right_up[0])
                )

    # import matplotlib.pyplot as plt

    # plt.figure(45, figsize=(10, 10))
    # plt.clf()
    # plt.ion()
    # plt.imshow(a, origin="upper", cmap="bwr", vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.pause(1e-3)

    return sigma


def get_mu(sigma):
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = np.nan_to_num(np.abs(sigma[:, :, 0]) / sigma[:, :, 1], nan=0.0, posinf=1e30, neginf=0.0)
    return mu


def get_sigma_xx(sigma, p):
    # NOTE: only implemented isotropic case
    K = 1.0 - np.sin(np.radians(p.repose_angle))
    sigma_xx = K * sigma[:, :, 1]
    return sigma_xx


def get_pressure(sigma, p):
    sigma_xx = get_sigma_xx(sigma, p)
    pressure = 0.5 * (sigma_xx + sigma[:, :, 1])
    return pressure


def get_deviatoric(sigma, p):
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]
    sigma_xx = get_sigma_xx(sigma, p)

    q = np.sqrt(((sigma_yy - sigma_xx) / 2) ** 2 + sigma_xy**2)
    return q
