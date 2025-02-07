import subprocess
import os
import warnings
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

from PIL import Image
from HGD import operators
from HGD import stress

# _video_encoding = ["-c:v", "libx265", "-preset", "fast", "-crf", "28", "-tag:v", "hvc1"] # nice small file sizes
_video_encoding = [
    "-c:v",
    "libx264",
    "-preset",
    "slow",
    "-profile:v",
    "high",
    "-level:v",
    "4.0",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "22",
]  # powerpoint compatible

_dpi = 10
# plt.rcParams["figure.dpi"] = _dpi


def is_ffmpeg_installed():
    try:
        result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except FileNotFoundError:
        return False


cdict = {
    "red": ((0.0, 1.0, 1.0), (0.25, 1.0, 1.0), (0.5, 1.0, 1.0), (0.75, 0.902, 0.902), (1.0, 0.0, 0.0)),
    "green": (
        (0.0, 0.708, 0.708),
        (0.25, 0.302, 0.302),
        (0.5, 0.2392, 0.2392),
        (0.75, 0.1412, 0.1412),
        (1.0, 0.0, 0.0),
    ),
    "blue": (
        (0.0, 0.4, 0.4),
        (0.25, 0.3569, 0.3569),
        (0.5, 0.6078, 0.6078),
        (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
}
orange_blue_cmap = LinearSegmentedColormap("grainsize", cdict, 256)
orange_blue_cmap.set_bad("w", 1.0)
orange_blue_cmap.set_under("w", 1.0)
grey = cm.get_cmap("gray")
grey.set_bad("w", 0.0)
bwr = cm.get_cmap("bwr")
bwr.set_bad("k", 1.0)
bwr.set_under("k")
bwr.set_over("k")

bwr2 = cm.get_cmap("bwr")
bwr2.set_bad("k", 1.0)
bwr2.set_under("b")
bwr2.set_over("r")

colors = [(1, 0, 0), (0, 0, 1)]
cmap_name = "my_list"
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
inferno = cm.get_cmap("inferno")
inferno.set_bad("w", 0.0)
inferno_r = cm.get_cmap("inferno_r")
inferno_r.set_bad("w", 0.0)


def size_colormap():
    return orange_blue_cmap


global fig, summary_fig, triple_fig, quad_fig
fig = plt.figure(1)
summary_fig = plt.figure(2)
triple_fig = plt.figure(3)
quad_fig = plt.figure(4)

replacements = {
    "repose_angle": "φ",
    "mu": "μ",
    "nu_cs": "ν_cs",
    "alpha": "α",
}


def array_to_png_buffer(array, colormap, vmin=None, vmax=None):
    """
    Convert a NumPy array to a PNG image buffer using a colormap.

    Parameters:
    array (numpy.ndarray): The input array with shape (height, width).
    colormap (function): A function that maps array values to RGB tuples.

    Returns:
    io.BytesIO: A buffer containing the PNG image.
    """

    # Normalize the array to the range [0, 1]
    if vmin is None:
        vmin = np.nanmin(array)
    if vmax is None:
        vmax = np.nanmax(array)
    norm_array = (array - vmin) / (vmax - vmin)

    # Apply the colormap to the normalized array to get RGBA values
    rgba_image = colormap(norm_array)

    # Convert the RGBA image to RGB (ignoring the alpha channel)
    rgb_image = np.flipud((rgba_image * 255).astype(np.uint8).transpose(1, 0, 2))

    buffer = io.BytesIO()

    # Create a PIL image from the RGB array
    img = Image.fromarray(rgb_image)
    img.save(buffer, format="PNG")
    buffer.seek(0)  # Reset buffer position to the beginning

    return buffer


def replace_strings(text, replacements):
    """
    Replaces substrings in a given text based on a dictionary of replacements.

    Parameters:
    - text (str): The original string in which substrings will be replaced.
    - replacements (dict): A dictionary where keys are substrings to be replaced and values are the new substrings.

    Returns:
    - str: The modified string with all specified replacements applied.

    This function iterates over the key-value pairs in the replacements dictionary. For each pair, it replaces all occurrences of the key (old substring) in the text with the corresponding value (new substring). The function returns the modified text after all replacements have been made.
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def set_plot_size(p):
    global fig, summary_fig, triple_fig, quad_fig

    # wipe any existing figures
    for i in plt.get_fignums():
        plt.close(i)

    fig = plt.figure(1, figsize=[p.nx / _dpi, p.ny / _dpi])
    triple_fig = plt.figure(2, figsize=[p.nx / _dpi, 3 * p.ny / _dpi])
    quad_fig = plt.figure(3, figsize=[p.nx / _dpi, 4 * p.ny / _dpi])
    summary_fig = plt.figure(4)


def check_folders_exist(p):
    if not os.path.exists(p.folderName):
        os.makedirs(p.folderName)
    if len(p.save) > 0:
        if not os.path.exists(p.folderName + "data/"):
            os.makedirs(p.folderName + "data/")


def update(p, state, *args):
    (
        s,
        u,
        v,
        c,
        T,
        # p_count,
        # p_count_s,
        # p_count_l,
        # non_zero_nu_time,
        last_swap,
        chi,
        sigma,
        outlet,
        # surface_profile,
    ) = state

    check_folders_exist(p)

    if p.queue is not None:
        vmin = None
        vmax = None
        if p.view == "s":
            if p.mask_s_bar:
                nu = operators.get_solid_fraction(s)
                to_plot = np.ma.masked_where(nu < p.nu_cs / 4.0, operators.get_average(s))
            else:
                to_plot = operators.get_average(s)
            colorbar = orange_blue_cmap
            vmin = p.s_m
            vmax = p.s_M
        elif p.view == "nu":
            nu = operators.get_solid_fraction(s)
            to_plot = np.ma.masked_where(nu == 0, nu)
            colorbar = inferno_r
            vmin = 0
            vmax = 1
        elif p.view == "pressure":
            if sigma is None:
                sigma = stress.calculate_stress(s, last_swap, p)
            pressure = stress.get_pressure(sigma, p)
            to_plot = np.ma.masked_where(pressure == 0.0, pressure)
            colorbar = inferno_r
        elif p.view == "deviatoric":
            if sigma is None:
                sigma = stress.calculate_stress(s, last_swap, p)
            deviatoric = stress.get_deviatoric(sigma, p)
            to_plot = np.ma.masked_where(deviatoric == 0.0, deviatoric)
            colorbar = inferno_r
        elif p.view == "minus_45":
            minus_45 = np.sum(s <= 45e-6 * (~np.isnan(s)), axis=2)  # less than 45 microns
            nu = operators.get_solid_fraction(s)
            to_plot = minus_45 / nu
            to_plot = np.ma.masked_where(nu == 0, to_plot)
            colorbar = inferno_r
            vmin = 0
            vmax = 1
        elif p.view == "plus_150":
            plus_150 = np.sum(s >= 150e-6 * (~np.isnan(s)), axis=2)  # greater than 150 microns
            nu = operators.get_solid_fraction(s)
            to_plot = plus_150 / nu
            to_plot = np.ma.masked_where(nu == 0, to_plot)
            colorbar = inferno_r
            vmin = 0
            vmax = 1
        else:
            raise ValueError(f"Unknown view '{p.view}'")

        buffer = array_to_png_buffer(to_plot, colorbar, vmin, vmax)
        p.queue.put(buffer)

        # if p.queue_popup is not None:
        #     buffer = io.BytesIO()
        #     plot_outlet(outlet, buffer)
        #     p.queue_popup.put(buffer)

    else:
        if "s" in p.plot:
            if hasattr(p, "charge_discharge"):
                plot_s(s, p, *args)
            else:
                plot_s(s, p)
        if "nu" in p.plot:
            plot_nu(s, p)
        if "rel_nu" in p.plot:
            plot_relative_nu(s, p)
        if "U_mag" in p.plot:
            plot_u(s, u, v, p)
        if "gamma_dot" in p.plot:
            plot_gamma_dot(s, chi, p)
        if "c" in p.plot:
            plot_c(s, c, p)
        if "temperature" in p.plot:
            plot_T(s, T, p)
        # if "density_profile" in p.plot:
        #     plot_profile(x, nu_time_x, p)
        if "permeability" in p.plot:
            plot_permeability(s, p)
        if "stable" in p.plot:
            plot_stable(s, p)
        if "chi" in p.plot:
            plot_chi(chi, p)
        if "anisotropy" in p.plot:
            plot_anisotropy(last_swap, p)
        if "h" in p.plot:
            plot_h(s, p)
        if "stress" in p.plot:
            plot_stress(s, sigma, last_swap, p)
        if "sigma_xx" in p.plot:
            plot_sigma_xx(s, sigma, last_swap, p)
        if "sigma_yy" in p.plot:
            plot_sigma_yy(s, sigma, last_swap, p)
        if "sigma_xy" in p.plot:
            plot_sigma_xy(s, sigma, last_swap, p)
        if "pressure" in p.plot:
            plot_pressure(s, sigma, last_swap, p)
        if "deviatoric" in p.plot:
            plot_deviatoric(s, sigma, last_swap, p)
        if "footing" in p.plot:
            plot_footing(p)

        if "s" in p.save:
            save_s(s, p)
        if "s_bar" in p.save:
            save_s_bar(s, p)
        if "nu" in p.save:
            save_nu(s, p)
        if "rel_nu" in p.save:
            save_relative_nu(s, p)
        if "chi" in p.save:
            save_chi(chi, p)
        if "footing" in p.save:
            save_footing(s, p)

        # if "U_mag" in p.save:
        #     save_u(s, u, v, p)
        if "permeability" in p.save:
            save_permeability(s, p)
        if "stress" in p.save:
            save_stress(s, sigma, last_swap, p)
        if "last_swap" in p.save:
            save_last_swap(last_swap, p)
        if "concentration" in p.save:
            save_c(c, p)
        if "outlet" in p.save:
            np.savetxt(p.folderName + "data/outlet.csv", np.array(outlet), delimiter=",")
        # if "temperature" in p.save:
        #     np.savetxt(p.folderName + "outlet_T.csv", np.array(outlet_T), delimiter=",")
        if "velocity" in p.save:
            save_velocity(u, v, p)
        if "charge_discharge" in p.save:
            c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l)
        if "col_depth" in p.save:
            get_col_depth(p, s)
        if "surface_profiles" in p.save:
            np.save(p.folderName + "data/surface_profiles.npy", surface_profile)


def plot_u_time(y, U, nu_time, p):
    plt.figure(summary_fig)

    U = np.ma.masked_where(nu_time < 0.2, U)
    # U = np.amax(U) - U

    plt.clf()
    plt.pcolormesh(U.T)  # ,cmap='bwr',vmin=-amax(abs(U))/2,vmax=amax(abs(U))/2)
    plt.colorbar()
    plt.savefig(p.folderName + "u_time.png")

    plt.clf()
    u_y = np.mean(U[p.nt // 2 :], 0)
    # gamma_dot_y = gradient(u_y,dy)
    # plt.plot(gamma_dot_y,y,'r')
    plt.plot(u_y, y, "b")
    plt.xlabel("Average horizontal velocity (m/s)")
    plt.ylabel("Height (m)")
    plt.savefig(p.folderName + "u_avg.png")

    np.save(p.folderName + "data/u_y.npy", np.ma.filled(u_y, np.nan))
    np.save(p.folderName + "data/nu.npy", np.mean(nu_time[p.nt // 2 :], axis=0))


def plot_gamma_dot(s, chi, p):
    solid_fraction = operators.get_solid_fraction(s)
    s_bar = operators.get_average(s)
    if chi is None:
        gamma_dot = np.zeros([p.nx, p.ny])
    else:
        # chi_mag = np.sqrt(chi[:, :, 0] ** 2 + chi[:, :, 1] ** 2)
        gamma_dot = chi * solid_fraction * np.sqrt(p.g / s_bar)

    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, gamma_dot.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "gamma_dot_" + str(p.tstep).zfill(6) + ".png")


def plot_chi(chi, p):
    plt.figure(fig, layout="constrained")
    plt.clf()
    plt.pcolormesh(p.x, p.y, chi.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(fraction=0.3)
    plt.savefig(p.folderName + "chi_" + str(p.tstep).zfill(6) + ".png")


def save_chi(chi, p):
    np.save(p.folderName + "data/chi_" + str(p.tstep).zfill(6) + ".npy", chi)


def plot_stable(s, p):
    plt.figure(fig)

    slope = np.zeros([p.nx, p.ny, 2])
    solid = np.zeros([p.nx, p.ny])
    for i in range(1, p.nx - 1):
        for j in range(p.ny):
            slope[i, j, 0] = operators.stable_slope(s, i, j, i - 1, p)
            slope[i, j, 1] = operators.stable_slope(s, i, j, i + 1, p)

    for i in range(p.nx):
        for j in range(p.ny):
            solid[i, j] = operators.locally_solid(s, i, j, p)

    nu = operators.get_solid_fraction(s)
    empty = operators.empty_nearby(nu, p)

    for f in [
        [slope[:, :, 0], "slope_right"],
        [slope[:, :, 1], "slope_left"],
        [solid, "solid"],
        [empty, "empty"],
    ]:
        plt.clf()
        plt.pcolormesh(p.x, p.y, f[0].T, cmap=inferno)
        plt.axis("off")
        plt.xlim(p.x[0], p.x[-1])
        plt.ylim(p.y[0], p.y[-1])
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(p.folderName + f[1] + f"_{p.t}.png")


def plot_s_bar(y, s_bar, nu_time, p):
    plt.figure(summary_fig)

    if p.mask_s_bar:
        masked_s_bar = np.ma.masked_where(nu_time < p.nu_cs / 10.0, s_bar)
    else:
        masked_s_bar = s_bar

    plt.clf()
    plt.pcolormesh(
        np.linspace(0, p.t_f, p.nt), y, masked_s_bar.T, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M
    )
    plt.colorbar()
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.savefig(p.folderName + "s_bar.png")
    np.save(p.folderName + "data/s_bar.npy", s_bar.T)

    plt.clf()
    plt.pcolormesh(np.linspace(0, p.t_f, p.nt), y, nu_time.T, cmap=inferno, vmin=0, vmax=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Height (m)")
    plt.colorbar()
    plt.savefig(p.folderName + "nu.png")


def plot_sigma_xx(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    # sigma_xx = stress.get_sigma_xx(sigma, p)
    # sigma_xx = np.ma.masked_where(sigma_xx == 0.0, sigma_xx)
    sigma_xx = np.ma.masked_where(sigma[:, :, 2] == 0.0, sigma[:, :, 2])
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, sigma_xx.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "sigma_xx_" + str(p.tstep).zfill(6) + ".png")


def plot_sigma_yy(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    sigma_yy = np.ma.masked_where(sigma[:, :, 1] == 0.0, sigma[:, :, 1])
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, sigma_yy.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "sigma_yy_" + str(p.tstep).zfill(6) + ".png")


def plot_sigma_xy(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    sigma_xy = np.ma.masked_where(sigma[:, :, 0] == 0.0, sigma[:, :, 0])
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, sigma_xy.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "sigma_xy_" + str(p.tstep).zfill(6) + ".png")


def plot_rel_mu(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    mu = np.ma.masked_where(sigma[:, :, 2] == 0.0, sigma[:, :, 2])
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, (mu / p.mu).T, cmap=bwr, vmin=0, vmax=2)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "rel_mu_" + str(p.tstep).zfill(6) + ".png")


def plot_anisotropy(last_swap, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a = np.nanmean(last_swap, axis=2)  # -1 for lef/right, 1 for up/down
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, a.T, cmap=bwr, vmin=-1, vmax=1)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "anisotropy_" + str(p.tstep).zfill(6) + ".png")


def plot_stress(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    pressure = stress.get_pressure(sigma, p, last_swap)
    deviatoric = stress.get_deviatoric(sigma, p, last_swap)
    fr = stress.get_friction_angle(sigma, p, last_swap)
    diff = stress.get_difference(sigma, p, last_swap)

    plt.figure(quad_fig)
    plt.clf()
    plt.subplot(411)
    plt.pcolormesh(
        p.x,
        p.y,
        pressure.T,
        # cmap="bwr",
        vmin=0,
        vmax=pressure.max(),
    )
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.7, location="top", pad=0.05)

    plt.subplot(412)
    plt.pcolormesh(p.x, p.y, deviatoric.T, vmin=0, vmax=deviatoric.max())
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)

    plt.subplot(413)
    plt.pcolormesh(p.x, p.y, fr.T, vmin=0, vmax=2 * p.repose_angle, cmap=bwr2)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)

    plt.subplot(414)
    plt.pcolormesh(p.x, p.y, diff.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "stress_" + str(p.tstep).zfill(6) + ".png")


def plot_pressure(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    pressure = stress.get_pressure(sigma, p)
    pressure = np.ma.masked_where(pressure == 0.0, pressure)
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, pressure.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "pressure_" + str(p.tstep).zfill(6) + ".png")


def plot_deviatoric(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    q = stress.get_deviatoric(sigma, p)
    q = np.ma.masked_where(q == 0.0, q)
    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, q.T)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.savefig(p.folderName + "deviatoric_" + str(p.tstep).zfill(6) + ".png")


def save_coordinate_system(p):
    check_folders_exist(p)
    np.savetxt(p.folderName + "data/x.csv", p.x, delimiter=",")
    np.savetxt(p.folderName + "data/y.csv", p.y, delimiter=",")


def c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l):
    np.save(p.folderName + "data/nu_non_zero_avg.npy", non_zero_nu_time)
    if p.gsd_mode == "mono":
        np.save(p.folderName + "data/cell_count.npy", p_count)
    elif p.gsd_mode == "bi":
        np.save(p.folderName + "data/cell_count_s.npy", p_count_s)
        np.save(p.folderName + "data/cell_count_l.npy", p_count_l)


def get_col_depth(p, s):
    den = 1 - np.mean(np.isnan(s), axis=2)
    ht = []
    for w in range(p.nx):
        if np.mean(den[w]) > 0:
            ht.append(np.max(np.nonzero(den[w])))
        else:
            ht.append(0)
    np.save(p.folderName + "data/each_col_ht.npy", ht)


def kozeny_carman(s):
    sphericity = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        porosity = np.mean(np.isnan(s), axis=2)
        s_bar = operators.get_average(s)
        permeability = sphericity**2 * (porosity**3) * s_bar**2 / (180 * (1 - porosity) ** 2)
    return permeability


def plot_permeability(s, p):
    """
    Calculate and save the permeability of the domain at time t.
    """
    permeability = kozeny_carman(s)

    plt.figure(fig)
    plt.clf()
    plt.pcolormesh(p.x, p.y, permeability.T, cmap=inferno)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "permeability_" + str(p.tstep).zfill(6) + ".png")


def save_permeability(s, p):
    permeability = kozeny_carman(s)
    np.savetxt(
        p.folderName + "data/permeability_" + str(p.tstep).zfill(6) + ".csv", permeability, delimiter=","
    )


def plot_s(s, p, t, *args):
    plt.figure(fig)
    plt.clf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        s_plot = np.nanmean(s, axis=2).T
    s_plot = np.ma.masked_where(np.isnan(s_plot), s_plot)
    if p.mask_s_bar:
        nu = operators.get_solid_fraction(s).T
        s_plot = np.ma.masked_where(nu < p.nu_cs / 4.0, s_plot)

    if p.gsd_mode == "fbi":
        plt.pcolormesh(
            p.x,
            p.y,
            s_plot,
            cmap=orange_blue_cmap,
            vmin=p.s_m - (p.s_m / 100),
            vmax=(p.Fr * p.s_M) + ((p.Fr * p.s_M) / 100),
        )
    else:
        plt.pcolormesh(p.x, p.y, s_plot, cmap=orange_blue_cmap, vmin=p.s_m, vmax=p.s_M)
        # plt.colorbar()

    if p.internal_geometry:
        for i in p.internal_geometry["perf_pts"]:
            plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    ticks = np.linspace(p.s_m, p.s_M, 3, endpoint=True)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01, ticks=ticks)
    plt.savefig(p.folderName + "s_" + str(p.tstep).zfill(6) + ".png")


def save_s(s, p):
    np.save(p.folderName + "data/s_" + str(p.tstep).zfill(6) + ".npy", s)


def save_s_bar(s, p):
    np.save(p.folderName + "data/s_bar_" + str(p.tstep).zfill(6) + ".npy", operators.get_average(s))


def plot_nu(s, p):
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    nu = np.ma.masked_where(nu == 0, nu)
    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    plt.clf()

    plt.pcolormesh(p.x, p.y, nu, cmap=inferno_r, vmin=0, vmax=1)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)
    plt.savefig(p.folderName + "nu_" + str(p.tstep).zfill(6) + ".png")


def save_nu(s, p):
    np.save(p.folderName + "data/nu_" + str(p.tstep).zfill(6) + ".npy", operators.get_solid_fraction(s))


def save_relative_nu(s, p):
    np.save(
        p.folderName + "data/nu_" + str(p.tstep).zfill(6) + ".npy", operators.get_solid_fraction(s) / p.nu_cs
    )


def plot_relative_nu(s, p):
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2).T
    nu /= p.nu_cs
    if p.internal_geometry:
        nu = np.ma.masked_where(p.boundary.T, nu)
    nu = np.ma.masked_where(nu == 0, nu)
    plt.clf()
    plt.pcolormesh(p.x, p.y, nu, cmap=bwr, vmin=0, vmax=2)
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
    plt.savefig(p.folderName + "rel_nu_" + str(p.tstep).zfill(6) + ".png")


def plot_u(s, u, v, p):
    plt.figure(fig)
    # mask = mean(isnan(s),axis=2) > 0.95
    # u = ma.masked_where(mask,u/sum(isnan(s),axis=2)).T
    # v = ma.masked_where(mask,v/sum(isnan(s),axis=2)).T
    # u = u.T
    # v = v.T

    u = np.nanmean(u, axis=2).T
    v = np.nanmean(v, axis=2).T

    if p.lagrangian:
        u = np.amax(u) - u  # subtract mean horizontal flow
        # v = np.amax(v) - v  # subtract mean horizontal flow

    plt.clf()
    # plt.quiver(X,Y,u,v)
    # print(u)
    plt.pcolormesh(p.x, p.y, u, vmin=-np.amax(np.abs(u)), vmax=np.amax(np.abs(u)), cmap="bwr")
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.savefig(p.folderName + "u_" + str(p.tstep).zfill(6) + ".png")

    plt.clf()
    # plt.quiver(X,Y,u,v)
    plt.pcolormesh(p.x, p.y, v, vmin=-np.amax(np.abs(v)), vmax=np.amax(np.abs(v)), cmap="bwr")
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "v_" + str(p.tstep).zfill(6) + ".png")

    U = np.sqrt(u**2 + v**2)
    plt.clf()
    plt.pcolormesh(
        p.x, p.y, np.ma.masked_where(p.boundary.T, U), vmin=0, vmax=np.amax(np.abs(U)), cmap=inferno
    )
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
    plt.savefig(p.folderName + "U_mag_" + str(p.tstep).zfill(6) + ".png")


def plot_c(s, c, p):
    # print(np.unique(c))
    plt.figure(fig)

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm

    plt.clf()
    plt.pcolormesh(
        p.x, p.y, np.ma.masked_where(mask, np.nanmean(c, axis=2)).T, cmap=inferno, vmin=0, vmax=p.num_cycles
    )
    if p.internal_geometry:
        if p.internal_geometry["perf_plate"]:
            for i in p.internal_geometry["perf_pts"]:
                plt.plot([p.x[i], p.x[i]], [p.y[0], p.y[-1]], "k--", linewidth=10)
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    if p.plot_colorbar:
        plt.colorbar(shrink=0.8, location="top", pad=0.01)  # ,ticks = ticks)
        # np.save(p.folderName + "c_" + str(p.tstep).zfill(6) + ".npy", np.nanmean(c, axis=2))
    plt.savefig(p.folderName + "c_" + str(p.tstep).zfill(6) + ".png")


def save_c(c, p):
    np.save(p.folderName + "data/c_" + str(p.tstep).zfill(6) + ".npy", np.nanmean(c, axis=2))


def save_stress(s, sigma, last_swap, p):
    if sigma is None:
        sigma = stress.calculate_stress(s, last_swap, p)
    np.save(p.folderName + "data/sigma_" + str(p.tstep).zfill(6) + ".npy", sigma)


def save_last_swap(last_swap, p):
    np.save(p.folderName + "data/last_swap_" + str(p.tstep).zfill(6) + ".npy", last_swap)


def save_velocity(u, v, p):
    np.save(p.folderName + "data/u_" + str(p.tstep).zfill(6) + ".npy", np.mean(u, axis=2))
    np.save(p.folderName + "data/v_" + str(p.tstep).zfill(6) + ".npy", np.mean(v, axis=2))


def plot_outlet(outlet, output):
    plt.figure(summary_fig)

    plt.clf()
    plt.plot(outlet)
    plt.xlabel("time")
    plt.ylabel("outflow")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    if type(output) is str:
        plt.savefig(output + "outflow.png")
    else:
        plt.savefig(output, format="png")


def save_footing(s, p, t, debug=True):
    if t == 0:
        with open(p.folderName + "data/footing.csv", "w") as f:
            f.write("time,depth,load\n")
    with open(p.folderName + "data/footing.csv", "a") as f:
        nu = operators.get_solid_fraction(s)

        nu_slice = nu[p.nx // 2, :]
        empty_ish = nu_slice < p.nu_1
        y1_index = np.where(empty_ish)[0][0]  # INDEX
        y1 = p.y[y1_index]
        slope = p.dy / (nu_slice[y1_index - 1] - nu_slice[y1_index])
        top = y1 + slope * (p.nu_1 / 2.0 - nu_slice[y1_index])

        depth = p.H - top
        current_load = p.point_load * p.t
        f.write(f"{t},{depth},{current_load}\n")

    if debug:
        plt.figure(fig)
        plt.clf()
        plt.pcolormesh(p.x, p.y, nu.T, cmap=inferno, vmin=0, vmax=1)
        plt.plot([p.x[p.nx // 2]], [top], "go")
        plt.savefig(p.folderName + "footing_" + str(p.tstep).zfill(6) + ".png")


def plot_footing(p):
    if os.path.exists(p.folderName + "data/footing.csv"):
        data = np.loadtxt(p.folderName + "data/footing.csv", delimiter=",", skiprows=1)

        if len(data.shape) < 2:
            return

        plt.figure(summary_fig, layout="constrained")
        plt.clf()
        depth = data[:, 1] - data[0, 1]
        plt.plot(depth, data[:, 2], "k.")
        plt.xlabel("Depth, m")
        plt.ylabel("Load, N")
        plt.savefig(p.folderName + "footing.png")


def plot_profile(x, nu_time_x, p):
    plt.figure(summary_fig)

    plt.clf()
    plt.pcolormesh(x, np.linspace(0, p.t_f, p.nt), nu_time_x)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig(p.folderName + "collapse_profile.png")


def plot_T(s, T, p):
    plt.figure(fig)

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm
    plt.clf()
    plt.pcolormesh(
        p.x,
        p.y,
        np.ma.masked_where(mask, np.nanmean(T, axis=2)).T,
        cmap=bwr,
        vmin=p.boundary_temperature,
        vmax=p.inlet_temperature,
    )
    plt.axis("off")
    plt.xlim(p.x[0], p.x[-1])
    plt.ylim(p.y[0], p.y[-1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "T_" + str(p.tstep).zfill(6) + ".png")


def plot_h(s, p):
    """
    Show the relative 'height' of the grains in each cell. Used for diagnostic purposes only, otherwise not that useful.
    """
    plt.figure(fig)
    nu = 1 - np.mean(np.isnan(s), axis=2)
    h = nu / p.nu_cs

    plt.clf()
    ax = plt.gca()

    # Loop through each value in the 2D array and create a rectangle
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            # Create a rectangle
            rect = plt.Rectangle((i, j), 1, h[i, j], color="k", alpha=1)

            # Add the rectangle to the plot
            ax.add_patch(rect)

    # Set the limits of the plot to fit all rectangles
    ax.set_xlim(0, h.shape[0])
    ax.set_ylim(0, h.shape[1])

    # Display the plot
    # plt.gca().invert_yaxis() # Optional: to invert the y-axis to match array indexing
    # plt.show()

    plt.axis("off")
    plt.xlim(0, h.shape[0])
    plt.ylim(0, h.shape[1])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(p.folderName + "h_" + str(p.tstep).zfill(6) + ".png", dpi=100)


def make_video(p):
    if is_ffmpeg_installed:
        fname = p.folderName.split("/")[-2]
        nice_name = "=".join(fname.rsplit("_", 1))
        nice_name = replace_strings(nice_name, replacements)
        subtitle = f"drawtext=text='{nice_name}':x=(w-text_w)/2:y=H-th-10:fontsize=10:fontcolor=white:box=1:boxcolor=black@0.5"
        # fps = p.save_inc / p.dt
        # print(f"Making video at {fps} fps, {p.save_inc} frames per cycle, {p.dt} s per frame")
        for i, video in enumerate(p.videos):
            cmd = [
                "ffmpeg",
                "-y",
                # "-r",
                # f"{fps}",
                "-pattern_type",
                "glob",
                "-i",
                f"{p.folderName}/{video}_*.png",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                #  "-c:v", "libx264", "-pix_fmt", "yuv420p"
            ]
            # add a title to the last video so we know whats going on
            if i == len(p.videos) - 1:
                # cmd.extend(["-vf", subtitle])
                cmd[-1] += f",{subtitle}"
            cmd.extend(["-r", "30", *_video_encoding, f"{p.folderName}/{video}_video.mp4"])
            subprocess.run(
                cmd,
                # stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    else:
        print("ffmpeg not installed, cannot make videos")


def stack_videos(paths, name, videos):
    if is_ffmpeg_installed:
        for video in videos:
            cmd = [
                "ffmpeg",
                "-y",
            ]
            for path in paths:
                cmd.extend(["-i", f"{path}/{video}_video.mp4"])
            pad_string = ""
            for i in range(len(paths)):
                pad_string += f"[{i}]pad=iw+5:color=black[left];[left][{i+1}]"

            cmd.extend(
                [
                    *_video_encoding,
                    # "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-filter_complex",
                    # f"{pad_string}hstack=inputs={len(paths)}",
                    f"hstack=inputs={len(paths)}",
                    f"{video}_videos.mp4",
                ]
            )
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not result.returncode == 0:
                print(f"Error stacking first pass {video} videos")
                print("Error message:", result.stderr)

        if len(videos) > 1:
            cmd = ["ffmpeg", "-y"]
            for video in videos:
                cmd.extend(["-i", f"{video}_videos.mp4"])
            cmd.extend(
                [
                    *_video_encoding,
                    # "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-filter_complex",
                    f"vstack=inputs={len(videos)}",
                    f"output/{name}.mp4",
                ]
            )
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                cmd = ["rm"]
                for video in videos:
                    cmd.append(f"{video}_videos.mp4")
                subprocess.run(cmd)
            else:
                print("Error stacking second pass videos")
                print("Error message:", result.stderr)
    else:
        print("ffmpeg not installed, cannot make videos")


# def generate_colormap_image(width, height, colormap_name='viridis'):
#     # Create a gradient image (2D array)
#     gradient = np.linspace(0, 1, width * height).reshape(height, width)

#     # Get the colormap
#     colormap = cm.get_cmap(colormap_name)

#     # Apply the colormap to the gradient
#     colored_img = colormap(gradient)

#     # Convert the image to 8-bit (0-255)
#     colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

#     # Convert the NumPy array to a PIL image
#     pil_img = Image.fromarray(colored_img)

#     # Save the image to a buffer
#     buffer = io.BytesIO()
#     pil_img.save(buffer, format='PNG')
#     buffer.seek(0)

#     return buffer

# # Generate the colormap image and save it to a file for later use (optional)
# buffer = generate_colormap_image(256, 256)
# with open('colormap.png', 'wb') as f:
#     f.write(buffer.read())
