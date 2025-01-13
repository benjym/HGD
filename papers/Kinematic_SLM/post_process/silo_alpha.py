import os

# from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps

# import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file

# from void_migration.plotter import size_colormap
from void_migration import cycles

# from mpl_toolkits.axes_grid1 import Grid
# from scipy.integrate import cumtrapz

from scipy.optimize import curve_fit


def gaussian(x, A, sigma):
    return A * np.exp(-0.5 * (x / sigma) ** 2)


plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

data_dir = "/Volumes/LTS/DynamiX/Hopper/"
exps = ["Jasmine_Rice_10_mm_opening", "Lentil_10_mm_opening", "Glass3mm_10_mm_opening"]
# exps = ["Jasmine_Rice_12_mm_opening", "Lentil_12_mm_opening", "Glass3mm_12_mm_opening"]
# exps = ["Jasmine_Rice_15_mm_opening", "Lentil_15_mm_opening", "Glass3mm_15_mm_opening"]
s_bar = [2.3, 4.5, 2.8]  # in mm, pulled from peak values in energy plot in our paper

with open("papers/Kinematic_SLM/json/silo_alpha.json5") as f:
    dict, p = load_file(f)

y = np.linspace(0, p.H, p.ny)
p.dy = p.H / p.ny
x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
# W = x[-1] - x[0]
W = p.H / p.aspect_ratio_y
p.dx = x[1] - x[0]

y_avg = 3

# H_max = 0.4698630137

cmap = colormaps["inferno"]
cmap.set_under("black")
cmap.set_over(cmap(0.99999))
cmap.set_bad("black")


fig = plt.figure(figsize=[3.31894680556, 5.2])
grid = fig.subplots(4, 3)
alphas = p.alpha
for i, val in enumerate(alphas):
    p.alpha = val
    p.update_before_time_march(cycles)

    t_plot = 0.667 * p.t_f
    delta = 0.333  # fraction of t_f to average over
    nt_plot = int(t_plot / p.dt / p.save_inc) * p.save_inc
    dt = int(p.t_f / p.dt * delta / p.save_inc) * p.save_inc

    for j, t in enumerate(range(nt_plot - dt, nt_plot + dt, p.save_inc)):
        this_nu = np.load(f"output/silo_alpha/alpha_{val}/data/nu_{str(t).zfill(6)}.npy")
        this_u = np.mean(np.load(f"output/silo_alpha/alpha_{val}/data/u_{str(t).zfill(6)}.npy"), axis=2)
        this_v = np.mean(np.load(f"output/silo_alpha/alpha_{val}/data/v_{str(t).zfill(6)}.npy"), axis=2)

        if j == 0:
            nu = this_nu.copy()
            u = this_u.copy()
            v = this_v.copy()
        else:
            nu += this_nu
            u += this_u
            v += this_v

    nu /= 2 * dt / p.save_inc
    u /= 2 * dt / p.save_inc
    v /= 2 * dt / p.save_inc

    u_mag = np.sqrt(u**2 + v**2)

    exp_data_u = np.load(f"{data_dir}{exps[i]}/u_mean.npy")
    exp_data_v = np.load(f"{data_dir}{exps[i]}/v_mean.npy")
    exp_data_x = np.linspace(x[0], x[-1], exp_data_u.shape[0])
    exp_data_y = np.linspace(y[0], 0.47, exp_data_u.shape[1])[::-1]

    exp_data_U = np.sqrt(exp_data_u**2 + exp_data_v**2)
    exp_data_U /= exp_data_U.max()
    im = grid[0, i].pcolormesh(exp_data_x, exp_data_y, exp_data_U.T, cmap=cmap, rasterized=True)
    grid[0, i].set_aspect("equal")
    grid[0, i].set_xticks([])
    grid[0, i].set_yticks([])

    # y_slice = 0.05
    dy_exp = exp_data_y[0] - exp_data_y[1]
    tt = []
    # for y_test in range(len(exp_data_y)):
    for y_test in [0.04]:
        y_slice = y_test * dy_exp

        y_slice_arg = np.argmin(np.abs(exp_data_y - y_slice))
        grid[3, i].plot(
            exp_data_x, exp_data_U[:, y_slice_arg] / np.max(exp_data_U[:, y_slice_arg]), label="Exp.", c="r"
        )

        # cutoff_velocity = u_mag[p.nx // 2, 8]
        cutoff_velocity = u_mag.max() / 2  # / 8  # / (1.5 * val)
        u_mag_norm = u_mag / cutoff_velocity

        y_slice_arg = np.argmin(np.abs(y - y_slice))
        u_fit = u_mag_norm[:, y_slice_arg - y_avg : y_slice_arg + y_avg + 1]
        u_fit = u_fit.mean(axis=1)
        u_fit /= np.max(u_fit)
        grid[3, i].plot(x, u_fit, label="Sim.", c="b")

        grid[0, i].plot([x[0], x[-1]], [y_slice, y_slice], c="k", ls="--")

        # sigma = np.sqrt(2 * p.alpha * s_bar * y_slice)
        # analytic = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x / sigma)**2)
        # fit gaussian
        y_slice_arg = np.argmin(np.abs(exp_data_y - y_slice))
        popt, pcov = curve_fit(
            gaussian, exp_data_x, exp_data_U[:, y_slice_arg] / np.max(exp_data_U[:, y_slice_arg]), p0=[1, 0.1]
        )
        analytic = gaussian(exp_data_x, *popt)
        sigma = popt[1]
        alpha = sigma**2 / (2 * y_slice * s_bar[i] * 1e-3)
        grid[3, i].plot(
            exp_data_x, analytic / np.max(analytic), c="k", ls="--", label=rf"Fit"
        )  #: $\alpha$ = {alpha:0.3f}")

        grid[2, i].pcolormesh(
            x,
            y,
            nu.T,
            # u_mag.T,
            # vmin=1e3 * p.s_m,
            # vmax=1e3 * p.s_M,
            # vmin=p.outlet_nu,
            # vmax=p.nu_fill,
            cmap=cmap,
            rasterized=True,
            norm=colors.LogNorm(vmin=p.outlet_nu, vmax=p.nu_fill),
        )

        tt.append([y_slice, alpha])
    # tt = np.array(tt)
    # alpha_min_index = np.argmin(tt[:,1])
    # plt.figure(99, layout="constrained")
    # plt.suptitle(exps[i])
    # plt.subplot(121)
    # plt.pcolormesh(exp_data_x, exp_data_y, exp_data_U.T, cmap=cmap, rasterized=True)
    # plt.xlabel("x (m)")
    # plt.ylabel("y (m)")
    # plt.subplot(122)
    # plt.plot(tt[:,1], tt[:,0], label=exps[i])
    # plt.plot(tt[alpha_min_index, 1], tt[alpha_min_index, 0], "r.")
    # plt.plot([tt[alpha_min_index, 1], tt[alpha_min_index, 1]], [0, tt[alpha_min_index, 0]], "r--")
    # plt.plot([0, tt[alpha_min_index, 1]], [tt[alpha_min_index, 0], tt[alpha_min_index, 0]], "r--")
    # plt.text(tt[alpha_min_index, 1], tt[alpha_min_index, 0], f"({tt[alpha_min_index, 1]:0.3f}, {tt[alpha_min_index, 0]:0.3f})")
    # plt.ylabel("y (m)")
    # plt.xlabel(r"$\alpha$ (-)")
    # plt.xlim([0, 20])
    # plt.ylim([0, 0.47])
    # plt.savefig(f"silo_alpha_fit_{exps[i]}.pdf")
    # plt.close(99)

    im = grid[1, i].pcolormesh(
        x,
        y,
        u_mag_norm.T,
        # u_mag.T,
        # vmin=1e3 * p.s_m,
        # vmax=1e3 * p.s_M,
        vmax=1,
        cmap=cmap,
        rasterized=True,
        # norm=colors.LogNorm(),
    )
    # grid[1, i].set_ylim([0, 0.4698630137])  # match experimental conditions
    # plt.ylim(ymax=0.4698630137)
    grid[1, i].set_aspect("equal")
    plt.sca(grid[1, i])
    plt.ylim([0, 0.47])

    # except IndexError:
    #     print(f"Missing file for {val}")
    # except ValueError as e:
    #     print(f"Old data file for {val}")
    #     print(e)

    grid[1, i].set_xticks([])
    grid[1, i].set_yticks([])

    grid[2, i].set_xticks([])
    grid[2, i].set_yticks([])

    grid[3, i].set_xticks([])
    grid[3, i].set_yticks([])

# plt.sca(grid[0, 0])
# plt.xlabel("$x$ (m)", labelpad=1)
# plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
# plt.xticks([-W / 2, 0, W / 2])
# plt.yticks([0, 0.47])

plt.sca(grid[2, 0])
plt.xlabel("$x$ (m)", labelpad=1)
plt.ylabel("$y$ (m)", labelpad=-14)
plt.xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.2f}", "0", f"{W/2:0.2f}"])
plt.yticks([0, 0.47], labels=["0", f"{0.47:0.2f}"])
plt.ylim([0, 0.47])


bottom = 0.14
top = 0.98
left = 0.16
right = 0.85


colorbar_ax = fig.add_axes([right + 0.02, bottom, 0.02, top - bottom])
cb = plt.colorbar(im, cax=colorbar_ax, extend="max")
cb.set_label(r"$|\mathbf{\hat{u}}|$ (-)", labelpad=-7, y=0.75)
cb.set_ticks([0, 0.5, 1])


plt.sca(grid[3, 0])
plt.xlabel("$x$ (m)", labelpad=1)
plt.ylabel(r"$|\mathbf{\hat{u}}|$ (-)")
plt.xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.2f}", "0", f"{W/2:0.2f}"])
plt.sca(grid[3, 2])
plt.legend(bbox_to_anchor=(1.2, 0.0), frameon=False, handlelength=1)


plt.text(-0.6, 0.5, "Experiment", ha="center", va="center", transform=grid[0, 0].transAxes, rotation=90)
plt.text(-0.6, 0.5, "Simulation", ha="center", va="center", transform=grid[1, 0].transAxes, rotation=90)

plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.1, wspace=0.1)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/silo_alpha.pdf"))
