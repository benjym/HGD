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
from scipy.ndimage import gaussian_filter


def gaussian(x, A, sigma):
    return A * np.exp(-0.5 * (x / sigma) ** 2)


plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

data_dir = "/Volumes/LTS/DynamiX/Hopper/"
exp = "Jasmine_Rice_10_mm_opening"
s_bar = 2.3e-3  # in m, pulled from peak values in energy plot in our paper
H = 0.2069  # top of radiograph

with open("papers/Kinematic_SLM/json/silo_alpha.json5") as f:
    dict, p = load_file(f)

# for nx in [21, 51, 101]:
# for nm in [100, 1000, 10000, 50000]:
for nx in [50]:
    for nm in [100, 1000]:
        p.nx = nx
        p.ny = p.aspect_ratio_y * p.nx
        p.nm = nm

        y = np.linspace(0, p.H, p.ny)
        p.dy = p.H / p.ny
        x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing

        # W = x[-1] - x[0]
        W = p.H / p.aspect_ratio_y
        p.dx = x[1] - x[0]

        cmap = colormaps["inferno"]
        cmap.set_under("black")
        cmap.set_over(cmap(0.99999))
        cmap.set_bad("black")

        fig = plt.figure(figsize=[3.31894680556, 4.0])
        grid = fig.subplots(2, 2)
        p.update_before_time_march(cycles)

        t_plot = 0.55 * p.t_f
        delta = 0.45
        # t_plot = 0.25*p.t_f
        # delta = 0.1

        nt_plot = int(t_plot / p.dt / p.save_inc) * p.save_inc
        dt = int(p.t_f / p.dt * delta / p.save_inc) * p.save_inc

        try:
            for j, t in enumerate(range(nt_plot - dt, nt_plot + dt, p.save_inc)):
                this_nu = np.load(f"output/silo_alpha/nx_{p.nx}/nm_{p.nm}/data/nu_{str(t).zfill(6)}.npy")
                # this_u = np.mean(np.load(f"output/silo_alpha/data/u_{str(t).zfill(6)}.npy"), axis=2)
                # this_v = np.mean(np.load(f"output/silo_alpha/data/v_{str(t).zfill(6)}.npy"), axis=2)
                this_u = np.load(f"output/silo_alpha/nx_{p.nx}/nm_{p.nm}/data/u_{str(t).zfill(6)}.npy")
                this_v = np.load(f"output/silo_alpha/nx_{p.nx}/nm_{p.nm}/data/v_{str(t).zfill(6)}.npy")

                if j == 0:
                    nu = this_nu.copy()
                    u = this_u.copy()
                    v = this_v.copy()
                else:
                    nu += this_nu
                    u += this_u
                    v += this_v
        except FileNotFoundError:
            completed = 100 * j / (2 * dt / p.save_inc)
            print(f"Missing data for nx={nx} and nm={nm}. Got up to {t}/{nt_plot + dt} ({completed:2.0f}%)")
            continue
        nu /= 2 * dt / p.save_inc
        u /= 2 * dt / p.save_inc
        v /= 2 * dt / p.save_inc

        u_mag = np.sqrt(u**2 + v**2)

        # u_mag = gaussian_filter(u_mag, 1)

        exp_data_u = np.load(f"{data_dir}{exp}/u_mean.npy")
        exp_data_v = np.load(f"{data_dir}{exp}/v_mean.npy")
        exp_data_x = np.linspace(x[0], x[-1], exp_data_u.shape[0])
        exp_data_y = np.linspace(y[0], H, exp_data_u.shape[1])[::-1]

        exp_data_U = np.sqrt(exp_data_u**2 + exp_data_v**2)
        exp_data_U /= exp_data_U.max()
        im = grid[0, 0].pcolormesh(exp_data_x, exp_data_y, exp_data_U.T, cmap=cmap, rasterized=True)
        grid[0, 0].set_aspect("equal")
        grid[0, 0].set_xticks([])
        grid[0, 0].set_yticks([])

        # y_slice = 0.05
        dy_exp = exp_data_y[0] - exp_data_y[1]
        y_slice = 0.07

        y_slice_arg = np.argmin(np.abs(exp_data_y - y_slice))
        grid[1, 1].plot(
            exp_data_x, exp_data_U[:, y_slice_arg] / np.max(exp_data_U[:, y_slice_arg]), label="Exp.", c="r"
        )

        # cutoff_velocity = u_mag[p.nx // 2, 8]
        # cutoff_velocity = u_mag.max() / 16  # / 8  # / (1.5 * val)
        cutoff_velocity = u_mag[:, np.argmin(np.abs(y - 0.03))].max()
        u_mag_norm = u_mag / cutoff_velocity

        y_slice_arg = np.argmin(np.abs(y - y_slice))
        y_avg = 0
        u_fit = u_mag_norm[:, y_slice_arg - y_avg : y_slice_arg + y_avg + 1]
        u_fit = u_fit.mean(axis=1)
        u_fit /= np.max(u_fit)
        grid[1, 1].plot(x, u_fit, label="Sim.", c="b")

        grid[0, 0].plot([x[0], x[-1]], [y_slice, y_slice], c="w", ls="--")

        # sigma = np.sqrt(2 * p.alpha * s_bar * y_slice)
        # analytic = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * (x / sigma)**2)
        # fit gaussian
        y_slice_arg = np.argmin(np.abs(exp_data_y - y_slice))
        popt, pcov = curve_fit(
            gaussian, exp_data_x, exp_data_U[:, y_slice_arg] / np.max(exp_data_U[:, y_slice_arg]), p0=[1, 0.1]
        )
        analytic = gaussian(exp_data_x, *popt)
        sigma = popt[1]
        alpha = sigma**2 / (2 * y_slice * s_bar)
        grid[1, 1].plot(
            exp_data_x, analytic / np.max(analytic), c="k", ls="--", label=rf"Fit"
        )  #: $\alpha$ = {alpha:0.3f}")

        # nu_im = grid[1, 1].pcolormesh(
        #     x,
        #     y,
        #     nu.T,
        #     # u_mag.T,
        #     # vmin=1e3 * p.s_m,
        #     # vmax=1e3 * p.s_M,
        #     vmin=p.outlet_nu,
        #     vmax=p.nu_fill,
        #     # vmin = 0,
        #     # vmax = 1,
        #     cmap=cmap,
        #     rasterized=True,
        #     # norm=colors.LogNorm(vmin=p.outlet_nu, vmax=p.nu_fill),
        # )

        im = grid[0, 1].pcolormesh(
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
        grid[0, 1].plot([x[0], x[-1]], [y_slice, y_slice], c="w", ls="--")

        # grid[1, i].set_ylim([0, 0.4698630137])  # match experimental conditions
        # plt.ylim(ymax=0.4698630137)
        for g in [grid[0, 0], grid[0, 1]]:  # , grid[1, 1]]:
            g.set_ylim([0, H])
            g.set_xlim([-W / 2, W / 2])
            g.set_aspect("equal")
            g.set_xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.3f}", "0", f"{W/2:0.3f}"])
            g.set_yticks([0, H], labels=["0", f"{H:0.2f}"])
            g.set_xlabel("$x$ (m)", labelpad=1)
            g.set_ylabel("$y$ (m)", labelpad=-14)

        grid[1, 1].set_xlim([-W / 2, W / 2])
        grid[1, 1].set_xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.3f}", "0", f"{W/2:0.3f}"])
        grid[1, 1].set_yticks([0, 1], labels=["0", "1"])
        grid[1, 1].set_xlabel("$x$ (m)", labelpad=1)
        grid[1, 1].set_ylabel(r"$|\mathbf{u}|/|\mathbf{u}(0,0.1)|$ (-)", labelpad=-3)
        # grid[1,0].legend(frameon=False, handlelength=1, fontsize=8, loc="upper right")

        bottom = 0.12
        top = 0.92
        left = 0.16
        right = 0.8
        hspace = 0.5
        bar_off = 0.08

        colorbar_ax = fig.add_axes([right + 0.05, 0.5 + 0.1, 0.02, (top - bottom) / 2.0 - bar_off])
        cb = plt.colorbar(im, cax=colorbar_ax, extend="max")
        cb.set_label(r"$|\mathbf{\hat{u}}|$ (-)", labelpad=-7, y=0.75)
        cb.set_ticks([0, 0.5, 1])

        tt = []
        for y_test in range(len(exp_data_y)):
            y_slice = y_test * dy_exp
            y_slice_arg = np.argmin(np.abs(exp_data_y - y_slice))
            popt, pcov = curve_fit(
                gaussian,
                exp_data_x,
                exp_data_U[:, y_slice_arg] / np.max(exp_data_U[:, y_slice_arg]),
                p0=[1, 0.1],
            )
            sigma = popt[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = sigma**2 / (2 * y_slice * s_bar)
            tt.append([y_slice, alpha])

        tt = np.array(tt)
        alpha_median = np.median(tt[:, 1])
        idx = np.argmin(np.abs(alpha_median - tt[:, 1]))

        grid[1, 0].plot(tt[:, 1], tt[:, 0], "r-")
        # grid[1,1].plot(tt[idx, 1], tt[idx, 0], ".")
        grid[1, 0].plot([tt[idx, 1], tt[idx, 1]], [0, H], "k--")
        # grid[1,1].plot([0, tt[idx, 1]], [tt[idx, 0], tt[idx, 0]], "k--")
        # grid[1,1].text(tt[idx, 1], H/2, f"({tt[idx, 1]:0.3f}, {tt[idx, 0]:0.3f})")

        grid[1, 0].set_ylabel("y (m)", labelpad=-14)
        grid[1, 0].set_xlabel(r"$\alpha$ (-)")
        grid[1, 0].set_xticks([0, tt[idx, 1], 2], labels=["0", f"{tt[idx, 1]:0.2f}", "2"])
        grid[1, 0].set_yticks([0, H], labels=["0", f"{H:0.2f}"])
        grid[1, 0].set_xlim([0, 2])
        grid[1, 0].set_ylim([0, H])
        # grid[1,1].savefig(f"silo_alpha_fit_{exps[i][j]}.pdf")

        # nu_cbar_ax = fig.add_axes([right + 0.05, bottom, 0.02, (top - bottom)/2. - bar_off])
        # cb = plt.colorbar(nu_im, cax=nu_cbar_ax)
        # cb.set_label(r"$\nu$ (-)", labelpad=-7, y=0.75)
        # cb.set_ticks([p.outlet_nu, p.nu_fill])

        # plt.sca(grid[3, 0])
        # plt.xlabel("$x$ (m)", labelpad=1)
        # plt.ylabel(r"$|\mathbf{\hat{u}}|$ (-)")
        # plt.xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.2f}", "0", f"{W/2:0.2f}"])
        # plt.sca(grid[3, 2])
        # plt.legend(bbox_to_anchor=(1.2, 0.0), frameon=False, handlelength=1)

        # plt.text(-0.45, 0.5, "Experiment", ha="center", va="center", transform=grid[0, 0].transAxes, rotation=90)
        # plt.text(-0.45, 0.5, "Simulation", ha="center", va="center", transform=grid[1, 0].transAxes, rotation=90)

        plt.text(0.5, 1.1, "(a)", ha="center", va="center", transform=grid[0, 0].transAxes)
        plt.text(0.5, 1.1, "(b)", ha="center", va="center", transform=grid[0, 1].transAxes)
        plt.text(0.5, 1.1, "(c)", ha="center", va="center", transform=grid[1, 0].transAxes)
        plt.text(0.5, 1.1, "(d)", ha="center", va="center", transform=grid[1, 1].transAxes)

        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=0.5)
        plt.savefig(
            os.path.expanduser(f"~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/silo_alpha_single_{nx}_{nm}.pdf")
        )
