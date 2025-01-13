import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file
from void_migration import stress, operators

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/collapse_stress.json5") as f:
    dict, p = load_file(f)

p.update_before_time_march(None)

y = np.linspace(0, p.H, p.ny)
p.dy = p.H / p.ny
x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
# W = x[-1] - x[0]
W = p.H / p.aspect_ratio_y
p.dx = x[1] - x[0]

L = W / 4.0  # length of dashed line
L_0 = W / 4.0  # intersection point with y data
y_off = 0  # -0.005

cmap = colormaps["inferno"]
cmap.set_bad("w", 0.0)

fig = plt.figure(figsize=[3.31894680556, 2.6], constrained_layout=True)
ax = fig.subplots(3, 2)

files = glob(f"output/collapse_stress/data/s_*.npy")
files.sort()

initial = np.load(files[0])
initial[initial == 0] = np.nan

final = np.load(files[-1])
final[final == 0] = np.nan

files = glob(f"output/collapse_stress/data/last_swap_*.npy")
files.sort()

last_swap_i = np.load(files[0])
last_swap_f = np.load(files[-1])

labelpad = 2

for i in [0, 1]:
    if i == 0:
        sigma = stress.calculate_stress(initial, last_swap_i, p)
        nu = operators.get_solid_fraction(initial)
    else:
        sigma = stress.calculate_stress(final, last_swap_f, p)
        nu = operators.get_solid_fraction(final)
    pressure = stress.get_pressure(sigma, p, last_swap_i)
    deviatoric = stress.get_deviatoric(sigma, p, last_swap_i)
    friction_angle = stress.get_friction_angle(sigma, p, last_swap_i)
    sigma_xx = stress.get_sigma_xx(sigma, p, last_swap_i)
    sigma_xy = sigma[:, :, 0]
    sigma_yy = sigma[:, :, 1]

    mask = nu < 0.1
    sigma_xx = np.ma.masked_where(mask, sigma_xx)
    sigma_xy = np.ma.masked_where(mask, sigma_xy)
    sigma_yy = np.ma.masked_where(mask, sigma_yy)

    print(sigma_xx.min(), sigma_xx.max())
    print(sigma_yy.min(), sigma_yy.max())
    print(sigma_xy.min(), sigma_xy.max())
    sigma_max = 3

    im = ax[0, i].pcolormesh(
        x,
        y,
        # 1e-3 * pressure.T,
        1e-3 * sigma_xx.T,
        cmap=cmap,
        vmin=0,
        vmax=sigma_max,
        # vmax=max_angle,
        rasterized=True,
    )
    if i == 1:
        cb = plt.colorbar(im, aspect=10, ticks=[0, sigma_max])
        cb.set_label(r"$\sigma_{xx}$ (kPa)", labelpad=labelpad)

    im = ax[1, i].pcolormesh(
        x,
        y,
        # 1e-3 * deviatoric.T,
        1e-3 * sigma_yy.T,
        cmap=cmap,
        vmin=0,
        vmax=sigma_max,
        # vmax=max_angle,
        rasterized=True,
    )
    if i == 1:
        cb = plt.colorbar(im, aspect=10, ticks=[0, sigma_max])
        cb.set_label(r"$\sigma_{yy}$ (kPa)", labelpad=labelpad)

    im = ax[2, i].pcolormesh(
        x,
        y,
        1e-3 * sigma_xy.T,
        # cmap=cmap,
        vmin=-2,
        vmax=2,
        # friction_angle.T,
        cmap="bwr",
        # vmin=p.repose_angle - 5,
        # vmax=p.repose_angle + 5,
        rasterized=True,
    )
    if i == 1:
        cb = plt.colorbar(
            im,
            aspect=10,
            ticks=[-2, 0, 2],
        )
        cb.set_label(r"$\sigma_{xy}$ (kPa)", labelpad=labelpad)
    # cb.set_ticks([0, p.repose_angle, 2 * p.repose_angle])
    # cb.set_ticks([p.repose_angle - 5, p.repose_angle, p.repose_angle + 5])


for j in [0, 1]:
    for i in [0, 1, 2]:
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_aspect("equal")

plt.sca(ax[-1, 0])
plt.xlabel("$x$ (m)", labelpad=0)
plt.ylabel("$y$ (m)", labelpad=-4)  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])


# plt.subplots_adjust(left=0.05, bottom=0.18, right=0.86, top=0.97, hspace=0.25, wspace=0.05)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/collapse_stress.pdf"))
