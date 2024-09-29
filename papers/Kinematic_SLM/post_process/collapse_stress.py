import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file
from void_migration import stress

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/collapse_stress.json5") as f:
    dict, p = load_file(f)

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

max_angle = 60

fig = plt.figure(figsize=[3.31894680556, 2.2])
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

p.stress_mode = "active"

for i in [0, 1]:
    if i == 0:
        sigma = stress.calculate_stress(initial, last_swap_i, p)
    else:
        sigma = stress.calculate_stress(final, last_swap_f, p)
    pressure = stress.get_pressure(sigma, p, last_swap_i)
    deviatoric = stress.get_deviatoric(sigma, p, last_swap_i)
    friction_angle = stress.get_friction_angle(sigma, p, last_swap_i)

    im = ax[0, i].pcolormesh(
        x,
        y,
        1e-3 * pressure.T,
        # 1e-3 * sigma[:, :, 1].T,
        cmap=cmap,
        vmin=0,
        # vmax=max_angle,
        rasterized=True,
    )
    cb = plt.colorbar(im, aspect=10)
    cb.set_label(r"$p$ (kPa)", labelpad=labelpad)

    im = ax[1, i].pcolormesh(
        x,
        y,
        1e-3 * deviatoric.T,
        cmap=cmap,
        vmin=0,
        # vmax=max_angle,
        rasterized=True,
    )
    cb = plt.colorbar(im, aspect=10)
    cb.set_label(r"$\tau$ (kPa)", labelpad=labelpad)

    im = ax[2, i].pcolormesh(
        x,
        y,
        friction_angle.T,
        cmap="Spectral",
        vmin=p.repose_angle - 5,
        vmax=p.repose_angle + 5,
        rasterized=True,
    )
    cb = plt.colorbar(
        im,
        aspect=10,
    )
    cb.set_label(r"$\phi_\mathrm{mob}$ (°)", labelpad=labelpad)
    # cb.set_ticks([0, p.repose_angle, 2 * p.repose_angle])
    cb.set_ticks([p.repose_angle - 5, p.repose_angle, p.repose_angle + 5])


for j in [0, 1]:
    for i in [0, 1, 2]:
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_aspect("equal")

plt.sca(ax[-1, 0])
plt.xlabel("$x$ (m)", labelpad=0)
plt.ylabel("$y$ (m)", labelpad=0)  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])


plt.subplots_adjust(left=0.11, bottom=0.18, right=0.9, top=0.97, hspace=0.4, wspace=0.4)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/collapse_stress.pdf"))
