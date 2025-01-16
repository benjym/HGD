import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from HGD.params import load_file
from HGD import stress, operators

plt.style.use("papers/HGD/paper.mplstyle")

with open("papers/HGD/json/collapse_stress.json5") as f:
    dict, p = load_file(f)

p.update_before_time_march(None)
p.D_0 = 0.1
y = np.linspace(0, p.H, p.ny)
p.dy = p.H / p.ny
x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
W = p.H / p.aspect_ratio_y
p.dx = x[1] - x[0]

cmap = colormaps["inferno"]
cmap.set_bad("w", 0.0)

fig = plt.figure(figsize=[3.5, 1.9])  # , constrained_layout=True)
ax = fig.subplots(2, 2)

files = glob(f"output/collapse_stress/data/s_*.npy")
files.sort()


final = np.load(files[-1])
final[final == 0] = np.nan

files = glob(f"output/collapse_stress/data/last_swap_*.npy")
files.sort()

last_swap_i = np.load(files[0])
last_swap_f = np.load(files[-1])

labelpad = 2

# for i in [0, 1]:
# if i == 0:
#     sigma = stress.calculate_stress(initial, last_swap_i, p)
#     nu = operators.get_solid_fraction(initial)
# else:

sigma = stress.calculate_stress(final, last_swap_f, p)
nu = operators.get_solid_fraction(final)
# pressure = stress.get_pressure(sigma, p, last_swap_i)
# deviatoric = stress.get_deviatoric(sigma, p, last_swap_i)
# friction_angle = stress.get_friction_angle(sigma, p, last_swap_i)
sigma_xx = sigma[:, :, 2]
sigma_xy = sigma[:, :, 0]
sigma_yy = sigma[:, :, 1]

mask = nu < 0.1
sigma_xx = np.ma.masked_where(mask, sigma_xx)
sigma_xy = np.ma.masked_where(mask, sigma_xy)
sigma_yy = np.ma.masked_where(mask, sigma_yy)

print(sigma_xx.min(), sigma_xx.max())
print(sigma_yy.min(), sigma_yy.max())
print(sigma_xy.min(), sigma_xy.max())
# sigma_max = 5

im = ax[0, 1].pcolormesh(
    x,
    y,
    # 1e-3 * pressure.T,
    1e-3 * sigma_xx.T,
    cmap=cmap,
    vmin=0,
    vmax=1,
    # vmax=max_angle,
    rasterized=True,
)
# if i == 1:
cb = plt.colorbar(im, aspect=10, ticks=[0, 1])
cb.set_label(r"$\sigma_{xx}$ (kPa)", labelpad=labelpad)

im = ax[0, 0].pcolormesh(
    x,
    y,
    # 1e-3 * deviatoric.T,
    1e-3 * sigma_yy.T,
    cmap=cmap,
    vmin=0,
    vmax=5,
    # vmax=max_angle,
    rasterized=True,
)
# if i == 1:
cb = plt.colorbar(im, aspect=10, ticks=[0, 5])
cb.set_label(r"$\sigma_{yy}$ (kPa)", labelpad=labelpad)

im = ax[1, 1].pcolormesh(
    x,
    y,
    1e-3 * sigma_xy.T,
    # cmap=cmap,
    vmin=-0.5,
    vmax=0.5,
    # friction_angle.T,
    cmap="bwr",
    # vmin=p.repose_angle - 5,
    # vmax=p.repose_angle + 5,
    rasterized=True,
)
# if i == 1:
cb = plt.colorbar(
    im,
    aspect=10,
    ticks=[-0.5, 0, 0.5],
)
cb.set_label(r"$\sigma_{xy}$ (kPa)", labelpad=labelpad)
# cb.set_ticks([0, p.repose_angle, 2 * p.repose_angle])
# cb.set_ticks([p.repose_angle - 5, p.repose_angle, p.repose_angle + 5])

plt.subplot(223)
gamma = p.solid_density * p.g * p.nu_cs
has_material = nu >= 0.5
# H = np.max(np.argmin(has_material, axis=1) * p.dy)
# print(H)
H = 0.3640856541  # height of pile back calculated from Trollope data (H = W/2*tan(32.5), W = 3'9" = 1.143 m)
trollope_A = [
    [0.0015248242598482342, 0.7909125914516749],
    [0.20125942323257906, 0.7993839045051983],
    [0.4020400909393702, 0.7339237581825181],
    [0.6007523473830747, 0.6984982672314207],
    [0.8008732150186566, 0.5976126299576432],
    [0.8015472205884209, 0.517905275317674],
    [0.999843890638213, 0.48902579899884446],
    [1.0000941492085118, 0.41817481709665005],
    [1.2008516951995685, 0.3592606854062381],
    [1.2008468592851667, 0.24951867539468586],
]  # material A, varphi = 32.5 deg
trollope_B = [
    [0.0011711798422186503, 0.7390963470880365],
    [0.14129351516579658, 0.7299076345074559],
    [0.28045655038283746, 0.7285007672232491],
    [0.41935646176772234, 0.6776368860139681],
    [0.561074798458695, 0.618080979390624],
    [0.6994662515806347, 0.5782058149658869],
    [0.8364382697815818, 0.545195235361081],
    [0.9759098316312818, 0.4677732824875064],
    [1.118479669966777, 0.42699501830383324],
    [1.2607752329050825, 0.3395072734091584],
    [1.3983921832839366, 0.14759768959043362],
]  # material B, varphi = 39 deg

# for D_0 in [0, 0.25, 0.5, 0.75, 1]:
# for D_0 in [0.1]:
#     p.D_0 = D_0
sigma = stress.calculate_stress(final, last_swap_f, p)
sigma_yy = sigma[:, :, 1]
plt.plot(x, sigma_yy[:, 0] / (gamma * H), "r-")  # , label=rf"HGD_{D_0}")
for rel_x, rel_sigma in trollope_A:
    plt.plot(rel_x * H, rel_sigma, "b.")
# plt.plot(x, sigma_yy[-1, :], label=r"$\sigma_{yy}$")
# plt.plot(x, sigma_xy[-1, :], label=r"$\sigma_{xy}$")
# plt.legend(loc='upper left')
plt.xlim(xmin=0, xmax=0.6)
plt.xticks([0, W / 2.0], ["0", "$W/2$"])
plt.yticks([0, 1.0])
plt.xlabel("$x$ (m)", labelpad=0)
plt.ylabel(r"$\sigma_{yy}/(\rho_s\nu g H)$")


for j in [0, 1]:
    for i in [0, 1]:
        if i == 1 and j == 0:
            continue
        ax[i, j].set_aspect("equal")
        ax[i, j].set_xlabel("$x$ (m)", labelpad=0)
        ax[i, j].set_ylabel("$y$ (m)", labelpad=1)  # , rotation="horizontal")  # ,labelpad=3)
        ax[i, j].set_xticks(
            [-W / 2, 0, W / 2],
        )
        ax[i, j].set_xticklabels(["$-W/2$", "0", "$W/2$"])
        ax[i, j].set_yticks([0, p.H])
        ax[i, j].set_yticklabels(["0", "$H$"])


# plt.text(0.5, 1.25, "(a)", ha="center", va="center", transform=ax[0, 0].transAxes)
# plt.text(0.5, 1.25, "(b)", ha="center", va="center", transform=ax[0, 1].transAxes)
# plt.text(0.4, 1.25, "(c)", ha="center", va="center", transform=ax[1, 0].transAxes)
# plt.text(0.5, 1.25, "(d)", ha="center", va="center", transform=ax[1, 1].transAxes)

plt.subplots_adjust(left=0.12, bottom=0.20, right=0.86, top=0.9, hspace=1.4, wspace=0.7)
plt.savefig(
    os.path.expanduser("~/Dropbox/Apps/Overleaf/Heterarchical Granular Dynamics/im/collapse_stress.pdf")
)
