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

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/silo_alpha.json5") as f:
    dict, p = load_file(f)

y = np.linspace(0, p.H, p.ny)
p.dy = p.H / p.ny
x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
W = x[-1] - x[0]
p.dx = x[1] - x[0]

cmap = colormaps["inferno"]
cmap.set_under("black")
cmap.set_over("black")
cmap.set_bad("black")

imgs = ["rice_10.png", "lentils_10.png", "glass_beads_10.png"]

fig = plt.figure(figsize=[3.31894680556, 3])
grid = fig.subplots(2, 3)
alphas = p.alpha
for i, val in enumerate(alphas):
    p.alpha = val
    p.update_before_time_march(cycles)

    t_plot = 0.02 * p.t_f
    nt_plot = int(t_plot / p.dt)
    dt = 10  # int(p.t_f / p.dt / 100)

    for j, t in enumerate(range(nt_plot - dt, nt_plot + dt)):
        this_u = np.load(f"output/silo_alpha/alpha_{val}/data/u_{str(t).zfill(6)}.npy")
        this_v = np.load(f"output/silo_alpha/alpha_{val}/data/v_{str(t).zfill(6)}.npy")
        if j == 0:
            u = this_u.copy()
            v = this_v.copy()
        else:
            u += this_u
            v += this_v
    u /= 2 * dt
    v /= 2 * dt

    u_mag = np.sqrt(u**2 + v**2)

    image = plt.imread(f"papers/Kinematic_SLM/other/im/{imgs[i]}")
    im = grid[0, i].imshow(image)
    grid[0, i].axis("off")

    im = grid[1, i].pcolormesh(
        x,
        y,
        u_mag.T,
        # vmin=1e3 * p.s_m,
        # vmax=1e3 * p.s_M,
        cmap=cmap,
        rasterized=True,
        norm=colors.LogNorm(),
    )
    grid[1, i].set_aspect("equal")

    # except IndexError:
    #     print(f"Missing file for {val}")
    # except ValueError as e:
    #     print(f"Old data file for {val}")
    #     print(e)

    grid[1, i].set_xticks([])
    grid[1, i].set_yticks([])

plt.sca(grid[0, 0])
plt.xlabel("$x$ (m)", labelpad=1)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])

plt.sca(grid[1, 0])
plt.xlabel("$x$ (m)", labelpad=1)
plt.ylabel("$y$ (m)", labelpad=-7)
plt.xticks([-W / 2, 0, W / 2], labels=[f"{-W/2:0.2f}", "0", f"{W/2:0.2f}"])
plt.yticks([0, p.H], labels=["0", f"{p.H:0.2f}"])

# plt.ylim([0, p.H])
# plt.xlim([-W / 2, W / 2])
# plt.legend(loc=0)

# # Create a ScalarMappable with the colormap you want to use
# norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
# scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Add colorbar to the plot
# cax = fig.add_axes([0.86, 0.28, 0.02, 0.95 - 0.28])  # x,y, width, height
# cbar = plt.colorbar(cb_ax, ax=ax[0], cax=cax)
# cbar.set_label(r"$\bar{s}$ (mm)")  # Label for the colorbar
# # cbar.set_ticks([0, 1])  # Set ticks at ends
# # Move the colorbar label to the right
# label_position = cbar.ax.get_position()
# new_x = label_position.x0 + 3.5  # Adjust this value to move the label
# cbar.ax.yaxis.set_label_coords(new_x, 0.5)


plt.subplots_adjust(left=0.10, bottom=0.13, right=0.99, top=0.99, hspace=0.2)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/silo_alpha.pdf"))
