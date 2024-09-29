import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file
from void_migration.plotter import size_colormap

# from mpl_toolkits.axes_grid1 import Grid
from scipy.integrate import cumtrapz

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/collapse_poly.json5") as f:
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

cmap = size_colormap()

fig = plt.figure(figsize=[3.31894680556, 1.55])
grid = fig.subplots(2, 3)

for i, val in enumerate(p.power_law_alpha):
    print(val)
    try:
        files = glob(f"output/collapse_poly/power_law_alpha_{val}/data/s_*.npy")
        files.sort()

        data = np.load(files[0])
        # data[data == 0] = np.nan
        data = data[~np.isnan(data)]

        # GSD, bin_edges = np.histogram(
        #     data.flatten(), bins=np.logspace(np.log10(p.s_m), np.log10(p.s_M), 50), density=True)

        GSD, bin_edges = np.histogram(data.flatten(), bins=np.linspace(p.s_m, p.s_M, 50), density=True)
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
        CGSD = cumtrapz(GSD, x=bin_centers)

        grid[0, i].semilogx(1e3 * bin_edges[1:-1], CGSD, label=f"$\\alpha = {val}$", color="k")
        grid[0, i].set_xticks([])
        grid[0, i].set_yticks([])
        # grid[0, i].semilogx(bin_centers, GSD, label=f"$\\alpha = {val}$")
        # grid[0, i].set_xscale("log")
        # grid[0, i].set_xticks([1e3 * p.s_m, 1e3 * p.s_M])
        # grid[2 * i + 0].set_yscale("log")

        data = np.load(files[-1])
        data[data == 0] = np.nan
        data = np.nanmean(data, axis=2)

        im = grid[1, i].pcolormesh(
            x,
            y,
            1e3 * data.T,
            # vmin=1e3 * p.s_m,
            # vmax=1e3 * p.s_M,
            cmap=cmap,
            rasterized=True,
            norm=colors.LogNorm(vmin=1e3 * p.s_m, vmax=1e3 * p.s_M),
        )

        grid[1, i].set_xticks([])
        grid[1, i].set_yticks([])
        grid[1, i].set_aspect("equal")

        # print(data[~np.isnan(data)].min(), data[~np.isnan(data)].max())

        # grid.cbar_axes[i].colorbar(im)
        # grid.cbar_axes[i].set_title(r"$\bar{s}$ (mm)", y=-5.5)
        # grid.cbar_axes[i].set_xticks([1e3 * p.s_m, 1e3 * p.s_M])

    except IndexError:
        print(f"Missing file for {val}")
    except ValueError as e:
        print(f"Old data file for {val}")
        print(e)

    # for j in [0, 1]:
    #     plt.sca(ax[j, i])
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis("equal")

grid[0, 0].set_xticks([1e3 * p.s_m, 1e3 * p.s_M])
grid[0, 0].set_yticks([0, 1])
grid[0, 0].set_xlabel(r"$s$ (mm)", labelpad=-3)
grid[0, 0].set_ylabel(r"CGSD (-)")

plt.sca(grid[1, 0])
plt.xlabel("$x$ (m)", labelpad=-3)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, W / 2])
plt.yticks([0, p.H])

# plt.ylim([0, p.H])
# plt.xlim([-W / 2, W / 2])
# plt.legend(loc=0)

# Create a ScalarMappable with the colormap you want to use
norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# Add colorbar to the plot
cax = fig.add_axes([0.83, 0.22, 0.02, (0.95 - 0.28 - 0.27) / 2.0])  # x,y, width, height
cbar = plt.colorbar(im, ax=grid[1, 0], cax=cax)
cbar.set_label(r"$\bar{s}$ (mm)")  # Label for the colorbar
cbar.set_ticks([1e3 * p.s_m, 1e3 * p.s_M])  # Set ticks at ends
cbar.ax.minorticks_off()
# Move the colorbar label to the right
label_position = cbar.ax.get_position()
new_x = label_position.x0 + 5  # Adjust this value to move the label
cbar.ax.yaxis.set_label_coords(new_x, 0.5)


plt.subplots_adjust(left=0.12, bottom=0.18, right=0.81, top=0.95, hspace=0.8)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/collapse_poly.pdf"))
