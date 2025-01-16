import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import colormaps
# import matplotlib.cm as cm
# import matplotlib.colors as colors
from void_migration.params import load_file
from void_migration.plotter import size_colormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/collapse_bi_conc.json5") as f:
    dict, p = load_file(f)

y = np.linspace(0, p.H, p.ny)
p.dy = p.H / p.ny
x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
W = x[-1] - x[0]
p.dx = x[1] - x[0]

L = W / 4.0  # length of dashed line
L_0 = W / 4.0  # intersection point with y data
y_off = 0  # -0.005

# cmap = colormaps["inferno_r"]
# cmap = colormaps["viridis"]
# cmap.set_bad("w", 0.0)
cmap = size_colormap()

fig = plt.figure(figsize=[3.31894680556, 1.4])
ax = fig.subplots(2, 3)

for i, val in enumerate(p.large_concentration):
    print(val)
    try:
        files = glob(f"output/collapse_bi_conc/large_concentration_{val}/data/s_bar_*.npy")
        files.sort()

        data = np.load(files[0])
        data[data == 0] = np.nan

        ax[0, i].pcolormesh(
            x, y, 1e3 * data.T, vmin=1e3 * p.s_m, vmax=1e3 * p.s_M, cmap=cmap, rasterized=True
        )

        data = np.load(files[-1])
        data[data == 0] = np.nan

        cb = ax[1, i].pcolormesh(
            x, y, 1e3 * data.T, vmin=1e3 * p.s_m, vmax=1e3 * p.s_M, cmap=cmap, rasterized=True
        )

        # Add colorbars spanning the full height of each column (columns 1 and 2)
        cbar_ax = inset_axes(
            ax[0, i],
            width="5%",
            height="100%",
            loc="center right",
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax[0, i].transAxes,
            borderpad=0,
        )
        cbar = fig.colorbar(cb, cax=cbar_ax, orientation="vertical")

    except IndexError:
        print(f"Missing file for {val}")
    except ValueError:
        print(f"Old data file for {val}")

    for j in [0, 1]:
        plt.sca(ax[j, i])
        plt.xticks([])
        plt.yticks([])
        plt.axis("equal")

plt.sca(ax[1, 0])
plt.xlabel("$x$ (m)", labelpad=0)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])

# plt.ylim([0, p.H])
# plt.xlim([-W / 2, W / 2])
# plt.legend(loc=0)

# # Create a ScalarMappable with the colormap you want to use
# norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
# scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# # # Add colorbar to the plot
# cax = fig.add_axes([0.86, 0.28, 0.02, 0.95 - 0.28])  # x,y, width, height
# cbar = plt.colorbar(cb_ax, ax=ax[0], cax=cax)
# cbar.set_label(r"$\bar{s}$ (mm)")  # Label for the colorbar
# # cbar.set_ticks([0, 1])  # Set ticks at ends
# # Move the colorbar label to the right
# label_position = cbar.ax.get_position()
# new_x = label_position.x0 + 3.5  # Adjust this value to move the label
# cbar.ax.yaxis.set_label_coords(new_x, 0.5)


plt.subplots_adjust(left=0.12, bottom=0.28, right=0.84, top=0.95, hspace=0.4)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/collapse_bi_conc.pdf"))
