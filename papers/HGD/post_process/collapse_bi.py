import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from HGD.params import load_file
from HGD.plotter import size_colormap
from mpl_toolkits.axes_grid1 import AxesGrid

plt.style.use("papers/HGD/paper.mplstyle")

with open("papers/HGD/json/collapse_bi.json5") as f:
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

# cmap = colormaps["inferno_r"]
# cmap = colormaps["viridis"]
# cmap.set_bad("w", 0.0)
cmap = size_colormap()

fig = plt.figure(figsize=[3.5, 2.1])
# ax = fig.subplots(2, 3)
grid = AxesGrid(
    fig,
    111,  # similar to subplot(121)
    nrows_ncols=(2, 3),
    axes_pad=0.10,
    share_all=False,
    label_mode="1",
    cbar_location="bottom",
    cbar_mode="edge",
    cbar_pad=0.42,
    cbar_size="15%",
    direction="column",
)


for cax in grid.cbar_axes:
    cax.axis[cax.orientation].set_label("Bar")

for i, s_M in enumerate(p.s_M):
    print(s_M)
    try:
        files = glob(f"output/collapse_bi/s_M_{s_M}/data/s_bar_*.npy")
        files.sort()

        data = np.load(files[0])
        data[data == 0] = np.nan

        grid[2 * i + 0].pcolormesh(
            x, y, 1e3 * data.T, vmin=1e3 * p.s_m, vmax=1e3 * p.s_M[i], cmap=cmap, rasterized=True
        )

        data = np.load(files[-1])
        data[data == 0] = np.nan

        im = grid[2 * i + 1].pcolormesh(
            x, y, 1e3 * data.T, vmin=1e3 * p.s_m, vmax=1e3 * p.s_M[i], cmap=cmap, rasterized=True
        )

        grid.cbar_axes[i].colorbar(im)
        grid.cbar_axes[i].set_title(r"$\bar{s}$ (mm)", y=-5.5)
        grid.cbar_axes[i].set_xticks([1e3 * p.s_m, 1e3 * p.s_M[i]])

    except IndexError:
        print(f"Missing file for {s_M}")
    except ValueError:
        print(f"Old data file for {s_M}")

    # for j in [0, 1]:
    #     plt.sca(ax[j, i])
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.axis("equal")

plt.sca(grid[1])
plt.xlabel("$x$ (m)", labelpad=1)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])

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


plt.subplots_adjust(left=0.11, bottom=0.20, right=0.99, top=0.95, hspace=0.4)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Heterarchical Granular Dynamics/im/collapse_bi.pdf"))
