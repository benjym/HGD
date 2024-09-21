import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/collapse_fill.json5") as f:
    dict, p = load_file(f)

W = p.H * (p.nx / p.ny)
x = np.linspace(-W / 2, W / 2, p.nx)
y = np.linspace(0, p.H, p.ny)
p.dx = x[1] - x[0]

L = W / 4.0  # length of dashed line
L_0 = W / 4.0  # intersection point with y data
y_off = 0  # -0.005

cmap = colormaps["inferno_r"]
cmap.set_bad("w", 0.0)

fig = plt.figure(figsize=[3.31894680556, 1.4])
ax = fig.subplots(2, 3)

for i, fill in enumerate(p.nu_fill):
    print(fill)
    try:
        files = glob(f"output/collapse_fill/nu_fill_{fill}/data/nu_*.npy")
        files.sort()

        data = np.load(files[0])
        data[data == 0] = np.nan

        ax[0, i].pcolormesh(x, y, data.T, vmin=0, vmax=1, cmap=cmap, rasterized=True)

        data = np.load(files[-1])
        data[data == 0] = np.nan

        ax[1, i].pcolormesh(x, y, data.T, vmin=0, vmax=1, cmap=cmap, rasterized=True)

    except IndexError:
        print(f"Missing file for {fill}")
    except ValueError:
        print(f"Old data file for {fill}")

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
norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Add colorbar to the plot
cax = fig.add_axes([0.92, 0.28, 0.02, 0.95 - 0.28])  # x,y, width, height
cbar = plt.colorbar(scalar_mappable, ax=ax[0], cax=cax)
cbar.set_label(r"$\nu$ (-)")  # Label for the colorbar
cbar.set_ticks([0, 1])  # Set ticks at ends
# Move the colorbar label to the right
label_position = cbar.ax.get_position()
new_x = label_position.x0 + 1  # Adjust this value to move the label
cbar.ax.yaxis.set_label_coords(new_x, 0.5)


plt.subplots_adjust(left=0.12, bottom=0.28, right=0.9, top=0.95, hspace=0.4)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic_SLM/im/collapse_fill.pdf"))
