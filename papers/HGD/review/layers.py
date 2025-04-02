import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from HGD.params import load_file

plt.style.use("papers/HGD/paper.mplstyle")

with open("papers/HGD/review/layers.json5") as f:
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

cmap = colormaps["inferno_r"]
cmap.set_bad("w", 0.0)

fig = plt.figure(figsize=[3.31894680556, 2.1])
ax = plt.subplot(111)
# ax = fig.subplots(11)

# for i, tstep in enumerate([0, 500, 1000, 1500, 2000, 2500]):
all_data = np.zeros((p.nt // p.save_inc, p.ny))
for tstep in range(0, p.nt, p.save_inc):
    filename = f"output/layers/data/nu_{tstep:06d}.npy"
    if os.path.exists(filename):
        data = np.load(filename)
        data[data == 0] = np.nan
        all_data[tstep, :] = data.mean(axis=0)
    else:
        all_data[tstep, :] = np.nan


# ax[i].pcolormesh(x, y, data.T, vmin=0, vmax=1, cmap=cmap, rasterized=True)
t = np.arange(0, p.nt, p.save_inc) * p.dt
plt.pcolormesh(t, y, all_data.T, vmin=0, vmax=1, cmap=cmap, rasterized=True)

# for i in [1, 2, 3, 4, 5]:
#     plt.sca(ax[i])
#     plt.xticks([])
#     plt.yticks([])
#     # plt.axis("equal")

# plt.sca(ax[0])
plt.xlabel("$t$ (s)", labelpad=0)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
# plt.xticks([-W / 2, W / 2])
plt.yticks([0, p.H])

# plt.ylim([0, p.H])
# plt.xlim([-W / 2, W / 2])
# plt.legend(loc=0)

# # Create a ScalarMappable with the colormap you want to use
norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

# # Add colorbar to the plot
cax = fig.add_axes([0.92, 0.20, 0.02, 0.95 - 0.20])  # x,y, width, height
cbar = plt.colorbar(scalar_mappable, ax=ax, cax=cax)
cbar.set_label(r"$\nu$ (-)")  # Label for the colorbar
cbar.set_ticks([0, 1])  # Set ticks at ends
# Move the colorbar label to the right
label_position = cbar.ax.get_position()
new_x = label_position.x0 + 1  # Adjust this value to move the label
cbar.ax.yaxis.set_label_coords(new_x, 0.5)


plt.subplots_adjust(left=0.12, bottom=0.20, right=0.9, top=0.95, hspace=0.4, wspace=0.65)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Heterarchical Granular Dynamics/im/layers.pdf"))
