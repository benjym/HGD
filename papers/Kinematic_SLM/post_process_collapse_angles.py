import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/collapse_angles.json5") as f:
    dict, p = load_file(f)

W = p.H * (p.nx / p.ny)
x = np.linspace(-W / 2, W / 2, p.nx)
p.dx = x[1] - x[0]

L = W / 6.0  # length of dashed line
L_0 = W / 6.0  # intersection point with y data
y_off = 0  # -0.005

cmap = colormaps["inferno"]


fig = plt.figure(figsize=[3.31894680556, 3])
ax = fig.subplots(2, 1)

for i, angle in enumerate(p.repose_angle):
    try:
        files = glob(f"output/collapse_angles/repose_angle_{angle}/data/nu_*.npy")
        files.sort()
        if i == 0:
            data = np.load(files[0])
            bw = data >= p.nu_cs / 2.0
            top = np.argmin(bw, axis=1) * p.H / p.ny

            ax[0].plot(x, top, label="Initial", color="k", ls="--")

        data = np.load(files[-1])
        bw = data >= p.nu_cs / 2.0
        top = np.argmin(bw, axis=1) * p.H / p.ny

        color = cmap(i / (len(p.repose_angle) - 1))

        ax[0].plot(x, top, label=rf"$\varphi={angle}^\circ$", color=color)

        fit_min_arg = np.argmin(np.abs(x + L_0 + L / 2.0))
        fit_max_arg = np.argmin(np.abs(x + L_0 - L / 2.0))
        x_fit = x[fit_min_arg:fit_max_arg]

        # ax[0].plot(
        #     x_fit,
        #     (x_fit + L_0) * np.tan(np.radians(angle)) + top[np.argmin(np.abs(x + L_0))] + y_off,
        #     # x[0:p.nx//4],
        #     # np.linspace(0,W/4,p.nx//4)* np.tan(np.radians(angle)) + p.H/2.,
        #     ls="-",
        #     lw=0.5,
        #     color=color,
        # )
        coefficients = np.polyfit(x_fit, top[fit_min_arg:fit_max_arg], 1)

        ax[0].plot(x_fit, coefficients[0] * x_fit + coefficients[1], ls="--", lw=2, color=color)

        ax[1].plot(angle, np.degrees(np.arctan(coefficients[0])), "k.")
        # ax[1].plot(np.tan(np.radians(angle)), coefficients[0], "k.")
        # linear_fit = np.poly1d(coefficients)
    except IndexError:
        print(f"Missing file for repose angle={angle}")
    except ValueError:
        print(f"Old data file for repose angle={angle}")


plt.sca(ax[0])
plt.xlabel("$x$ (m)", labelpad=0)
plt.ylabel("$y$ (m)")  # , rotation="horizontal")  # ,labelpad=3)
plt.ylim([0, p.H])
plt.xlim([-W / 2, W / 2])
plt.xticks([-W / 2, 0, W / 2])
plt.yticks([0, p.H])
plt.axis("equal")
# plt.legend(loc=0)

# Create a ScalarMappable with the colormap you want to use
norm = colors.Normalize(vmin=0, vmax=1)  # Set the range for the colorbar
scalar_mappable = cm.ScalarMappable(norm=norm, cmap="inferno")

# Add colorbar to the plot
cbar = plt.colorbar(scalar_mappable, ax=ax[0])
cbar.set_label(r"$\varphi$ (degrees)")  # Label for the colorbar
cbar.set_ticks([0, 1])  # Set ticks at ends
cbar.set_ticklabels([p.repose_angle[0], p.repose_angle[-1]])  # Label the ticks
# Move the colorbar label to the right
label_position = cbar.ax.get_position()
new_x = label_position.x0 + 4  # Adjust this value to move the label
cbar.ax.yaxis.set_label_coords(new_x, 0.5)

plt.sca(ax[1])
plt.plot([0, 45], [0, 45], "k--")
plt.xlabel(r"$\varphi$ (degrees)", labelpad=0)
plt.ylabel("Measured angle of repose\n(degrees)")
plt.xticks([0, 15, 30, 45])
plt.yticks([0, 15, 30, 45])
plt.xlim([0, 45])
plt.ylim([0, 45])

plt.subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, hspace=0.4)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic SLM/im/collapse_angle.pdf"))