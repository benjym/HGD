# import os
# from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

# import matplotlib.cm as cm
# import matplotlib.colors as colors
from HGD.params import load_file, dict_to_class, update_before_time_march

plt.style.use("papers/Kinematic_SLM/post_process/paper.mplstyle")

with open("json/test_slope.json5") as f:
    dict, p = load_file(f)

W = p.H * (p.nx / p.ny)
x = np.linspace(-W / 2, W / 2, p.nx)
p.dx = x[1] - x[0]

L = W / 6.0  # length of dashed line
L_0 = W / 10.0  # intersection point with y data
y_off = 0  # -0.005

cmap = colormaps["inferno"]

fig = plt.figure(figsize=[3.31894680556, 3])
ax = fig.subplots(2, 1)

fig2 = plt.figure(figsize=[3.31894680556, 3])
ax2 = fig2.subplots(1, 1)

var1 = "repose_angle"
var2 = "P_stab"

for j, b in enumerate(getattr(p, var2)):
    for i, a in enumerate(getattr(p, var1)):
        dict_copy = dict.copy()
        dict_copy[var1] = a
        dict_copy[var2] = b
        this_p = dict_to_class(dict_copy)
        this_p.set_defaults()
        this_p = update_before_time_march(this_p, None)

        try:
            # init = f"output/test_slope/repose_angle_{angle}/P_stab_{P_stab}/data/nu_000000.npy"
            init = f"output/test_slope/{var1}_{a}/{var2}_{b}/data/nu_" + str(0).zfill(6) + ".npy"
            final = f"output/test_slope/{var1}_{a}/{var2}_{b}/data/nu_" + str(this_p.nt - 1).zfill(6) + ".npy"
            if i == 0:
                data = np.load(init)
                bw = data >= p.nu_cs / 2.0
                # bw = data > 0
                # bw = data >= p.nu_cs
                top = np.argmin(bw, axis=1) * p.H / p.ny

                ax[0].plot(x, top, label="Initial", color="blue", ls="-", lw=3)

            # if not os.path.exists(final):
            #     print(f"Missing final file for repose angle={angle}, P_stab={P_stab}. Using most recent file.")
            #     files = glob(f"output/test_slope/repose_angle_{angle}/P_stab_{P_stab}/data/nu_*.npy")
            #     files.sort()
            #     final = files[-1]

            data = np.load(final)
            bw = data >= p.nu_cs / 2.0
            # bw = data > 0
            # bw = data >= p.nu_cs
            top = np.argmin(bw, axis=1) * p.H / p.ny

            color = cmap(i / (len(getattr(p, var1)) - 1))

            ax[0].plot(x, top, label=rf"$\varphi={a}^\circ$", color=color)
            # plt.show()

            max_height = np.max(top)
            min_height = np.min(top)
            range_height = max_height - min_height

            # left = top[: np.argmin(np.abs(top - max_height))]
            left = top[: p.nx // 2]

            fit_min_arg = np.argmin(np.abs(left - (min_height + range_height / 4.0)))
            fit_max_arg = np.argmin(np.abs(left - (max_height - range_height / 4.0)))

            if fit_min_arg == fit_max_arg:
                coefficients = [0, min_height]
                x_fit = x
            elif a == 90:
                fit_max_arg = np.argmin(np.abs(top - max_height))
                fit_min_arg = fit_max_arg - 1

                x_fit = x[fit_min_arg : fit_max_arg + 1]
                coefficients = np.polyfit(x_fit, top[fit_min_arg : fit_max_arg + 1], 1)
            else:
                x_fit = x[fit_min_arg:fit_max_arg]

                coefficients = np.polyfit(x_fit, top[fit_min_arg:fit_max_arg], 1)

            ax[0].plot(x_fit, coefficients[0] * x_fit + coefficients[1], ls="--", lw=2, color=color)

            color = cmap(j / (len(getattr(p, var2)) - 1))
            ax[1].plot(a, np.degrees(np.arctan(coefficients[0])), "x", mec=color, mfc=color)

            ax2.plot(
                p.delta_limit[i] / p.nu_cs, np.degrees(np.arctan(coefficients[0])), "x", mec=color, mfc=color
            )

        except IndexError:
            print(f"Missing file for {var1}={a}, {var2}={b}")
        except ValueError:
            print(f"Old data file for {var1}={a}, {var2}={b}")
        # except TypeError:
        # print(f"TypeError for {var1}={a}, {var2}={b}")
        except FileNotFoundError:
            print(f"FileNotFoundError for {var1}={a}, {var2}={b}")

plt.sca(ax[1])
plt.plot([0, 90], [0, 90], "k--")
plt.xlabel(r"$\varphi$ (degrees)", labelpad=0)
plt.ylabel("Measured angle of repose\n(degrees)")
plt.xticks([0, 30, 60, 90])
plt.yticks([0, 30, 60, 90])
# plt.xlim([0, 90])
# plt.ylim([0, 90])

plt.subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, hspace=0.4)
plt.savefig("slope_test.png")

plt.sca(ax2)
phi = np.linspace(1, 89, 100)
mu = np.tan(np.radians(phi))
delta_nu = 1 / (1 / mu + 1)
plt.plot(delta_nu, phi, "k-")
plt.xlabel(r"$\Delta\nu/\nu_{cs}$")
plt.ylabel("Measured angle of repose\n(degrees)")
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, hspace=0.4)
plt.savefig("delta_limit.png")
# plt.show()
