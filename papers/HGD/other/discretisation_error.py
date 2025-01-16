import os
import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps

matplotlib.style.use("../paper.mplstyle")

cmap = colormaps["inferno"]

# for mu in [0.1,0.2,0.5,1,2,10]:
phis = [10, 30, 50, 70]
for i, phi in enumerate(phis):
    mu = np.tan(np.radians(phi))

    delta_nu = 1 / (1 / mu + 1)
    Ms = np.logspace(1, 3, 100)
    delta_phis = []

    for M in Ms:
        mu_eff = 1 / (1 / (delta_nu + 1.0 / (2 * M)) - 1)
        delta_phi = np.degrees(np.arctan(mu_eff)) - np.degrees(np.arctan(mu))
        delta_phis.append(delta_phi)

    # delta_phi = np.degrees(np.arctan(delta_mus))
    color = cmap(i / (len(phis) - 1))

    plt.loglog(Ms, delta_phis, label=rf"$\varphi={phi:0.0f}^\circ$", c=color)
plt.xlabel("M")
# plt.ylabel(r'$\Delta\mu$')
plt.ylabel(r"$\Delta\varphi$ ($^\circ$)")
# plt.yticks([0.1, 1, 10, 100])
plt.legend(
    # loc=0,
    loc="upper right",
    bbox_to_anchor=(1.04, 1.04),
    # labelspacing=0.25,
    handlelength=1,
)
plt.subplots_adjust(left=0.2, right=0.99, top=0.95, bottom=0.2)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic SLM/im/discretisation_error.pdf"))

plt.clf()
M = 100
delta_M = 1.0 / M
for m in range(1, M):
    # if m == 1:
    # psi = 90
    # else:
    delta_nu = m * delta_M
    mu = 1 / (1 / delta_nu - 1)
    psi = np.degrees(np.arctan(mu))
    plt.plot(delta_nu, psi, "k.")
plt.xlabel(r"$\Delta\nu$")
plt.ylabel(r"$\psi$ ($^\circ$)")
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic SLM/im/possible_values.pdf"))
