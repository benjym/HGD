import os
import numpy as np
import matplotlib.pyplot as plt
from void_migration.params import load_file

# from matplotlib.colors import LogNorm

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/analytic.json5") as f:
    dict, p = load_file(f)

alphas = p.alpha
aspects = p.aspect_ratio_m

for aspect in aspects:
    for alpha in alphas:
        p.alpha = alpha
        p.aspect_ratio_m = aspect
        p.set_defaults()
        p.update_before_time_march(None)

        # W = p.H * (p.nx / p.ny)
        # x = np.linspace(-W / 2, W / 2, p.nx)
        # dx = x[1] - x[0]
        # y = np.linspace(0, p.H, p.ny)

        X, Y = np.meshgrid(p.x, p.y)

        D = p.diffusivity
        u = p.free_fall_velocity

        # at x=0 and y=0, the concentration is c = Q/(4*pi*D*u). We want this to be unity, so
        Q = (
            4 * np.pi * D * u
        )  # NOTE: THIS DOESN'T SEEM TO WORK VERY WELL - ROUNDING ERRORS OR SOMETHING WEIRD?
        # Q = 1

        with np.errstate(divide="ignore", invalid="ignore"):
            c = Q / np.sqrt(4 * np.pi * D * u * Y) * np.exp(-u * X**2 / (4 * D * Y))

        c_0 = np.nanmax(c)
        c /= c_0
        print(c_0)

        plt.clf()
        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            X,
            Y,
            1 - c,
            # vmin=0,
            # vmax=1,
            #    norm=LogNorm(),
        )
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        # Now compare with simulation output
        f = f"output/analytic/aspect_ratio_m_{aspect}/alpha_{alpha}/data/"
        if os.path.exists(f + f"nu_{p.nt-1:06d}.npy"):
            # x = np.loadtxt(f + "x.csv")
            # y = np.loadtxt(f + "y.csv")
            nu = np.load(f + f"nu_{p.nt-1:06d}.npy")

            print(f"Found alpha = {alpha}")
            plt.subplot(1, 3, 2)
            plt.pcolormesh(
                p.x,
                p.y,
                nu.T,
                vmin=0,
                vmax=1,
                #    norm=LogNorm(),
            )
            plt.colorbar()

            diff = np.abs(nu.T - 1 + c)
            plt.subplot(1, 3, 3)
            plt.pcolormesh(
                p.x,
                p.y,
                diff,
                cmap="bwr",
                vmin=-0.1,
                vmax=0.1,
                #    norm=LogNorm(),
            )
            plt.colorbar()

        plt.savefig(f"papers/Kinematic_SLM/other/analytic_solution_{alpha}_{aspect}.pdf")
