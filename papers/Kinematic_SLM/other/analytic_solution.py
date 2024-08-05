import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from void_migration.params import load_file
from matplotlib.colors import LogNorm

plt.style.use("papers/Kinematic_SLM/paper.mplstyle")

with open("papers/Kinematic_SLM/json/analytic.json5") as f:
    dict, p = load_file(f)
    p.set_defaults()
    p.update_before_time_march(None)

W = p.H * (p.nx / p.ny)
x = np.linspace(-W / 2, W / 2, p.nx)
y = np.linspace(0, p.H, p.ny)
dx = x[1] - x[0]
X, Y = np.meshgrid(x, y)

D = p.diffusivity
u = p.free_fall_velocity

# at x=0 and y=0, the concentration is c = Q/(4*pi*D*u). We want this to be unity, so
Q = 4 * np.pi * D * u  # NOTE: THIS DOESN'T SEEM TO WORK VERY WELL - ROUNDING ERRORS OR SOMETHING WEIRD?
# Q = 1

with np.errstate(divide="ignore", invalid="ignore"):
    c = Q / np.sqrt(4 * np.pi * D * u * Y) * np.exp(-u * X**2 / (4 * D * Y))

plt.pcolormesh(X, Y, 1 - c, vmin=0, vmax=1)  # , norm=LogNorm())
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Analytic solution")
plt.show()
