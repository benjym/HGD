import os
import sys
import json5
import numpy as np


class dict_to_class:
    """
    A convenience class to store the information from the parameters dictionary. Used because I prefer using p.variable to p['variable'].
    """

    def __init__(self, dict: dict):
        list_keys: List[str] = []  # noqa
        lists: List[List] = []  # noqa
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list) and key not in [
                "save",
                "plot",
                "videos",
                "T_cycles",
                "boundaries",
            ]:
                list_keys.append(key)
                lists.append(dict[key])
        setattr(self, "lists", lists)
        setattr(self, "list_keys", list_keys)

    def print_optimal_resolution(p):
        # NOTE: Well defined for monodisperse conditions only
        # Optimal resolution is when P_adv = 1 and P_diff=0.5.
        # Under these conditions we have that
        # dx = 2*alpha*s_bar

        # So for any given s_bar and alpha there is an optimal spatial resolution
        s_bar = (p.s_m + p.s_M) / 2.0  # mean diameter (m)
        dx = 2 * p.alpha * s_bar
        # dt = p.dx / np.sqrt(p.g * s_bar)

        ny = int(p.H / dx)
        print(f"Optimal condition: ny = {ny}.")

    def set_defaults(self):
        defaults_path = resource_path("json/defaults.json5")
        with open(defaults_path, "r") as f:
            defaults_dict = json5.loads(f.read())

        for key in defaults_dict:
            if not hasattr(self, key):
                setattr(self, key, defaults_dict[key])

        if hasattr(self, "aspect_ratio_y"):
            if isinstance(self.aspect_ratio_y, list):
                self.ny = [int(self.nx * ar) for ar in self.aspect_ratio_y]
            else:
                self.ny = int(self.nx * self.aspect_ratio_y)
        if hasattr(self, "aspect_ratio_m"):
            if isinstance(self.aspect_ratio_m, list):
                self.nm = [int(self.nx * ar) for ar in self.aspect_ratio_m]
            else:
                self.nm = int(self.nx * self.aspect_ratio_m)

        if self.gsd_mode == "mono":
            if hasattr(self, "s_m") and hasattr(self, "s_M"):
                if not self.s_m == self.s_M:
                    print("WARNING: s_m and s_M must be equal for monodisperse grains. Setting both to s_m.")
                    self.s_M = self.s_m
            if hasattr(self, "s_m"):
                self.s_M = self.s_m
            if hasattr(self, "s_M"):
                self.s_m = self.s_M
            else:
                self.s_m = 1
                self.s_M = 1

        # user can define mu or repose_angle. If both defined, mu takes precedence.
        if hasattr(self, "mu"):
            self.repose_angle = np.degrees(np.arctan(self.mu))
        if hasattr(self, "repose_angle"):
            self.mu = np.tan(np.radians(self.repose_angle))

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_mu = np.nan_to_num(1.0 / self.mu, nan=0.0, posinf=1e30, neginf=0.0)
        self.delta_limit = self.nu_cs / (inv_mu + 1)

    def update_before_time_march(self, cycles):
        self.y = np.linspace(0, self.H, self.ny)
        self.dy = self.y[1] - self.y[0]
        self.x = np.arange(
            -(self.nx - 0.5) / 2 * self.dy, (self.nx - 0.5) / 2 * self.dy, self.dy
        )  # force equal grid spacing
        self.dx = self.x[1] - self.x[0]
        if not np.isclose(self.dx, self.dy):
            sys.exit(f"Fatal error: dx != dy. dx = {self.dx}, dy = {self.dy}")

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.y += self.dy / 2.0
        self.t = 0

        # self.t_p = self.s_m / np.sqrt(self.g * self.H)  # smallest confinement timescale (at bottom) (s)
        s_bar = (self.s_m + self.s_M) / 2.0  # mean diameter (m)
        if self.advection_model == "average_size":
            self.free_fall_velocity = np.sqrt(self.g * s_bar)  # typical speed to fall one mean diameter (s)
        elif self.advection_model == "freefall":
            self.free_fall_velocity = np.sqrt(
                2 * self.g * self.dy
            )  # typical speed to fall one grid spacing (m/s)
        elif self.advection_model == "stress":
            max_pressure = self.solid_density * self.g * self.H
            self.free_fall_velocity = np.sqrt(
                2 * max_pressure / self.solid_density
            )  # confinement velocity (m/s)

        self.diffusivity = self.alpha * self.free_fall_velocity * s_bar  # diffusivity (m^2/s)

        safe = False
        stability = 0.5
        while not safe:
            self.P_u_ref = stability
            self.dt = self.P_u_ref * self.dy / self.free_fall_velocity

            self.P_lr_ref = (
                self.diffusivity * self.dt / self.dy**2
            )  # ignoring factor of 2 because that assumes P=0.5
            # self.P_lr_ref = self.alpha * self.P_u_ref

            self.P_u_max = self.P_u_ref * (self.s_M / self.s_m)
            self.P_lr_max = self.P_lr_ref * (self.s_M / self.s_m)

            if self.motion_model == "d2q4_array":
                if self.P_u_max <= 1 and self.P_lr_max <= self.P_stab:
                    safe = True
                else:
                    stability *= 0.95
            else:
                if self.P_u_max + 2 * self.P_lr_max <= self.P_stab:
                    safe = True
                else:
                    stability *= 0.95

        # print(f"P_u_ref : {self.P_u_ref}")
        # print(f"P_u_max : {self.P_u_max}")
        # print(f"P_lr_ref : {self.P_lr_ref}")
        # print(f"P_lr_max : {self.P_lr_max}")

        if self.charge_discharge:
            self.nt = cycles.set_nt(self)
        else:
            if self.t_f is None:
                self.nt = 0
            else:
                self.nt = int(np.ceil(self.t_f / self.dt))

        if hasattr(self, "saves"):
            if self.nt == 0:
                sys.exit("Fatal error: nt = 0. Cannot use `saves` parameter.")
            else:
                self.save_inc = int(self.nt / self.saves)

    def process_charge_discharge_csv(self, array):
        self.cycles = []
        for row in array:
            cycle = {}
            cycle["mode"] = row[0]
            cycle["-45"] = row[1]
            cycle["-53"] = row[2]
            cycle["-76"] = row[3]
            cycle["-106"] = row[4]
            cycle["+150"] = row[5]
            cycle["mass"] = row[6]
            self.cycles.append(cycle)


def load_file(f):
    # parse file
    dict = json5.loads(f.read())
    if len(sys.argv) > 1:
        dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
    p = dict_to_class(dict)
    p.set_defaults()
    return dict, p


def resource_path(relative_path):
    """Get the absolute path to the resource, works for both development and PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
