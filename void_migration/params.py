import os
import sys
import json5
import numpy as np


class dict_to_class:
    """
    A convenience class to store the information from the parameters dictionary. Used because I prefer using p.variable to p['variable'].
    """

    def __init__(self, dict: dict):
        list_keys: List[str] = []
        lists: List[List] = []
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list) and key not in ["save", "plot", "videos", "T_cycles"]:
                list_keys.append(key)
                lists.append(dict[key])
        setattr(self, "lists", lists)
        setattr(self, "list_keys", list_keys)

    def set_defaults(self):
        with open("json/defaults.json5", "r") as f:
            defaults_dict = json5.loads(f.read())

        for key in defaults_dict:
            if not hasattr(self, key):
                setattr(self, key, defaults_dict[key])

        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)
        if len(self.save) > 0:
            if not os.path.exists(self.folderName + "data/"):
                os.makedirs(self.folderName + "data/")
        if hasattr(self, "aspect_ratio_y"):
            self.ny = int(self.nx * self.aspect_ratio_y)
        if hasattr(self, "aspect_ratio_m"):
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


def load_file(f):
    # parse file
    dict = json5.loads(f.read())
    if len(sys.argv) > 1:
        dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
    p = dict_to_class(dict)
    return dict, p
