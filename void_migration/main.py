#!/usr/bin/python

__doc__ = """
void_migration.py

This script simulates the migration of voids in a granular material.
"""
__author__ = "Benjy Marks, Shivakumar Athani"
__version__ = "0.3"

import sys
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
from itertools import product
import importlib

# from numba import jit, njit
from void_migration import params
from void_migration import plotter
from void_migration import thermal
from void_migration import cycles
from void_migration import initial
from void_migration import stress
from void_migration import boundary

# import void_migration.params as params
# import void_migration.plotter as plotter
# import void_migration.thermal as thermal
# import void_migration.motion as motion
# import void_migration.cycles as cycles
# import void_migration.initial as initial
# import void_migration.stress as stress


def init(p):
    p.move_voids = importlib.import_module(f"void_migration.motion.{p.motion_model}").move_voids

    plotter.set_plot_size(p)

    p.update_before_time_march(cycles)

    if p.show_optimal_resolution:
        p.print_optimal_resolution()

    s = initial.IC(p)  # non-dimensional size

    u = np.zeros([p.nx, p.ny])
    v = np.zeros([p.nx, p.ny])
    p_count = np.zeros([p.nt])
    p_count_s = np.zeros([p.nt])
    p_count_l = np.zeros([p.nt])
    non_zero_nu_time = np.zeros([p.nt])

    # last_swap is used to keep track of the last time a void was swapped
    # start off homoegeous and nan where s is voids
    last_swap = np.zeros_like(s)
    # last_swap[np.isnan(s)] = np.nan

    c = initial.set_concentration(s, p.X, p.Y, p)

    initial.set_boundary(s, p.X, p.Y, p)

    if hasattr(p, "temperature"):
        T = p.temperature["inlet_temperature"] * np.ones_like(s)
        T[np.isnan(s)] = np.nan
    else:
        T = None

    if p.calculate_stress:
        sigma = stress.calculate_stress(s, last_swap, p)
    else:
        sigma = None

    outlet = []
    surface_profile = []

    N_swap = None
    p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(p.indices)

    state = (
        s,
        u,
        v,
        c,
        T,
        p_count,
        p_count_s,
        p_count_l,
        non_zero_nu_time,
        N_swap,
        last_swap,
        sigma,
        outlet,
        surface_profile,
    )

    if len(p.save) > 0:
        plotter.save_coordinate_system(p)
    plotter.update(p, state, 0)

    return state


def time_step(p, state, t):
    if p.queue2 is not None:
        while not p.queue2.empty():
            update = p.queue2.get()
            for key, value in update.items():
                print("Updating parameter {} to {}".format(key, value))
                setattr(p, key, value)

    (
        s,
        u,
        v,
        c,
        T,
        p_count,
        p_count_s,
        p_count_l,
        non_zero_nu_time,
        N_swap,
        last_swap,
        sigma,
        outlet,
        surface_profile,
    ) = state

    if p.stop_event is not None and p.stop_event.is_set():
        raise KeyboardInterrupt

    u = np.zeros_like(u)
    v = np.zeros_like(v)

    if p.calculate_temperature:
        T = thermal.update_temperature(s, T, p)

    if p.calculate_stress:
        sigma = stress.calculate_stress(s, last_swap, p)

    if p.charge_discharge:
        p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile = cycles.update(
            s, c, p, t, p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile
        )

    u, v, s, c, T, N_swap, last_swap = p.move_voids(u, v, s, p, c=c, T=T, N_swap=N_swap, last_swap=last_swap)

    u, v, s, c, outlet = boundary.update(u, v, s, p, c, outlet)

    if t % p.save_inc == 0:
        plotter.update(p, state, t)

    return (
        s,
        u,
        v,
        c,
        T,
        p_count,
        p_count_s,
        p_count_l,
        non_zero_nu_time,
        N_swap,
        last_swap,
        sigma,
        outlet,
        surface_profile,
    )


def time_march(p):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    state = init(p)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        state = time_step(
            p,
            state,
            t,
        )

    plotter.update(p, state, t)


def run_simulation(sim_with_index):
    index, sim = sim_with_index
    with open(sys.argv[1], "r") as f:
        dict, p_init = params.load_file(f)
    folderName = f"output/{dict['input_filename']}/"
    dict_copy = dict.copy()
    for i, key in enumerate(p_init.list_keys):
        dict_copy[key] = sim[i]
        folderName += f"{key}_{sim[i]}/"
    p = params.dict_to_class(dict_copy)
    p.concurrent_index = index
    p.folderName = folderName
    p.set_defaults()
    time_march(p)
    plotter.make_video(p)
    return folderName


if __name__ == "__main__":
    with open(sys.argv[1], "r") as f:
        dict, p_init = params.load_file(f)
        if not hasattr(p_init, "max_workers"):
            p_init.max_workers = None

    # run simulations
    all_sims = list(product(*p_init.lists))

    folderNames = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=p_init.max_workers) as executor:
        results = list(
            tqdm(
                executor.map(run_simulation, enumerate(all_sims)),
                total=len(all_sims),
                desc="Sim",
                leave=False,
            )
        )

    folderNames.extend(results)

    if len(all_sims) > 1:
        plotter.stack_videos(folderNames, dict["input_filename"], p_init.videos)
