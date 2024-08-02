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


def init(p, queue=None):
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

    N_swap = None
    p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(p.indices)

    state = s, u, v, c, T, p_count, p_count_s, p_count_l, non_zero_nu_time, N_swap, last_swap, sigma, outlet

    if len(p.save) > 0:
        plotter.save_coordinate_system(p)
    plotter.update(p, state, 0, queue)

    return state


def time_step(
    p,
    state,
    t,
    queue=None,
    stop_event=None,
):
    s, u, v, c, T, p_count, p_count_s, p_count_l, non_zero_nu_time, N_swap, last_swap, sigma, outlet = state
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt

    outlet.append(0)
    u = np.zeros_like(u)
    v = np.zeros_like(v)

    if hasattr(p, "temperature"):
        T = thermal.update_temperature(s, T, p)

    if p.calculate_stress:
        sigma = stress.calculate_stress(s, last_swap, p)

    if p.charge_discharge:
        Mass_inside = np.count_nonzero(~np.isnan(s)) * p.M_of_each_cell
        p = cycles.charge_discharge(p, t, Mass_inside)
        p_count[t], p_count_s[t], p_count_l[t], non_zero_nu_time[t] = cycles.save_quantities(p, s)
        if p.get_ht == True:
            ht = plotter.get_profile(s, c, p, t)
            surface_profile.append(ht)

    u, v, s, c, T, N_swap, last_swap = p.move_voids(u, v, s, sigma, last_swap, p, c=c, T=T, N_swap=N_swap)

    u, v, s, c, outlet = boundary.add_voids(u, v, s, p, c, outlet)

    if p.close_voids:
        u, v, s = boundary.close_voids(u, v, s)

    if t % p.save_inc == 0:
        plotter.update(p, state, t, queue)

        # if hasattr(p, "charge_discharge") and (p.gsd_mode == 'bi' or p.gsd_mode == 'fbi'):
        #     plotter.plot_pdf_cdf(p,s,xpoints,ypoints,t)

        ## to simulate wall motion
        if p.wall_motion:
            if t % p.save_wall == 0:
                s_mean = np.nanmean(s, axis=2)

                start_sim = np.min(np.argwhere(s_mean > 0), axis=0)[
                    0
                ]  # gives the start position of column in x-direction
                end_sim = np.max(np.argwhere(s_mean > 0), axis=0)[
                    0
                ]  # gives the end position of column in x-direction

                if start_sim > 1 and end_sim + 1 < p.nx - 1:
                    # s[start_sim-2:start_sim-1,:,:] = np.nan
                    s[end_sim + 2 : end_sim + 3, :, :] = np.nan

    return s, u, v, c, T, p_count, p_count_s, p_count_l, non_zero_nu_time, N_swap, last_swap, sigma, outlet


def time_march(p, queue=None, stop_event=None):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    state = init(p, queue)

    for t in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
        state = time_step(
            p,
            state,
            t,
            queue,
            stop_event,
        )

    plotter.update(p, state, t, queue)

    if p.charge_discharge:
        plotter.c_d_saves(p, non_zero_nu_time, p_count, p_count_s, p_count_l)
    plotter.update(x, y, s, u, v, c, T, outlet, p, t)
    col_ht = plotter.get_col_depth(s, p)
    np.save(p.folderName + "ht_" + str(p.repose_angle) + ".npy", col_ht)
    np.save(p.folderName + "surface_profiles.npy", surface_profile)


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
