import sys
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
from itertools import product
import importlib

# from numba import jit, njit
from HGD import params
from HGD import plotter
from HGD import thermal
from HGD import initial
from HGD import stress
from HGD import boundary


def init(p, cycles=None):
    p.move_voids = importlib.import_module(f"HGD.motion.{p.motion_model}").move_voids

    plotter.set_plot_size(p)

    p.update_before_time_march(cycles)

    if p.show_optimal_resolution:
        p.print_optimal_resolution()

    s = initial.IC(p)  # non-dimensional size

    u = np.zeros([p.nx, p.ny, p.nm])
    v = np.zeros([p.nx, p.ny, p.nm])
    # p_count = np.zeros([p.nt])
    # p_count_s = np.zeros([p.nt])
    # p_count_l = np.zeros([p.nt])
    # non_zero_nu_time = np.zeros([p.nt])
    outlet = np.zeros([p.nt])
    # surface_profile = np.zeros([p.nt])

    # last_swap is used to keep track of the last time a void was swapped
    # start off homoegeous and nan where s is voids
    last_swap = np.zeros_like(s)
    chi = np.zeros([p.nx, p.ny])
    # last_swap[np.isnan(s)] = np.nan
    # chi = np.zeros([p.nx, p.ny, 2])

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

    p.indices = np.arange(p.nx * (p.ny - 1) * p.nm)
    np.random.shuffle(p.indices)

    state = (
        s,
        u,
        v,
        c,
        T,
        # p_count,
        # p_count_s,
        # p_count_l,
        # non_zero_nu_time,
        last_swap,
        chi,
        sigma,
        outlet,
        # surface_profile,
    )

    if len(p.save) > 0:
        plotter.save_coordinate_system(p)
    plotter.update(p, state, 0)

    return state


def time_step(p, state):
    if p.queue2 is not None:
        while not p.queue2.empty():
            update = p.queue2.get()
            if type(update) is str:
                if update == "Save state":
                    save_state(p, state)
                elif update == "Load state":
                    state = load_state(p)
            elif type(update) is dict:
                for key, value in update.items():
                    print("Updating parameter {} to {}".format(key, value))
                    setattr(p, key, value)
            elif type(update) is list:
                p.process_charge_discharge_csv(update)
            else:
                print("Unknown update: {}".format(update))

    (
        s,
        u,
        v,
        c,
        T,
        # p_count,
        # p_count_s,
        # p_count_l,
        # non_zero_nu_time,
        last_swap,
        chi,
        sigma,
        outlet,
        # surface_profile,
    ) = state

    p.update_every_time_step(state)

    if p.stop_event is not None and p.stop_event.is_set():
        raise KeyboardInterrupt

    if p.calculate_temperature:
        T = thermal.update_temperature(s, T, p)

    if p.calculate_stress:
        sigma = stress.calculate_stress(s, last_swap, p)

    # if p.charge_discharge:
    #     p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile = cycles.update(
    #         s, c, p, t, p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile
    #     )
    if len(p.cycles) > 0:
        p = cycles.update(p, state)

    u, v, s, c, T, chi, last_swap = p.move_voids(u, v, s, p, c=c, T=T, chi=chi, last_swap=last_swap)

    state = (
        s,
        u,
        v,
        c,
        T,
        # p_count,
        # p_count_s,
        # p_count_l,
        # non_zero_nu_time,
        last_swap,
        chi,
        sigma,
        outlet,
        # surface_profile,
    )

    state = boundary.update(p, state)

    if p.tstep % p.save_inc == 0:
        plotter.update(p, state)

    # if all voids are stopped
    is_stopped = chi.sum() < 1.0 / (p.nx * p.ny * p.nm)

    if is_stopped:
        p.stopped_times += 1
    else:
        p.stopped_times = 0

    p.t += p.dt
    p.tstep += 1

    return state


def time_march(p, cycles=None):
    """
    Run the actual simulation(s) as defined in the input json file `p`.
    """

    state = init(p, cycles)
    if p.t_f is not None:
        for tstep in tqdm(range(1, p.nt), leave=False, desc="Time", position=p.concurrent_index + 1):
            state = time_step(p, state)
    else:
        chi_progress_bar = tqdm(total=1.0, leave=False, desc=p.this_sim, position=p.concurrent_index + 1)

        p.stopped_times = 0
        while not p.stop_event:
            chi = state[6]
            # Update progress bar for chi using a log scale
            with np.errstate(divide="ignore"):
                progress = (np.log10(p.max_chi) - np.log10(chi.mean())) / (
                    np.log10(p.max_chi) - np.log10(p.min_chi)
                )
            chi_progress_bar.n = max(0, min(1.0, progress))  # Ensure bounds
            chi_progress_bar.refresh()

            state = time_step(p, state)

            if p.stopped_times > p.stop_after:
                p.stop_event = True

    plotter.update(p, state)


def run_simulation(sim_with_index):
    index, sim = sim_with_index
    with open(sys.argv[1], "r") as f:
        dict, p_init = params.load_file(f)
    folderName = f"output/{dict['input_filename']}/"
    dict_copy = dict.copy()
    this_sim = ""
    for i, key in enumerate(p_init.list_keys):
        dict_copy[key] = sim[i]
        folderName += f"{key}_{sim[i]}/"
        this_sim += f"{key}={sim[i]},"
    p = params.dict_to_class(dict_copy)
    p.this_sim = this_sim[:-1]
    p.concurrent_index = index
    p.folderName = folderName
    p.set_defaults()
    time_march(p)
    plotter.make_video(p)
    return folderName


def save_state(p, state):
    new_state = []
    for i, d in enumerate(state):
        if d is None:
            print("State[{}] is None".format(i))
            new_state.append([-1])
        else:
            new_state.append(d)
    np.savez(p.folderName + "state.npz", *new_state)


def load_state(p):
    data = np.load(p.folderName + "state.npz")
    state = []
    for key in data.files:
        if data[key].shape == (1,) and data[key][0] == -1:
            state.append(None)
        else:
            state.append(data[key])
    return state


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
