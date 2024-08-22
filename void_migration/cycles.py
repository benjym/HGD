import numpy as np

op_arr = []


def update(s, c, p, t, p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile):
    Mass_inside = np.count_nonzero(~np.isnan(s)) * p.M_of_each_cell
    p = charge_discharge(p, t, Mass_inside)
    p_count[t], p_count_s[t], p_count_l[t], non_zero_nu_time[t] = save_quantities(p, s)
    if p.get_ht is True:
        ht = get_profile(s, c, p, t)
        surface_profile.append(ht)

    return p_count, p_count_s, p_count_l, non_zero_nu_time, surface_profile


def charge_discharge(p, t, Mass_inside):
    """
    As of now two times (t_fill and t_empty) are calculated and t_settle is given
    t_fill - filling time
    t_settle - allow the cells to settle
    t_empty - end time
    for mono-disperse conditions, increase very slightly the grainsize for different cycles
    (mainly to assign different colors for different cycle), and
    this is done in the time_march function
    """

    p.get_ht = False
    res = [sub["t_settle2"] for sub in op_arr]  # Just take the t_empty values from the op_arr dictionary

    yj = []
    for j in range(len(res)):
        if t < int(np.ceil(res[j] / p.dt)):
            yj.append(op_arr[j])

    if t <= int(
        np.ceil(yj[0].get("t_fill") / p.dt)
    ):  # To always pick the time belonging to the first element in the array
        p.half_width = p.half_width_top
        if p.gsd_mode == "mono":
            p.s_M = p.s_m
            p.boundaries = ["place_on_top"]  # 'pour_base'
        else:
            p.boundaries = ["place_on_top"]

    elif int(np.ceil(yj[0].get("t_fill") / p.dt)) < t <= int(np.ceil(yj[0].get("t_settle1") / p.dt)):
        if t == int(np.ceil(yj[0].get("t_settle1") / p.dt)):
            p.get_ht = True
        p.boundaries = ["None"]

    elif int(np.ceil(yj[0].get("t_settle1") / p.dt)) < t <= int(np.ceil(yj[0].get("t_empty") / p.dt)):
        p.half_width = p.half_width_bottom

        if Mass_inside > p.Mass_remain_cumsum[len(op_arr) - len(yj)]:
            if p.silo_width == "half":
                p.boundaries = ["wall"]
            elif p.silo_width == "full":
                p.boundaries = ["central_outlet"]
            p.save_outlet = True
        else:
            p.boundaries = ["None"]

    elif int(np.ceil(yj[0].get("t_empty") / p.dt)) < t < int(np.ceil(yj[0].get("t_settle2") / p.dt)):
        if t == int(np.ceil(yj[0].get("t_settle2") / p.dt)) - 1:
            p.get_ht = True
        p.boundaries = ["None"]

    p.current_cycle = len(op_arr) - len(yj) + 1
    p.num_cycles = len(op_arr)

    return p


def cals_in_cd(p):
    """
    Calculates the time required for filling, settling and emptying for each cycle and a desired number of cycles
    M_total : Total mass for a given solid fraction
    Mass_in_per_u_t : Mass going in per unit time
    Mass_out_per_u_t : Mass coming out of silo per unit time
    T_in : filling time calculated based on M_total, Mass_in_per_u_t and p.dt
    T_settle : settling time
    T_out : emptying time calculated based on M_in, Mass_out_per_u_t and p.dt
    op_arr : holds the values for t_fill, t_settle and t_empty for each cycle and for a desired number of cycles

    """
    if hasattr(p, "give_mass"):
        p.M_of_each_cell = p.dx * p.dy * p.rho_p / p.nm

        # Total mass = no. of non-nan cells * Mass of each cell
        M_total = p.nx * p.ny * p.nm * p.M_of_each_cell * p.nu_cs

        # Mass per unit time
        if p.silo_width == "half":
            Mass_in_per_u_t = p.half_width_top * p.nm * p.fill_ratio * p.M_of_each_cell
            Mass_out_per_u_t = p.half_width_bottom * p.nm * p.free_par * p.M_of_each_cell
        elif p.silo_width == "full":
            Mass_in_per_u_t = (p.half_width_top * 2) * p.nm * p.fill_ratio * p.M_of_each_cell
            Mass_out_per_u_t = (p.half_width_bottom * 2) * p.nm * p.free_par * p.M_of_each_cell

        # Mass in
        Mass_in = M_total * np.array(list(p.M_ins.values()))

        # Mass out
        Mass_out = Mass_in * np.array(list(p.M_empty.values()))

        # Mass remaining
        p.Mass_remain = Mass_in - Mass_out
        p.Mass_remain_cumsum = np.cumsum(p.Mass_remain)

        # Time in
        T_in = [
            int((Mass_in[0] / Mass_in_per_u_t) * p.dt),
            int((Mass_in[1] / Mass_in_per_u_t) * p.dt),
            int((Mass_in[2] / Mass_in_per_u_t) * p.dt),
        ]

        # Time settle - stage1
        T_settle1 = np.array(list(p.T_settle1.values()))

        # Time out
        T_out = [
            int((Mass_out[0] / Mass_out_per_u_t) * p.dt),
            int((Mass_out[1] / Mass_out_per_u_t) * p.dt),
            int((Mass_out[2] / Mass_out_per_u_t) * p.dt),
        ]

        # Time settle - stage2
        T_settle2 = np.array(list(p.T_settle2.values()))

        tmp = 0

        for i in range(p.no_of_cycles):
            T_in[i] = T_in[i] + tmp
            T_settle1[i] = T_settle1[i] + T_in[i]
            T_out[i] = T_out[i] + T_settle1[i]
            T_settle2[i] = T_settle2[i] + T_out[i]
            # tmp = T_out[i]
            tmp = T_settle2[i]

    if hasattr(p, "give_time") and hasattr(p, "T_in") and hasattr(p, "T_settle") and hasattr(p, "T_empty"):
        T_in = list(p.T_in.values())
        T_settle = list(p.T_settle.values())
        T_out = list(p.T_empty.values())
        tmp = 0
        for i in range(p.no_of_cycles):
            T_in[i] = T_in[i] + tmp
            T_settle[i] = T_settle[i] + T_in[i]
            T_out[i] = T_out[i] + T_settle[i]
            tmp = T_out[i]

    all_Ts = np.transpose([T_in, T_settle1, T_out, T_settle2])  # to arange the times in a cycle fashion

    # Initialize dictionary
    times = ["t_fill", "t_settle1", "t_empty", "t_settle2"]

    for i in range(p.no_of_cycles):
        op_arr.append({x: {} for x in times})  # creating empty dictionary with empty times

    for j in range(p.no_of_cycles):
        op_arr[j] = {
            key: all_Ts[j][i] for i, key in enumerate(op_arr[j])
        }  # assigned the values for times t_fill t_settle and t_empty

    return op_arr


def save_quantities(p, s):
    p_count = 0
    p_count_s = 0
    p_count_l = 0
    non_zero_nu_time = 0

    if p.gsd_mode == "mono":
        p_count = np.count_nonzero(s[~np.isnan(s)])
    elif p.gsd_mode == "bi":
        p_count = np.count_nonzero(s[~np.isnan(s)])
        p_count_s = np.count_nonzero(s[~np.isnan(s)] == p.s_m)
        p_count_l = np.count_nonzero(s[~np.isnan(s)] == p.s_M)

    non_zero_avg_nu = 1 - np.mean(np.isnan(s), axis=2)  # get the solid fraction
    non_zero_nu_time = np.mean(
        non_zero_avg_nu[np.nonzero(non_zero_avg_nu)]
    )  # get the avg solid fraction ignoring the zeros

    return p_count, p_count_s, p_count_l, non_zero_nu_time


def set_nt(p):
    if (hasattr(p, "give_mass")) or (
        hasattr(p, "give_time") and hasattr(p, "T_in") and hasattr(p, "T_settle") and hasattr(p, "T_empty")
    ):
        op_arr = cals_in_cd(p)  # get the time cycles

    else:
        op_arr = p.T_cycles
    print("OOOOOOOOOOOOOOOOO", op_arr, int(np.ceil(op_arr[-1].get("t_settle2"))))
    return int(np.ceil(op_arr[-1].get("t_settle2") / p.dt))  # cal the number of steps based on final t_empty


def get_profile(s, c, p, t):
    # if p.get_ht == True:

    nm = s.shape[2]
    mask = np.sum(np.isnan(s), axis=2) > 0.95 * nm

    creq = np.ma.masked_where(mask, np.nanmean(c, axis=2))

    # den = 1 - np.mean(np.isnan(s), axis=2)
    # den = np.ma.masked_where(den < p.nu_cs / 7.0, den)

    # if p.current_cycle == 1:
    #     val = p.current_cycle
    # else:
    #     val = (p.current_cycle - 1 + p.current_cycle) / 2

    ht = []

    for w in range(p.nx):
        if np.argmin(creq[w]) == 0 and np.ma.is_masked(np.ma.max(creq[w])):
            ht.append(0)
        else:
            ht.append(np.max(np.nonzero(creq[w] == np.max(creq[w]))))

    np.save(p.folderName + "c_" + str(t).zfill(6) + ".npy", np.nanmean(c, axis=2))
    return ht
