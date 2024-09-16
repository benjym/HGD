def update(p, t, state):
    if len(p.cycles) > 0:
        this_cycle = p.cycles[0]

        p.size_choices = [45e-6, 53e-6, 76e-6, 106e-6, 150e-6]
        p.size_weights = [
            this_cycle["-45"],
            this_cycle["-53"],
            this_cycle["-76"],
            this_cycle["-106"],
            this_cycle["+150"],
        ]
        if this_cycle["mode"] == "charge":
            if p.inlet >= this_cycle["mass"]:
                p.boundaries = []
                this_cycle["completed"] = True
            else:
                p.boundaries = ["charge"]
        elif this_cycle["mode"] == "discharge":
            if p.outlet >= this_cycle["mass"]:
                p.boundaries = []
                this_cycle["completed"] = True
            else:
                p.boundaries = ["central_outlet"]

        if this_cycle["completed"] and p.stopped_times > p.stop_after / 2:
            print(f"Cycle {p.n_cycles - len(p.cycles) + 1}/{p.n_cycles} completed")
            p.cycles.pop(0)
            p.inlet = 0
            p.outlet = 0
    else:
        p.boundaries = []

    return p
