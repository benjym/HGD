from pathlib import Path
import os
import re

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("papers/HGD/paper.mplstyle")
from HGD.params import load_file


def parse_float(name: str, prefix: str) -> float | None:
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix) :])
    except ValueError:
        return None


def find_last_step(data_dir: Path, field: str) -> int | None:
    pattern = re.compile(rf"{re.escape(field)}_(\d+)\.npy$")
    steps: list[int] = []
    for item in data_dir.iterdir():
        match = pattern.search(item.name)
        if match:
            steps.append(int(match.group(1)))
    return max(steps) if steps else None


def reduce_to_2d(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 2:
        arr = np.nanmean(arr, axis=-1)
    return arr


def load_final_nu(data_dir: Path) -> np.ndarray | None:
    if not data_dir.exists():
        return None
    step = find_last_step(data_dir, "nu")
    if step is None:
        return None
    nu = np.load(data_dir / f"nu_{step:06d}.npy")
    return reduce_to_2d(nu)


def mean_bottom_half(nu: np.ndarray) -> float:
    arr = nu.astype(float, copy=True)
    arr[arr == 0] = np.nan
    ny = arr.shape[1]
    bottom = arr[:, : ny // 2]
    return float(np.nanmean(bottom))


def ig_bidisperse(s_m: float, s_M: float, large_conc: float) -> float:
    large = float(large_conc)
    if large <= 0.0 or large >= 1.0 or s_M <= s_m:
        return 1.0
    ln_small = np.log(s_m)
    ln_large = np.log(s_M)
    ln_mean = large * ln_large + (1.0 - large) * ln_small
    var = large * (ln_large - ln_mean) ** 2 + (1.0 - large) * (ln_small - ln_mean) ** 2
    return float(np.exp(np.sqrt(var)))


def main() -> None:
    json_path = Path("papers/percolation/json/yu_standish.json5")
    with json_path.open("r") as f:
        _, p = load_file(f)

    s_m = float(p.s_m)
    base_dir = Path("output/yu_standish")

    results_loose: dict[float, list[tuple[float, float, float]]] = {}
    results_dense: dict[float, list[tuple[float, float, float]]] = {}

    for s_dir in sorted(base_dir.glob("s_M_*")):
        s_M = parse_float(s_dir.name, "s_M_")
        if s_M is None:
            continue
        ratio = s_M / s_m

        for conc_dir in sorted(s_dir.glob("large_concentration_*")):
            conc = parse_float(conc_dir.name, "large_concentration_")
            if conc is None:
                continue

            ig_val = ig_bidisperse(s_m, s_M, conc)
            for swap_dir in sorted(conc_dir.glob("applied_swap_rate_*")):
                swap_rate = parse_float(swap_dir.name, "applied_swap_rate_")
                if swap_rate is None:
                    continue
                nu = load_final_nu(swap_dir / "data")
                if nu is None:
                    continue
                mean_nu = mean_bottom_half(nu)
                if swap_rate == 0:
                    results_loose.setdefault(ratio, []).append((conc, mean_nu, ig_val))
                elif swap_rate == 0.1:
                    results_dense.setdefault(ratio, []).append((conc, mean_nu, ig_val))

    if not results_loose and not results_dense:
        raise RuntimeError("No yu_standish output data found to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    ratios = sorted(set(results_loose.keys()) | set(results_dense.keys()))
    cmap = plt.get_cmap("viridis", max(len(ratios), 1))

    for i, ratio in enumerate(ratios):
        color = cmap(i)
        loose = sorted(results_loose.get(ratio, []), key=lambda t: t[0])
        dense = sorted(results_dense.get(ratio, []), key=lambda t: t[0])

        if loose:
            concs = [d[0] for d in loose]
            means = [d[1] for d in loose]
            n = 1 - np.array(means)
            ax.plot(
                concs,
                n,
                marker="o",
                ls="-",
                color=color,
                label=rf"$s_M/s_m={ratio:g}$ loose",
            )

        if dense:
            concs = [d[0] for d in dense]
            means = [d[1] for d in dense]
            n = 1 - np.array(means)
            ax.plot(
                concs,
                n,
                marker="s",
                ls="--",
                color=color,
                label=rf"$s_M/s_m={ratio:g}$ dense",
            )

    ax.set_xlabel(r"$c_L$")
    ax.set_ylabel(r"Porosity, $n$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    os.makedirs("papers/percolation/figures", exist_ok=True)
    plt.savefig("papers/percolation/figures/yu_standish_final_nu.png", dpi=300)

    fig_ig, ax_ig = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    for i, ratio in enumerate(ratios):
        color = cmap(i)
        loose = sorted(results_loose.get(ratio, []), key=lambda t: t[0])
        dense = sorted(results_dense.get(ratio, []), key=lambda t: t[0])

        if loose:
            concs = [d[0] for d in loose]
            means = [d[1] for d in loose]
            ig_vals = [d[2] for d in loose]
            n = 1 - np.array(means)
            ax_ig.plot(
                ig_vals,
                n,
                marker="o",
                ls="-",
                color=color,
                label=rf"$s_M/s_m={ratio:g}$ loose",
            )

        if dense:
            concs = [d[0] for d in dense]
            means = [d[1] for d in dense]
            ig_vals = [d[2] for d in dense]
            n = 1 - np.array(means)
            ax_ig.plot(
                ig_vals,
                n,
                marker="s",
                ls="--",
                color=color,
                label=rf"$s_M/s_m={ratio:g}$ dense",
            )

    ax_ig.set_xlabel(r"$I_G$")
    ax_ig.set_ylabel(r"Porosity, $n$")
    ax_ig.grid(True, alpha=0.3)
    ax_ig.legend(frameon=False)

    plt.savefig("papers/percolation/figures/yu_standish_ig.png", dpi=300)

    fig_c, ax_c = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    for i, ratio in enumerate(ratios):
        color = cmap(i)
        loose = results_loose.get(ratio, [])
        dense = results_dense.get(ratio, [])

        loose_map = {d[0]: 1 - d[1] for d in loose}
        dense_map = {d[0]: 1 - d[1] for d in dense}
        concs = sorted(set(loose_map.keys()) & set(dense_map.keys()))
        if not concs:
            continue
        contraction = [loose_map[c] - dense_map[c] for c in concs]
        ax_c.plot(
            concs,
            contraction,
            marker="^",
            ls="-",
            color=color,
            label=rf"$s_M/s_m={ratio:g}$",
        )

    ax_c.set_xlabel(r"$c_L$")
    ax_c.set_ylabel(r"Contraction, $\Delta n$")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(frameon=False)

    plt.savefig("papers/percolation/figures/yu_standish_contraction.png", dpi=300)


if __name__ == "__main__":
    main()
