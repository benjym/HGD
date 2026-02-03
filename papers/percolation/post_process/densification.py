from pathlib import Path
import re
import os
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


def ig_uniform(s_m: float, s_M: float, n: int = 20001) -> float:
    if s_M <= s_m:
        return 1.0
    s = np.linspace(s_m, s_M, n)
    f = 1.0 / (s_M - s_m)
    ln_s = np.log(s)
    bar_s = np.exp(np.trapezoid(f * ln_s, s))
    ln_ratio_sq = np.log(s / bar_s) ** 2
    ig = np.exp(np.sqrt(np.trapezoid(f * ln_ratio_sq, s)))
    return float(ig)


def process_s_dir(
    beta_val: float,
    s_dir: Path,
    applied_root: Path,
    s_m: float,
    results_loose: dict[float, list[tuple[float, float, float]]],
    results_dense: dict[float, list[tuple[float, float, float]]],
) -> None:
    s_M = parse_float(s_dir.name, "s_M_")
    if s_M is None:
        return
    ratio = s_M / s_m
    ig = ig_uniform(s_m, s_M)

    loose_dir = applied_root / "applied_swap_rate_0" / "data"
    dense_dir = applied_root / "applied_swap_rate_0.1" / "data"

    nu_loose = load_final_nu(loose_dir)
    nu_dense = load_final_nu(dense_dir)

    if nu_loose is not None:
        mean_loose = mean_bottom_half(nu_loose)
        results_loose.setdefault(beta_val, []).append((ratio, ig, mean_loose))

    if nu_dense is not None:
        mean_dense = mean_bottom_half(nu_dense)
        results_dense.setdefault(beta_val, []).append((ratio, ig, mean_dense))


def main() -> None:
    json_path = Path("papers/percolation/json/densification.json5")
    with json_path.open("r") as f:
        _, p = load_file(f)

    s_m = float(p.s_m)
    base_dir = Path("output/densification")

    results_loose: dict[float, list[tuple[float, float, float]]] = {}
    results_dense: dict[float, list[tuple[float, float, float]]] = {}

    beta_dirs = sorted(base_dir.glob("beta_*"))
    if beta_dirs:
        for beta_dir in beta_dirs:
            beta_val = parse_float(beta_dir.name, "beta_")
            if beta_val is None:
                continue
            for s_dir in sorted(beta_dir.glob("s_M_*")):
                process_s_dir(beta_val, s_dir, s_dir, s_m, results_loose, results_dense)
    else:
        s_dirs = sorted(base_dir.glob("s_M_*"))
        for s_dir in s_dirs:
            beta_subdirs = sorted(s_dir.glob("beta_*"))
            if beta_subdirs:
                for beta_dir in beta_subdirs:
                    beta_val = parse_float(beta_dir.name, "beta_")
                    if beta_val is None:
                        continue
                    process_s_dir(beta_val, s_dir, beta_dir, s_m, results_loose, results_dense)

    if not results_loose and not results_dense:
        raise RuntimeError("No densification output data found to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    betas = sorted(set(results_loose.keys()) | set(results_dense.keys()))
    cmap = plt.get_cmap("viridis", max(len(betas), 1))

    for i, beta_val in enumerate(betas):
        color = cmap(i)
        loose = results_loose.get(beta_val, [])
        dense = results_dense.get(beta_val, [])

        if loose:
            loose_sorted = sorted(loose, key=lambda t: t[0])
            ratios = [d[0] for d in loose_sorted]
            means = [d[2] for d in loose_sorted]
            ax.plot(ratios, means, marker="o", ls="-", color=color, label=rf"$\beta={beta_val:g}$ loose")

        if dense:
            dense_sorted = sorted(dense, key=lambda t: t[0])
            ratios = [d[0] for d in dense_sorted]
            means = [d[2] for d in dense_sorted]
            ax.plot(ratios, means, marker="s", ls="--", color=color, label=rf"$\beta={beta_val:g}$ dense")

    ax.set_xlabel(r"$s_M/s_m$")
    ax.set_ylabel(r"$\nu$")
    # ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    os.makedirs("papers/percolation/figures", exist_ok=True)
    plt.savefig("papers/percolation/figures/densification_final_nu.png", dpi=300)

    fig_ig, ax_ig = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    for i, beta_val in enumerate(betas):
        color = cmap(i)
        loose = results_loose.get(beta_val, [])
        dense = results_dense.get(beta_val, [])

        if loose:
            loose_sorted = sorted(loose, key=lambda t: t[1])
            ig_vals = [d[1] for d in loose_sorted]
            means = [d[2] for d in loose_sorted]
            ax_ig.plot(
                ig_vals,
                1 - np.array(means),
                marker="o",
                ls="-",
                color=color,
                label=rf"$\beta={beta_val:g}$ loose",
            )

        if dense:
            dense_sorted = sorted(dense, key=lambda t: t[1])
            ig_vals = [d[1] for d in dense_sorted]
            means = [d[2] for d in dense_sorted]
            ax_ig.plot(
                ig_vals,
                1 - np.array(means),
                marker="s",
                ls="--",
                color=color,
                label=rf"$\beta={beta_val:g}$ dense",
            )

    ax_ig.set_xlabel(r"$I_G$")
    ax_ig.set_ylabel(r"Porosity, $n$")
    ax_ig.grid(True, alpha=0.3)
    ax_ig.legend(frameon=False)

    plt.savefig("papers/percolation/figures/densification_final_nu_ig.png", dpi=300)


if __name__ == "__main__":
    main()
