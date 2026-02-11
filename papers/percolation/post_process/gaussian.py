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


def normalize_beta_values(beta: object) -> list[float]:
    if isinstance(beta, (list, tuple, np.ndarray)):
        return [float(b) for b in beta]
    return [float(beta)]


def process_s_dir(
    s_std: float,
    applied_root: Path,
    beta_val: float,
    s_mean: float,
    results_loose: dict[float, list[tuple[float, float]]],
    results_dense: dict[float, list[tuple[float, float]]],
) -> None:
    if s_mean == 0:
        return
    ratio = s_std / s_mean

    for swap_dir in sorted(applied_root.glob("applied_swap_rate_*")):
        swap_rate = parse_float(swap_dir.name, "applied_swap_rate_")
        if swap_rate is None:
            continue
        nu = load_final_nu(swap_dir / "data")
        if nu is None:
            continue
        mean_nu = mean_bottom_half(nu)
        if swap_rate == 0:
            results_loose.setdefault(beta_val, []).append((ratio, mean_nu))
        elif swap_rate == 0.1:
            results_dense.setdefault(beta_val, []).append((ratio, mean_nu))


def main() -> None:
    json_path = Path("papers/percolation/json/gaussian.json5")
    with json_path.open("r") as f:
        _, p = load_file(f)

    s_mean = float(p.s_mean)
    base_dir = Path("output/gaussian")

    results_loose: dict[float, list[tuple[float, float]]] = {}
    results_dense: dict[float, list[tuple[float, float]]] = {}

    beta_vals = normalize_beta_values(p.beta)
    beta_dirs = sorted(base_dir.glob("beta_*"))
    if beta_dirs:
        for beta_dir in beta_dirs:
            beta_val = parse_float(beta_dir.name, "beta_")
            if beta_val is None:
                continue
            for s_dir in sorted(beta_dir.glob("s_std_*")):
                s_std = parse_float(s_dir.name, "s_std_")
                if s_std is None:
                    continue
                process_s_dir(s_std, s_dir, beta_val, s_mean, results_loose, results_dense)
    else:
        s_dirs = sorted(base_dir.glob("s_std_*"))
        has_beta_subdirs = any(any(s_dir.glob("beta_*")) for s_dir in s_dirs)
        if has_beta_subdirs:
            for s_dir in s_dirs:
                s_std = parse_float(s_dir.name, "s_std_")
                if s_std is None:
                    continue
                for beta_dir in sorted(s_dir.glob("beta_*")):
                    beta_val = parse_float(beta_dir.name, "beta_")
                    if beta_val is None:
                        continue
                    process_s_dir(s_std, beta_dir, beta_val, s_mean, results_loose, results_dense)
        else:
            beta_val = beta_vals[0]
            for s_dir in s_dirs:
                s_std = parse_float(s_dir.name, "s_std_")
                if s_std is None:
                    continue
                process_s_dir(s_std, s_dir, beta_val, s_mean, results_loose, results_dense)

    if not results_loose and not results_dense:
        raise RuntimeError("No gaussian output data found to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.2), layout="constrained")

    betas = sorted(set(results_loose.keys()) | set(results_dense.keys()))
    cmap = plt.get_cmap("viridis", max(len(betas), 1))

    for i, beta_val in enumerate(betas):
        color = cmap(i)
        loose = sorted(results_loose.get(beta_val, []), key=lambda t: t[0])
        dense = sorted(results_dense.get(beta_val, []), key=lambda t: t[0])

        if loose:
            loose_ratios = [d[0] for d in loose]
            loose_n = [1 - d[1] for d in loose]
            ax.plot(
                loose_ratios,
                loose_n,
                marker="o",
                ls="-",
                color=color,
                label=rf"$\beta={beta_val:g}$ loose",
            )
        if dense:
            dense_ratios = [d[0] for d in dense]
            dense_n = [1 - d[1] for d in dense]
            ax.plot(
                dense_ratios,
                dense_n,
                marker="s",
                ls="--",
                color=color,
                label=rf"$\beta={beta_val:g}$ dense",
            )

    ax.set_xlabel(r"$\sigma/\mu$")
    ax.set_ylabel(r"Porosity, $n$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    os.makedirs("papers/percolation/figures", exist_ok=True)
    plt.savefig("papers/percolation/figures/gaussian_porosity.png", dpi=300)


if __name__ == "__main__":
    main()
