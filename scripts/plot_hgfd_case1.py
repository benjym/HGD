#!/usr/bin/env python3
"""Plot the three HGFD paper Case 1 settling curves from saved runs."""

from pathlib import Path

import json5
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("0p5mm", "0.5 mm", "black", "-", 0.07425),
    ("1p5mm", "1.5 mm", "#1f77b4", ":", 0.21144),
    ("2p0mm", "2.0 mm", "#d62728", "--", 0.26153),
)


def load_curve(case_name):
    config_path = ROOT / "json" / f"hgfd_paper_case1_{case_name}.json5"
    config = json5.loads(config_path.read_text())
    files = sorted((ROOT / "output" / config_path.stem / "data").glob("v_*.npy"))
    if not files:
        raise FileNotFoundError(f"No saved velocities for {config_path.stem}; run the case first")

    # save_velocity stores the mean over M.  Case 1 contains exactly one
    # occupied coordinate, so multiplying by M recovers its layer velocity.
    steps = np.array([int(path.stem.rsplit("_", 1)[1]) for path in files])
    speed = np.array([config["nm"] * np.max(np.abs(np.load(path))) for path in files])
    return 1e3 * steps * config["defined_time_step_size"], speed


def main():
    fig, ax = plt.subplots(figsize=(6.2, 6.8), constrained_layout=True)
    rows = []
    for case_name, label, color, linestyle, paper_terminal in CASES:
        time_ms, speed = load_curve(case_name)
        ax.plot(time_ms, speed, color=color, linestyle=linestyle, linewidth=3, label=label)
        rows.append((label, speed[-1], paper_terminal))

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 0.285)
    ax.set_xticks(np.arange(0, 401, 100))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Particle velocity [m/s]")
    ax.legend(title="Particle diameter", frameon=True, loc="lower right")
    ax.tick_params(direction="in", width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    output = ROOT / "docs" / "images" / "hgfd_case1_reproduction.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)

    print(f"Wrote {output.relative_to(ROOT)}")
    for label, reproduced, paper in rows:
        relative_error = 100.0 * (reproduced - paper) / paper
        print(
            f"{label}: reproduced={reproduced:.8f} m/s, paper={paper:.8f} m/s, error={relative_error:+.3f}%"
        )


if __name__ == "__main__":
    main()
