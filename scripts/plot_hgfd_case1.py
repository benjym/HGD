#!/usr/bin/env python3
"""Plot the three HGFD paper Case 1 settling curves from saved runs."""

from pathlib import Path

import json5
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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


def style_axes(ax):
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 0.285)
    ax.set_xticks(np.arange(0, 401, 100))
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Particle velocity [m/s]")
    ax.legend(title="Particle diameter", frameon=True, loc="lower right")
    ax.tick_params(direction="in", width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def save_static(curves):
    fig, ax = plt.subplots(figsize=(6.2, 6.8), constrained_layout=True)
    for (_, label, color, linestyle, _), (time_ms, speed) in zip(CASES, curves):
        ax.plot(time_ms, speed, color=color, linestyle=linestyle, linewidth=3, label=label)
    style_axes(ax)
    output = ROOT / "docs" / "images" / "hgfd_case1_reproduction.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output.relative_to(ROOT)}")


def save_animation(curves):
    fig, ax = plt.subplots(figsize=(6.2, 6.8), constrained_layout=True)
    lines = []
    for _, label, color, linestyle, _ in CASES:
        (line,) = ax.plot([], [], color=color, linestyle=linestyle, linewidth=3, label=label)
        lines.append(line)
    style_axes(ax)

    last_frame = min(len(time_ms) for time_ms, _ in curves) - 1

    def update(frame):
        for line, (time_ms, speed) in zip(lines, curves):
            line.set_data(time_ms[: frame + 1], speed[: frame + 1])
        return lines

    # Hold the converged curves for one second at the end of the loop.
    frames = list(range(last_frame + 1)) + [last_frame] * 10
    animation = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    output = ROOT / "docs" / "images" / "hgfd_case1_reproduction.gif"
    animation.save(output, writer=PillowWriter(fps=10), dpi=120)
    plt.close(fig)
    print(f"Wrote {output.relative_to(ROOT)}")


def main():
    curves = [load_curve(case_name) for case_name, *_ in CASES]
    save_static(curves)
    save_animation(curves)

    for (_, label, _, _, paper), (_, speed) in zip(CASES, curves):
        reproduced = speed[-1]
        relative_error = 100.0 * (reproduced - paper) / paper
        print(
            f"{label}: reproduced={reproduced:.8f} m/s, paper={paper:.8f} m/s, error={relative_error:+.3f}%"
        )


if __name__ == "__main__":
    main()
