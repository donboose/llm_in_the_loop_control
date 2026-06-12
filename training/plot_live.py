"""
Live training dashboard.  Run in a separate terminal while training is running:

    PYTHONPATH=. uv run python training/plot_live.py

The plot auto-refreshes every 15 seconds.  Close the window to stop.

Six panels:
  [0,0] Reward mean ± std          — primary learning signal
  [0,1] Gradient norm (log)        — confirms weights are actually updating
  [1,0] Completion length          — should stabilise around 29 tokens
  [1,1] Importance Sampling Ratio  — must stay near 1.0; collapse → policy drift
  [2,0] Policy entropy             — decreases as policy becomes more confident
  [2,1] Loss                       — GRPO surrogate loss
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np

plt.style.use('dark_background')

METRICS_FILE = "./checkpoints/drone_grpo/metrics.jsonl"
REFRESH_MS   = 15_000   # poll interval in milliseconds


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rows() -> list[dict]:
    if not os.path.exists(METRICS_FILE):
        return []
    rows = []
    with open(METRICS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def col(rows: list[dict], key: str) -> tuple[list, list]:
    """Return (steps, values) for rows that contain key."""
    steps, vals = [], []
    for r in rows:
        if key in r:
            try:
                steps.append(r["step"])
                vals.append(float(r[key]))
            except (ValueError, TypeError):
                pass
    return steps, vals


# ── Plot setup ────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(13, 10))
fig.suptitle("GRPO Training — Drone2D Navigator", fontsize=14, fontweight="bold")
fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])

PANEL_CFG = [
    # (row, col, key,                                         title,                          colour,   logy)
    (0, 0, "reward",                                          "Reward (mean ± std)",          "#2196F3", False),
    (0, 1, "grad_norm",                                       "Gradient Norm",                "#F44336", True ),
    (1, 0, "completions/mean_length",                         "Completion Length",            "#4CAF50", False),
    (1, 1, "sampling/importance_sampling_ratio/mean",         "Importance Sampling Ratio",    "#FF9800", False),
    (2, 0, "entropy",                                         "Policy Entropy",               "#9C27B0", False),
    (2, 1, "loss",                                            "GRPO Loss",                    "#795548", False),
]


def draw(frame):
    rows = load_rows()
    n    = len(rows)

    for r, c, key, title, colour, logy in PANEL_CFG:
        ax = axes[r][c]
        ax.cla()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--")
        if logy:
            ax.set_yscale("log")

        steps, vals = col(rows, key)
        if not vals:
            ax.text(0.5, 0.5, "waiting for data…", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue

        ax.plot(steps, vals, color=colour, linewidth=1.4, label=key.split("/")[-1])

        # Reward: shade ±std band
        if key == "reward":
            s_std, v_std = col(rows, "reward_std")
            if v_std and len(v_std) == len(vals):
                lo = [m - s for m, s in zip(vals, v_std)]
                hi = [m + s for m, s in zip(vals, v_std)]
                ax.fill_between(steps, lo, hi, alpha=0.18, color=colour)

        # Completion length: draw 48-token ceiling
        if key == "completions/mean_length":
            ax.axhline(48, color="red", linewidth=0.8, linestyle="--",
                       alpha=0.6, label="max (48)")
            ax.legend(fontsize=7)

        # ISR: healthy zone and collapse warning
        if key == "sampling/importance_sampling_ratio/mean":
            ax.axhline(1.0, color="green",  linewidth=0.9, linestyle="--", alpha=0.7, label="ideal (1.0)")
            ax.axhline(0.5, color="orange", linewidth=0.9, linestyle="--", alpha=0.7, label="warn (0.5)")
            ax.axhline(0.2, color="red",    linewidth=0.9, linestyle="--", alpha=0.7, label="collapse (0.2)")
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7)

        # Running mean overlay (window = 10 steps)
        if len(vals) >= 10:
            rm = np.convolve(vals, np.ones(10) / 10, mode="valid")
            rm_steps = steps[9:]
            ax.plot(rm_steps, rm, color="white", linewidth=2.0, alpha=0.5)
            ax.plot(rm_steps, rm, color=colour,  linewidth=1.2, alpha=0.8,
                    linestyle="--", label="10-step avg")
            ax.legend(fontsize=7)

        # Latest value annotation
        ax.annotate(
            f"{vals[-1]:.4g}",
            xy=(steps[-1], vals[-1]),
            xytext=(5, 0), textcoords="offset points",
            fontsize=7, color=colour,
        )

    # Global step counter in title
    fig.suptitle(
        f"GRPO Training — Drone2D Navigator  "
        f"[{n} steps logged]",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])

if __name__ == "__main__":
    if "--save" in sys.argv:
        print("[plot] Saving plot to final_plot.png...")
        draw(0)
        fig.savefig("final_plot.png", dpi=150)
        print("[plot] Done.")
    else:
        ani = animation.FuncAnimation(fig, draw, interval=REFRESH_MS, cache_frame_data=False)
        plt.show()
