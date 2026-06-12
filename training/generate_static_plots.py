"""
Generate static training-metrics plots without requiring a display.

Usage (from project root):
    PYTHONPATH=. python training/generate_static_plots.py
    PYTHONPATH=. python training/generate_static_plots.py --metrics checkpoints/drone_grpo/metrics.jsonl
    PYTHONPATH=. python training/generate_static_plots.py --out results/

Always uses the Agg backend so it works inside Docker / headless CI.
"""

import argparse
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")          # MUST come before any other matplotlib import
import matplotlib.pyplot as plt
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="./checkpoints/drone_grpo/metrics.jsonl",
                   help="Path to metrics.jsonl produced by training")
    p.add_argument("--out", default="./results",
                   help="Directory to write PNGs into")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "reward" in r:      # skip the final training-summary row
                    rows.append(r)
            except json.JSONDecodeError:
                pass
    return rows


def col(rows: list[dict], key: str):
    steps, vals = [], []
    for r in rows:
        if key in r:
            try:
                steps.append(int(r["step"]))
                vals.append(float(r[key]))
            except (ValueError, TypeError):
                pass
    return np.array(steps, dtype=float), np.array(vals, dtype=float)


def rolling_mean(v: np.ndarray, w: int = 50) -> np.ndarray:
    if len(v) < w:
        return v
    return np.convolve(v, np.ones(w) / w, mode="valid")


# ── Main chart ────────────────────────────────────────────────────────────────

def make_training_chart(rows: list[dict], out_path: str) -> None:
    last_step  = rows[-1]["step"]
    last_epoch = rows[-1].get("epoch", "?")
    n          = len(rows)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(
        f"GRPO Training — Drone2D Navigator\n"
        f"Steps 1→{last_step}  |  Epoch {float(last_epoch):.2f}  |  {n} logged steps\n"
        f"Structured outputs · 120-step rollouts · Curriculum goals ≤ 3 m · β = 0.5",
        fontsize=10, fontweight="bold",
    )

    panels = [
        (0, 0, "reward",                                      "Reward mean ± std",            "#2196F3", False),
        (0, 1, "sampling/importance_sampling_ratio/mean",     "Importance Sampling Ratio",     "#FF9800", False),
        (0, 2, "kl",                                          "KL Divergence",                 "#E91E63", False),
        (1, 0, "grad_norm",                                   "Gradient Norm",                 "#F44336", True),
        (1, 1, "entropy",                                     "Policy Entropy",                "#9C27B0", False),
        (1, 2, "reward_std",                                  "Within-Batch Reward Std",       "#00BCD4", False),
        (2, 0, "loss",                                        "GRPO Loss",                     "#795548", False),
        (2, 1, "completions/mean_length",                     "Completion Length (tokens)",    "#4CAF50", False),
        (2, 2, "sampling/importance_sampling_ratio/min",      "ISR min (per-sample floor)",    "#FF5722", False),
    ]

    W = 50    # rolling-mean window

    for ri, ci, key, title, colour, logy in panels:
        ax = axes[ri][ci]
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=7)
        ax.grid(True, alpha=0.2, linestyle="--")
        if logy:
            ax.set_yscale("log")

        s, v = col(rows, key)
        if len(v) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue

        ax.plot(s, v, color=colour, lw=0.6, alpha=0.25)

        if len(v) >= W:
            ax.plot(s[W - 1:], rolling_mean(v, W), color=colour, lw=2.2,
                    label=f"{W}-step avg")

        ax.annotate(
            f"{v[-1]:.4g}",
            xy=(s[-1], v[-1]), xytext=(5, 0),
            textcoords="offset points", fontsize=7,
            color=colour, fontweight="bold",
        )

        # ── Panel-specific extras ──────────────────────────────────────────
        if key == "reward":
            _, v2 = col(rows, "reward_std")
            if len(v2) == len(v):
                ax.fill_between(s, v - v2, v + v2, alpha=0.07, color=colour)
            ax.axhline(0,  color="white", lw=0.8, ls="--", alpha=0.5, label="zero")
            ax.axhline(10, color="lime",  lw=0.8, ls="--", alpha=0.5, label="goal bonus ~10")

        if key == "sampling/importance_sampling_ratio/mean":
            ax.axhline(1.0, color="green",  lw=1.0, ls="--", alpha=0.8, label="ideal (1.0)")
            ax.axhline(0.7, color="orange", lw=0.9, ls="--", alpha=0.6, label="warn (0.7)")
            ax.set_ylim(0.4, 1.4)

        if key == "sampling/importance_sampling_ratio/min":
            ax.axhline(0.0, color="red", lw=0.8, ls="--", alpha=0.6, label="collapse")
            ax.set_ylim(-0.05, 1.1)

        if len(v) >= W:
            ax.legend(fontsize=7)

    fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved training chart → {out_path}")


def make_reward_progress_chart(rows: list[dict], out_path: str) -> None:
    """50-step bucket reward summary + distribution histogram."""
    s_all, r_all = col(rows, "reward")
    if len(r_all) == 0:
        print("[plot] No reward data — skipping reward-progress chart")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Reward Analysis", fontsize=12, fontweight="bold")

    # ── Left: 50-step bucket means ─────────────────────────────────────────
    bucket = 50
    b_steps, b_means, b_mins, b_maxs = [], [], [], []
    for i in range(0, len(r_all), bucket):
        chunk = r_all[i : i + bucket]
        b_steps.append(float(s_all[i]))
        b_means.append(float(np.mean(chunk)))
        b_mins.append(float(np.min(chunk)))
        b_maxs.append(float(np.max(chunk)))

    ax1.fill_between(b_steps, b_mins, b_maxs, alpha=0.2, color="#2196F3", label="min–max")
    ax1.plot(b_steps, b_means, color="#2196F3", lw=2.2, marker="o", ms=4, label="mean")
    ax1.axhline(0,  color="white",  lw=0.8, ls="--", alpha=0.5)
    ax1.axhline(10, color="lime",   lw=0.8, ls="--", alpha=0.5, label="goal bonus ~10")
    ax1.set_title("Reward per 50-step bucket", fontsize=10)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.legend(fontsize=8)

    # Key stats annotation
    pos_pct = 100 * np.mean(r_all > 0)
    goal_pct = 100 * np.mean(r_all > 5)
    ax1.text(0.02, 0.97,
             f"reward>0: {pos_pct:.1f}%\nreward>5 (goal): {goal_pct:.1f}%\nmean: {np.mean(r_all):+.2f}",
             transform=ax1.transAxes, fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    # ── Right: distribution histogram ──────────────────────────────────────
    ax2.hist(r_all, bins=60, color="#2196F3", alpha=0.75, edgecolor="none")
    ax2.axvline(0,  color="white", lw=1.0, ls="--", alpha=0.7, label="zero")
    ax2.axvline(10, color="lime",  lw=1.0, ls="--", alpha=0.7, label="goal bonus ~10")
    ax2.axvline(float(np.mean(r_all)), color="orange", lw=1.5, ls="-", label=f"mean {np.mean(r_all):+.2f}")
    ax2.set_title("Reward distribution", fontsize=10)
    ax2.set_xlabel("Reward")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.legend(fontsize=8)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved reward analysis → {out_path}")


# ── Stats summary (mirrors stats_checker.py output to a text file) ────────────

def write_stats_summary(rows: list[dict], out_path: str) -> None:
    r_all = [float(r["reward"]) for r in rows]
    isr   = [float(r["sampling/importance_sampling_ratio/mean"]) for r in rows if "sampling/importance_sampling_ratio/mean" in r]
    ent   = [float(r["entropy"]) for r in rows if "entropy" in r]
    kl    = [float(r["kl"]) for r in rows if "kl" in r]
    std   = [float(r["reward_std"]) for r in rows if "reward_std" in r]

    lines = [
        "=" * 70,
        f"  TRAINING SUMMARY  |  {len(rows)} steps logged",
        f"  Epoch range: {rows[0]['epoch']:.4f} → {rows[-1]['epoch']:.4f}",
        "=" * 70,
        "",
        "-- REWARD --",
        f"  mean:        {statistics.mean(r_all):+.4f}",
        f"  first-50:    {statistics.mean(r_all[:50]):+.4f}",
        f"  last-50:     {statistics.mean(r_all[-50:]):+.4f}",
        f"  max ever:    {max(r_all):+.4f}",
        f"  min ever:    {min(r_all):+.4f}",
        f"  reward > 0:  {sum(1 for v in r_all if v > 0)}/{len(r_all)} ({100*sum(1 for v in r_all if v > 0)/len(r_all):.1f}%)",
        f"  reward > 5 (goal): {sum(1 for v in r_all if v > 5)}/{len(r_all)} ({100*sum(1 for v in r_all if v > 5)/len(r_all):.1f}%)",
        "",
        "-- ISR HEALTH --",
        f"  ISR mean:     {statistics.mean(isr):.4f}" if isr else "  ISR: no data",
        f"  ISR last-50:  {statistics.mean(isr[-50:]):.4f}" if len(isr) >= 50 else "",
        "",
        "-- KL DIVERGENCE --",
        f"  first-50 mean: {statistics.mean(kl[:50]):.6f}" if len(kl) >= 50 else "  KL: insufficient data",
        f"  last-50 mean:  {statistics.mean(kl[-50:]):.6f}" if len(kl) >= 50 else "",
        "",
        "-- POLICY ENTROPY --",
        f"  first-50: {statistics.mean(ent[:50]):.4f}" if len(ent) >= 50 else "  Entropy: insufficient data",
        f"  last-50:  {statistics.mean(ent[-50:]):.4f}" if len(ent) >= 50 else "",
        "",
        "-- WITHIN-BATCH REWARD STD --",
        f"  mean: {statistics.mean(std):.3f}" if std else "  Std: no data",
        f"  (higher = more GRPO signal)",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[plot] Saved stats summary → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.metrics):
        print(f"[plot] ERROR: metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    rows = load_rows(args.metrics)
    if len(rows) == 0:
        print(f"[plot] ERROR: no reward rows found in {args.metrics}", file=sys.stderr)
        sys.exit(1)

    print(f"[plot] Loaded {len(rows)} training steps from {args.metrics}")

    make_training_chart(rows,
                        os.path.join(args.out, "training_dashboard.png"))
    make_reward_progress_chart(rows,
                               os.path.join(args.out, "reward_analysis.png"))
    write_stats_summary(rows,
                        os.path.join(args.out, "training_summary.txt"))

    print(f"\n[plot] All outputs written to: {args.out}/")


if __name__ == "__main__":
    main()
