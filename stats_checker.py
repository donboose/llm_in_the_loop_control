import json
import statistics
import math

RUN_NUM = 15

rows = []
with open("checkpoints/drone_grpo/metrics.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))

n = len(rows)

rows.pop()

def m(rs, k):
    v = [r[k] for r in rs if k in r]
    return statistics.mean(v) if v else 0.0


def trend(vals):
    nv = len(vals)
    if nv < 2:
        return 0.0
    xm = (nv - 1) / 2
    ym = statistics.mean(vals)
    num = sum((i - xm) * (v - ym) for i, v in enumerate(vals))
    den = sum((i - xm) ** 2 for i in range(nv))
    return num / den if den else 0.0


print("=" * 80)
print(f"  RUN {RUN_NUM} ANALYSIS  |  {n} steps | epoch {rows[0]['epoch']:.4f}..{rows[-1]['epoch']:.4f}")
print(f"  step_time mean={m(rows, 'step_time'):.2f}s  (~{n * m(rows, 'step_time') / 3600:.1f}h total)")
print("=" * 80)

rewards   = [r["reward"]     for r in rows]
rew_std   = [r["reward_std"] for r in rows]
isr_mean  = [r["sampling/importance_sampling_ratio/mean"] for r in rows]
isr_min   = [r["sampling/importance_sampling_ratio/min"]  for r in rows]
kls       = [r["kl"]         for r in rows]
ent       = [r["entropy"]    for r in rows]
cl        = [r["completions/mean_length"] for r in rows]
fzs       = [r.get("frac_reward_zero_std", 0) for r in rows]
clip_lo   = [r.get("clip_ratio/low_mean", 0)  for r in rows]

print("\n-- 100-STEP WINDOWS " + "-" * 60)
print("step-rng   | reward     rew_std   ISR_mn   ISR_min  KL       entropy  frac_0std")
print("-" * 95)
for start in range(0, n, 100):
    chunk = rows[start:start + 100]
    if not chunk:
        break
    lo, hi = chunk[0]["step"], chunk[-1]["step"]
    print(
        f" {lo:4d}-{hi:4d} | {m(chunk, 'reward'):+.3f}   {m(chunk, 'reward_std'):7.3f}   "
        f"{m(chunk, 'sampling/importance_sampling_ratio/mean'):.4f}   "
        f"{m(chunk, 'sampling/importance_sampling_ratio/min'):.4f}   "
        f"{m(chunk, 'kl'):.5f}  {m(chunk, 'entropy'):.4f}   "
        f"{m(chunk, 'frac_reward_zero_std'):.3f}"
    )

print("\n-- REWARD DISTRIBUTION " + "-" * 57)
sr = sorted(rewards)
print(f"  min={min(rewards):+.2f}  max={max(rewards):+.2f}  mean={statistics.mean(rewards):+.3f}  stdev={statistics.stdev(rewards):.3f}")
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
    print(f"  p{int(q * 100):02d}: {sr[int(q * n)]:+.3f}")
print()
for thr, lbl in [
    (-10, ">-10"),
    (-5,  ">-5"),
    (0,   ">0"),
    (1,   ">1"),
    (3,   ">3"),
    (5,   ">5 (goal bonus)"),
    (6,   ">6 (goal+milestones)"),
]:
    c = sum(1 for r in rewards if r > thr)
    print(f"  reward {lbl}: {c}/{n} ({100 * c / n:.1f}%)")

print("\n-- REWARD_STD (key: is it wide enough for good advantages?) " + "-" * 20)
print(f"  overall mean={statistics.mean(rew_std):.3f}  first100={statistics.mean(rew_std[:100]):.3f}  last100={statistics.mean(rew_std[-100:]):.3f}")
print(f"  slope={trend(rew_std):+.5f}/step")

print("\n-- REWARD TRAJECTORY " + "-" * 59)
print(f"  overall slope:   {trend(rewards):+.6f}/step")
print(f"  last-200 slope:  {trend(rewards[-200:]):+.6f}/step  mean={statistics.mean(rewards[-200:]):+.3f}")
print(f"  last-100 slope:  {trend(rewards[-100:]):+.6f}/step  mean={statistics.mean(rewards[-100:]):+.3f}")
print(f"  first-50 mean:   {statistics.mean(rewards[:50]):+.3f}")
print(f"  last-50 mean:    {statistics.mean(rewards[-50:]):+.3f}")

print("\n-- ISR HEALTH " + "-" * 66)
print(f"  ISR_mean: overall={statistics.mean(isr_mean):.4f}  last100={statistics.mean(isr_mean[-100:]):.4f}")
print(f"  ISR_min=0.0: {sum(1 for v in isr_min if v == 0.0)}/{n} ({100 * sum(1 for v in isr_min if v == 0.0) / n:.1f}%)")
print(f"  ISR_min<0.1: {sum(1 for v in isr_min if v < 0.1)}/{n}")

print("\n-- KL " + "-" * 74)
print(f"  first50={statistics.mean(kls[:50]):.5f}  last50={statistics.mean(kls[-50:]):.5f}  max={max(kls):.5f}")

print("\n-- ENTROPY " + "-" * 69)
print(f"  first50={statistics.mean(ent[:50]):.4f}  last50={statistics.mean(ent[-50:]):.4f}  slope={trend(ent):+.6f}/step")

print("\n-- COMPLETION LENGTH " + "-" * 59)
cl_max = [r["completions/max_length"] for r in rows]
cl_min = [r["completions/min_length"] for r in rows]
print(f"  mean={statistics.mean(cl):.2f}  spread(max-min) mean={statistics.mean([cl_max[i] - cl_min[i] for i in range(n - 1)]):.2f}")



print("\n-- REWARD DECOMPOSITION ESTIMATE " + "-" * 47)
mean_r       = statistics.mean(rewards)
alive_total  = -0.005 * 120                   # -0.6 for full rollout
progress_est = mean_r - alive_total           # everything above the alive floor
goal_steps   = sum(1 for r in rewards if r > 5.0)
milestone_steps = sum(1 for r in rewards if 0.3 < r <= 5.0)
print(f"  Mean reward:              {mean_r:+.3f}")
print(f"  Alive penalty floor:      {alive_total:.3f}  (0.005 × 120 steps)")
print(f"  Progress + shaping est:   {progress_est:+.3f}  (mean_r minus alive floor)")
print(f"  Steps with goal bonus:    {goal_steps}/{n}  ({100*goal_steps/n:.1f}%)")
print(f"  Steps with milestones:    {milestone_steps}/{n}  ({100*milestone_steps/n:.1f}%)")
print(f"  Efficiency bonus range:   0.0 to +2.0  (fires when goal reached)")
print(f"  Terminal dist bonus range: 0.0 to +1.5 (fires when goal not reached)")



print("\n-- CROSS-RUN COMPARISON " + "-" * 57)
print(f"  Metric                    Run {RUN_NUM} (this, {n} st)")
print(f"  reward mean               {statistics.mean(rewards):+.3f}")
print(f"  reward first-50           {statistics.mean(rewards[:50]):+.3f}")
print(f"  reward last-50            {statistics.mean(rewards[-50:]):+.3f}")
print(f"  reward_std mean           {statistics.mean(rew_std):.3f}")
print(f"  max reward ever           {max(rewards):+.3f}")
print(f"  ISR_mean overall          {statistics.mean(isr_mean):.4f}")
print(f"  entropy last50            {statistics.mean(ent[-50:]):.4f}")
print(f"  KL last50                 {statistics.mean(kls[-50:]):.5f}")
print(f"  frac_zero_std last50      {statistics.mean(fzs[-50:]):.4f}")
print(f"  goal bonus fired (>3.5)   {sum(1 for r in rewards if r > 3.5)}")
print(f"  milestone 1.5m (>0.3)     {sum(1 for r in rewards if r > 0.3)}")


print(f"\n-- RUN {RUN_NUM} SHAPING DIAGNOSTICS " + "-" * 50)
# Efficiency bonus fires when goal is reached — reward will be > 10.0
# (goal +10, efficiency up to +2, milestones up to +3)
efficiency_fired = sum(1 for r in rewards if r > 10.0)
# Terminal distance bonus fires when goal not reached but drone got close
# Reward > 0 but <= 10 suggests progress + terminal bonus without goal
close_no_goal   = sum(1 for r in rewards if 0.5 < r <= 10.0)
print(f"  Efficiency bonus fired (reward > 10.0):  {efficiency_fired}/{n}  ({100*efficiency_fired/n:.1f}%)")
print(f"  Close-but-no-goal (0.5 < reward <= 10):  {close_no_goal}/{n}  ({100*close_no_goal/n:.1f}%)")
print(f"  Never moved toward goal (reward <= -0.5): {sum(1 for r in rewards if r <= -0.5)}/{n}")
