"""
Phase 2 test: ParallelEnv correctness and performance.
Checks auto-reset, snapshot structure, and steps/sec throughput.
"""

import time
import random
import math

from sim.parallel_env import ParallelEnv
from sim.tasks.drone_2d import Drone2DEnv, MAX_FORCE

NUM_ENVS  = 32
NUM_STEPS = 300


def random_action():
    return {
        "force_x":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "force_y":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "torque_z": random.uniform(-MAX_FORCE, MAX_FORCE),
    }


def test_parallel_env():
    penv = ParallelEnv(Drone2DEnv, NUM_ENVS)

    # ── 1. reset_all ─────────────────────────────────────────────
    obs_list = penv.reset_all()
    assert len(obs_list) == NUM_ENVS, "reset_all must return N obs"
    for i, obs in enumerate(obs_list):
        assert "drone_x" in obs,       f"Env {i}: missing drone_x in obs"
        assert "goal_x"  in obs,       f"Env {i}: missing goal_x in obs"
        assert len(obs["ray_distances"]) == 8, f"Env {i}: wrong ray count"
    print(f"  ✓ reset_all() returned {NUM_ENVS} valid observations")

    # ── 2. step_all + auto-reset ──────────────────────────────────
    t0 = time.time()
    total_done = 0

    for step in range(NUM_STEPS):
        actions = [random_action() for _ in range(NUM_ENVS)]
        obs_list, rewards, dones, infos = penv.step_all(actions)

        assert len(obs_list) == NUM_ENVS
        assert len(rewards)  == NUM_ENVS
        assert len(dones)    == NUM_ENVS

        for i, (obs, r, done) in enumerate(zip(obs_list, rewards, dones)):
            assert not math.isnan(r),            f"Env {i} step {step}: NaN reward"
            assert not math.isnan(obs["drone_x"]), f"Env {i} step {step}: NaN pos"
            # After auto-reset, step counter should be 0 or 1 (fresh episode)
            if done:
                total_done += 1
                assert obs["step"] <= 2, (
                    f"Env {i}: post-reset obs has step={obs['step']}, expected ~0"
                )

    elapsed = time.time() - t0
    sps = (NUM_STEPS * NUM_ENVS) / elapsed
    print(f"  ✓ {NUM_STEPS * NUM_ENVS:,} steps in {elapsed:.2f}s  →  {sps:,.0f} steps/sec")
    print(f"  ✓ {total_done} auto-resets triggered")

    # ── 3. get_state_snapshot ─────────────────────────────────────
    snapshot = penv.get_state_snapshot()
    assert len(snapshot) == NUM_ENVS, "Snapshot must have N entries"
    for i, entry in enumerate(snapshot):
        assert entry is not None,           f"Env {i}: None snapshot"
        assert "drone_x"  in entry,         f"Env {i}: missing drone_x in snapshot"
        assert "obstacles" in entry,         f"Env {i}: missing obstacles in snapshot"
        assert "world_size" in entry,        f"Env {i}: missing world_size"
        assert isinstance(entry["obstacles"], list), f"Env {i}: obstacles not a list"
    print(f"  ✓ get_state_snapshot() returned {NUM_ENVS} valid entries")
    print(f"  ✓ Sample snapshot[0]: drone=({snapshot[0]['drone_x']:.2f}, {snapshot[0]['drone_y']:.2f})  goal=({snapshot[0]['goal_x']:.2f}, {snapshot[0]['goal_y']:.2f})  obstacles={len(snapshot[0]['obstacles'])}")

    # ── 4. pop_completed_episodes ─────────────────────────────────
    completed = penv.pop_completed_episodes()
    assert isinstance(completed, list)
    print(f"  ✓ pop_completed_episodes(): {len(completed)} episodes logged")
    if completed:
        ep = completed[0]
        assert "total_reward"   in ep
        assert "episode_length" in ep
        assert "goal_reached"   in ep
        print(f"    Sample: reward={ep['total_reward']:.2f}  len={ep['episode_length']}  goal={ep['goal_reached']}")

    # ── 5. action mismatch guard ──────────────────────────────────
    try:
        penv.step_all([random_action()] * (NUM_ENVS - 1))   # wrong count
        assert False, "Should have raised AssertionError"
    except AssertionError:
        print(f"  ✓ step_all() correctly rejects wrong-length action list")

    # ── 6. close ─────────────────────────────────────────────────
    penv.close()
    print(f"\n{'─'*52}")
    print(f"  Phase 2 ✓  —  ParallelEnv fully operational")
    print(f"{'─'*52}\n")


if __name__ == "__main__":
    print(f"\nRunning Phase 2: ParallelEnv ({NUM_ENVS} envs × {NUM_STEPS} steps)\n")
    test_parallel_env()
