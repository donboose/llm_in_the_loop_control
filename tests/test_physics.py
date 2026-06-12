"""
Phase 1 smoke test.
Runs 32 isolated Drone2DEnv instances for 200 steps each
with random actions and checks for crashes, NaNs, and valid obs structure.
"""
import time
import math
import random
from sim.tasks.drone_2d import Drone2DEnv, MAX_FORCE

NUM_ENVS = 32
NUM_STEPS = 200


def random_action() -> dict:
    return {
        "force_x":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "force_y":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "torque_z": random.uniform(-MAX_FORCE, MAX_FORCE),
    }


def test_all_envs():
    print(f"\nSpawning {NUM_ENVS} environments...")
    envs = [Drone2DEnv() for _ in range(NUM_ENVS)]

    print("Checking initial observations...")
    for i, env in enumerate(envs):
        obs = env.get_obs()
        assert "drone_x" in obs, f"Env {i}: missing drone_x"
        assert len(obs["ray_distances"]) == 8, f"Env {i}: wrong ray count"
        assert not math.isnan(obs["dist_to_goal"]), f"Env {i}: NaN in dist_to_goal"
    print("  ✓ All initial observations valid")

    print(f"Running {NUM_STEPS} steps on all {NUM_ENVS} envs...")
    t0 = time.time()
    total_rewards = [0.0] * NUM_ENVS
    done_counts = [0] * NUM_ENVS

    for step in range(NUM_STEPS):
        for i, env in enumerate(envs):
            obs, reward, done = env.step(random_action())

            # Sanity checks every step
            assert not math.isnan(reward), f"Env {i} step {step}: NaN reward"
            assert not math.isnan(obs["drone_x"]), f"Env {i} step {step}: NaN pos"
            assert -6 < obs["drone_x"] < 6, f"Env {i} step {step}: drone escaped arena X={obs['drone_x']}"
            assert -6 < obs["drone_y"] < 6, f"Env {i} step {step}: drone escaped arena Y={obs['drone_y']}"

            total_rewards[i] += reward
            if done:
                done_counts[i] += 1
                env.reset()

    elapsed = time.time() - t0
    steps_per_sec = (NUM_STEPS * NUM_ENVS) / elapsed

    print(f"\n{'─'*50}")
    print(f"  ✓ {NUM_STEPS * NUM_ENVS:,} total steps completed")
    print(f"  ✓ {elapsed:.2f}s elapsed  →  {steps_per_sec:,.0f} steps/sec")
    print(f"  ✓ Mean cumulative reward: {sum(total_rewards)/NUM_ENVS:.2f}")
    print(f"  ✓ Total episode completions: {sum(done_counts)}")
    print(f"  ✓ No NaNs, no arena escapes, no crashes")
    print(f"{'─'*50}")

    for env in envs:
        env.close()
    print("\nAll environments closed cleanly. Phase 1 ✓")


if __name__ == "__main__":
    test_all_envs()
