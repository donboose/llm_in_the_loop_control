"""
Phase 3 visual test.
Opens the 4×8 grid window and runs 32 environments with random actions
for 5 seconds. You should see:
  - Dark background with 32 cells in a 8×4 grid
  - Grey walls around each arena
  - Brown rectangular obstacles (5 per env, randomly placed)
  - A green circle = goal
  - A blue circle = drone with a yellow heading line
  - The drone bouncing around randomly
"""

import time
import random
from sim.parallel_env import ParallelEnv
from sim.tasks.drone_2d import Drone2DEnv, MAX_FORCE
from rendering.renderer import Renderer

NUM_ENVS    = 32
RUN_SECONDS = 10


def random_action():
    return {
        "force_x":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "force_y":  random.uniform(-MAX_FORCE, MAX_FORCE),
        "torque_z": random.uniform(-MAX_FORCE, MAX_FORCE),
    }


def main():
    print(f"[test_renderer] Starting {NUM_ENVS} envs + renderer for {RUN_SECONDS}s")
    print("[test_renderer] Close the window or wait to exit\n")

    penv     = ParallelEnv(Drone2DEnv, NUM_ENVS)
    renderer = Renderer(num_envs=NUM_ENVS, title="Drone2D — Phase 3 Visual Test")

    penv.reset_all()

    step     = 0
    t_start  = time.time()
    t_last   = t_start

    while True:
        elapsed = time.time() - t_start
        if elapsed >= RUN_SECONDS:
            print(f"[test_renderer] {RUN_SECONDS}s elapsed — exiting")
            break
        if renderer.should_close():
            print("[test_renderer] Window closed — exiting")
            break

        actions = [random_action() for _ in range(NUM_ENVS)]
        penv.step_all(actions)
        snapshot = penv.get_state_snapshot()
        renderer.draw(snapshot)
        step += 1

        # Print stats every second
        now = time.time()
        if now - t_last >= 1.0:
            sps = step / (now - t_start)
            print(
                f"  t={elapsed:.1f}s  step={step:,}  "
                f"~{sps:.0f} steps/sec  "
                f"mean_reward={penv.mean_reward():.2f}"
            )
            t_last = now

    renderer.destroy()
    penv.close()
    print("\n[test_renderer] Phase 3 ✓")


if __name__ == "__main__":
    main()
