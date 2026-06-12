"""
ParallelEnv: orchestrates N independent BaseEnv instances.

Key design rules:
  - Each env has its own pybullet DIRECT client (fully isolated memory).
  - Step loop is synchronous here — async LLM batching happens in Phase 4.
  - The state snapshot is the ONLY thing the renderer ever reads.
    Physics and rendering never share a reference, only this dict.
  - Auto-reset: any done env is immediately reset and its obs replaced,
    so all N envs are always running. This matches how GRPO expects
    a continuous stream of (obs, action, reward) tuples.
"""

import time
from typing import Type

from sim.env_base import BaseEnv


class ParallelEnv:
    def __init__(self, env_class: Type[BaseEnv], num_envs: int, env_kwargs: dict = None):
        """
        Args:
            env_class:  A concrete BaseEnv subclass (e.g. Drone2DEnv).
            num_envs:   How many parallel instances to run (typically 32–128).
            env_kwargs: Keyword arguments forwarded to each env constructor.
        """
        env_kwargs = env_kwargs or {}
        self.num_envs = num_envs
        self.env_class = env_class

        print(f"[ParallelEnv] Spawning {num_envs} × {env_class.__name__}...")
        t0 = time.time()
        self.envs: list[BaseEnv] = [env_class(**env_kwargs) for _ in range(num_envs)]
        print(f"[ParallelEnv] Ready in {time.time() - t0:.2f}s")

        # Live observation cache — always holds the latest obs for each env.
        # Populated by reset_all() and kept current by step_all().
        self._obs: list[dict] = [None] * num_envs

        # Per-env episode stats for logging
        self._episode_rewards: list[float] = [0.0] * num_envs
        self._episode_lengths: list[int]   = [0]   * num_envs
        self._completed_episodes: list[dict] = []   # finished episode summaries

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset_all(self) -> list[dict]:
        """
        Reset every environment and return their initial observations.
        Call this once at the start of training before the first step_all().
        """
        for i, env in enumerate(self.envs):
            self._obs[i] = env.reset()
            self._episode_rewards[i] = 0.0
            self._episode_lengths[i] = 0
        return list(self._obs)

    def step_all(
        self, actions: list[dict]
    ) -> tuple[list[dict], list[float], list[bool], list[dict]]:
        """
        Step every environment with its corresponding action.

        Args:
            actions: list of N action dicts, one per env.
                     Missing or malformed actions default to zero-force.

        Returns:
            obs_list   — new observations after the step
            reward_list — scalar rewards
            done_list  — True if the episode ended (goal reached or timeout)
            info_list  — metadata dict per env (episode stats on done, else {})

        Auto-reset behaviour:
            When done=True, the env is immediately reset.
            The returned obs for that env is the FIRST obs of the NEW episode,
            not the terminal obs. This matches standard vectorised env APIs
            (e.g. stable-baselines3 VecEnv, TRL's rollout collector).
        """
        assert len(actions) == self.num_envs, (
            f"Expected {self.num_envs} actions, got {len(actions)}"
        )

        obs_list    = [None] * self.num_envs
        reward_list = [0.0]  * self.num_envs
        done_list   = [False] * self.num_envs
        info_list   = [{}]   * self.num_envs

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Defensive: fall back to zero action if something is malformed
            if not isinstance(action, dict):
                action = {"force_x": 0.0, "force_y": 0.0, "torque_z": 0.0}

            obs, reward, done = env.step(action)

            self._episode_rewards[i] += reward
            self._episode_lengths[i] += 1

            if done:
                # Record completed episode before resetting
                summary = {
                    "env_idx":        i,
                    "total_reward":   self._episode_rewards[i],
                    "episode_length": self._episode_lengths[i],
                    "goal_reached":   reward > 50.0,  # heuristic: +100 bonus
                }
                self._completed_episodes.append(summary)
                info_list[i] = summary

                # Immediately reset — obs_list[i] is now the NEW episode's first obs
                obs = env.reset()
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0

            self._obs[i]    = obs
            obs_list[i]     = obs
            reward_list[i]  = reward
            done_list[i]    = done

        return obs_list, reward_list, done_list, info_list

    def set_state_all(self, obs_list: list) -> None:
        """
        Teleport each environment to the state described in obs_list[i].

        Envs whose corresponding entry is None are left untouched — their
        current physics state carries forward into the next step_all() call.

        Args:
            obs_list: Sequence of length ≤ num_envs.  Each element is either
                      an obs dict (as returned by Drone2DEnv.get_obs()) or
                      None.  Extra envs beyond len(obs_list) are also skipped.
        """
        for i, obs in enumerate(obs_list):
            if i >= self.num_envs:
                break
            if obs is None:
                continue
            env = self.envs[i]
            if hasattr(env, "set_state"):
                env.set_state(obs)

    def get_obs(self) -> list[dict]:
        """Return the cached latest observation for each env (no physics call)."""
        return list(self._obs)

    def get_state_snapshot(self) -> list[dict]:
        """
        Return a lightweight copy of all env states for the renderer.

        This is the ONLY channel between physics and rendering.
        The renderer reads this; it never touches self.envs directly.

        Each entry contains just what the renderer needs to draw one env:
          - drone position and heading
          - goal position
          - obstacle positions (pulled from the env directly)
          - world size (so renderer can scale viewports)
        """
        snapshot = []
        for i, env in enumerate(self.envs):
            obs = self._obs[i]
            if obs is None:
                snapshot.append(None)
                continue

            # Pull obstacle geometry directly from the env for rendering.
            # This is read-only — renderer never modifies these.
            entry = {
                "env_idx":   i,
                "drone_x":   obs["drone_x"],
                "drone_y":   obs["drone_y"],
                "drone_yaw": obs["drone_yaw"],
                "goal_x":    obs["goal_x"],
                "goal_y":    obs["goal_y"],
                "dist_to_goal": obs["dist_to_goal"],
                "step":      obs["step"],
                # Obstacle and wall data for rendering (positions + half-extents)
                "obstacles": _get_obstacle_data(env),
                "world_size": 10.0,
            }
            snapshot.append(entry)
        return snapshot

    # ── Stats & Logging ───────────────────────────────────────────────────────

    def pop_completed_episodes(self) -> list[dict]:
        """
        Return and clear all episodes that finished since the last call.
        Use this in your training loop to log episode stats.
        """
        episodes = self._completed_episodes
        self._completed_episodes = []
        return episodes

    def mean_reward(self) -> float:
        """Mean in-progress episode reward across all envs."""
        return sum(self._episode_rewards) / self.num_envs

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        """Disconnect all physics clients cleanly."""
        for env in self.envs:
            env.close()
        print(f"[ParallelEnv] All {self.num_envs} environments closed.")


# ── Helper (module-level, not a method) ───────────────────────────────────────

def _get_obstacle_data(env) -> list[dict]:
    """
    Extract obstacle position + size from a Drone2DEnv for the renderer.
    Returns a list of dicts: {x, y, half_w, half_h}

    We import pybullet here (not at the top of parallel_env.py) to keep
    this module free of physics imports — only the helper touches PyBullet.
    """
    import pybullet as p

    obstacles = []
    for obs_id in getattr(env, "_obstacle_ids", []):
        try:
            pos, _ = p.getBasePositionAndOrientation(
                obs_id, physicsClientId=env.client
            )
            # getCollisionShapeData returns shape info including half-extents
            shape_data = p.getCollisionShapeData(
                obs_id, -1, physicsClientId=env.client
            )
            if shape_data:
                half_extents = shape_data[0][3]   # (half_w, half_d, half_h)
                obstacles.append({
                    "x":      round(float(pos[0]), 3),
                    "y":      round(float(pos[1]), 3),
                    "half_w": round(float(half_extents[0]), 3),
                    "half_h": round(float(half_extents[1]), 3),
                })
        except Exception:
            pass
    return obstacles
