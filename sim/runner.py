"""
SimRunner: the main loop that ties physics, LLM, and rendering together.
"""

import asyncio
import signal
import time
from dataclasses import dataclass
from typing import Type

from sim.env_base import BaseEnv
from sim.parallel_env import ParallelEnv
from sim.llm_client import LLMClient


@dataclass
class RunnerConfig:
    env_class:    Type[BaseEnv] = None
    num_envs:     int           = 32
    model:        str           = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url:     str           = "http://localhost:8000"
    temperature:  float         = 0.7
    max_tokens:   int           = 64
    head:         bool          = True
    window_title: str           = "Drone2D Sim"
    max_steps:    int           = 0
    target_fps:   float         = 0.0
    log_interval: int           = 20


class SimRunner:
    def __init__(self, config: RunnerConfig):
        self.cfg = config
        self._running = True

        self.penv = ParallelEnv(config.env_class, config.num_envs)

        self.llm = LLMClient(
            model       = config.model,
            base_url    = config.base_url,
            temperature = config.temperature,
            max_tokens  = config.max_tokens,
        )

        self.renderer = None
        if config.head:
            from rendering.renderer import Renderer
            self.renderer = Renderer(
                num_envs = config.num_envs,
                title    = config.window_title,
            )

        self._step          = 0
        self._t_start       = 0.0
        self._t_last_log    = 0.0
        self._reward_window = []

        signal.signal(signal.SIGINT, self._handle_sigint)

    def run(self, steps: int = 0):
        max_steps = steps or self.cfg.max_steps

        # Run health check AND the main loop inside ONE asyncio.run() call.
        # httpx.AsyncClient binds its transport to the event loop it first
        # uses. If we call asyncio.run() twice, the second loop sees a client
        # whose transport is tied to the now-closed first loop, causing
        # "RuntimeError: Event loop is closed" on the very first request.
        asyncio.run(self._startup_and_loop(max_steps))

        # Synchronous cleanup only — no more asyncio.run() after this point
        self._close_sync()

    async def _startup_and_loop(self, max_steps: int):
        """Health check + main loop share one event loop so httpx stays valid."""
        healthy = await self.llm.health_check()
        if not healthy:
            print("[SimRunner] vLLM server is not healthy. Aborting.")
            return

        print(f"\n[SimRunner] Starting loop — {self.cfg.num_envs} envs, "
              f"model={self.cfg.model}, head={self.cfg.head}")
        print(f"[SimRunner] Press Ctrl+C or close the window to stop.\n")

        obs_list = self.penv.reset_all()
        self._t_start    = time.monotonic()
        self._t_last_log = self._t_start

        await self._loop_and_close(obs_list, max_steps)

    async def _loop_and_close(self, obs_list: list[dict], max_steps: int):
        """
        Runs the main loop and closes the HTTP client at the end —
        all inside the same event loop so connections close cleanly.
        """
        try:
            await self._loop(obs_list, max_steps)
        finally:
            # Close HTTP client here, inside the loop that owns the connections
            await self.llm.close()

    async def _loop(self, obs_list: list[dict], max_steps: int):
        while self._running:
            if max_steps and self._step >= max_steps:
                print(f"\n[SimRunner] Reached {max_steps} steps — stopping.")
                break
            if self.renderer and self.renderer.should_close():
                print("\n[SimRunner] Window closed — stopping.")
                break

            t_step_start = time.monotonic()

            # 1. LLM: observe → act
            actions = await self.llm.query_batch(obs_list)

            # 2. Physics: act → step → new obs
            obs_list, rewards, dones, _ = self.penv.step_all(actions)

            # 3. Render
            if self.renderer:
                snapshot = self.penv.get_state_snapshot()
                self.renderer.draw(snapshot)

            # 4. Stats
            mean_r = sum(rewards) / len(rewards)
            self._reward_window.append(mean_r)
            if len(self._reward_window) > 50:
                self._reward_window.pop(0)

            self._step += 1
            self._maybe_log(t_step_start)

            # 5. Optional FPS cap
            if self.cfg.target_fps > 0:
                frame_budget = 1.0 / self.cfg.target_fps
                elapsed = time.monotonic() - t_step_start
                if elapsed < frame_budget:
                    await asyncio.sleep(frame_budget - elapsed)

    def _close_sync(self):
        """
        Synchronous-only cleanup. Never calls asyncio.run() — by the time
        this runs, the event loop is already closed.
        """
        print("\n[SimRunner] Shutting down...")
        if self.renderer:
            self.renderer.destroy()
        self.penv.close()
        print("[SimRunner] Done.")

    def _maybe_log(self, t_step_start: float):
        if self._step % self.cfg.log_interval != 0:
            return

        now       = time.monotonic()
        elapsed   = now - self._t_start
        sps       = self._step / elapsed
        step_ms   = (now - t_step_start) * 1000
        llm_ms    = self.llm.last_latency_ms
        rolling_r = sum(self._reward_window) / max(len(self._reward_window), 1)
        completed = self.penv.pop_completed_episodes()
        goal_rate = (
            sum(1 for e in completed if e["goal_reached"]) / len(completed)
            if completed else 0.0
        )

        print(
            f"  step={self._step:>6}  "
            f"sps={sps:>6.1f}  "
            f"llm={llm_ms:>5.0f}ms  "
            f"step={step_ms:>5.0f}ms  "
            f"reward={rolling_r:>+6.3f}  "
            f"goal_rate={goal_rate:.0%}  "
            f"eps_done={len(completed)}"
        )

    def _handle_sigint(self, sig, frame):
        print("\n[SimRunner] SIGINT received — stopping after current step...")
        self._running = False
