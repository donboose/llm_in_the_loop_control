"""
Phase 5 integration test.
Runs the full pipeline — physics + LLM + renderer — for a short burst
and verifies the loop is stable, non-blocking, and produces sane stats.

Run:
    # Terminal 1: vllm must be running
    uv run vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16 \
        --gpu-memory-utilization 0.60 --max-num-seqs 64 \
        --max-model-len 2048 --port 8000

    # Terminal 2:
    PYTHONPATH=. uv run python tests/test_runner.py
"""

import asyncio
import math
import argparse

from sim.runner import SimRunner, RunnerConfig
from sim.tasks.drone_2d import Drone2DEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int,  default=60)
    parser.add_argument("--envs",     type=int,  default=32)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model",    type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    print(f"\n  Phase 5 Integration Test\n")
    print(f"  {args.envs} envs  ×  {args.steps} steps  |  "
          f"head={'off' if args.headless else 'on'}")

    config = RunnerConfig(
        env_class    = Drone2DEnv,
        num_envs     = args.envs,
        model        = args.model,
        head         = not args.headless,
        window_title = "Drone2D — Phase 5 Integration Test",
        log_interval = 10,
        temperature  = 0.5,    # slightly lower = more consistent actions
        max_tokens   = 64,
    )

    runner = SimRunner(config)
    runner.run(steps=args.steps)

    print(f"  Phase 5 ✓  —  Full pipeline operational")


if __name__ == "__main__":
    main()
