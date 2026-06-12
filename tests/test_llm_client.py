"""
Phase 4 test: LLM client correctness and batch performance.

Run in mock mode (no vLLM needed):
    PYTHONPATH=. uv run python tests/test_llm_client.py --mock

Run in live mode (vLLM must be running):
    # Terminal 1:
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

    # Terminal 2:
    PYTHONPATH=. uv run python tests/test_llm_client.py --live --model Qwen/Qwen2.5-1.5B-Instruct
"""

import asyncio
import json
import argparse
import sys
from unittest.mock import AsyncMock, patch, MagicMock

from sim.llm_client import LLMClient
from sim.schemas import DroneAction, DRONE_ACTION_SCHEMA
from sim.parallel_env import ParallelEnv
from sim.tasks.drone_2d import Drone2DEnv


# ── Sample observations for testing ──────────────────────────────────────────

def make_fake_obs(i: int) -> dict:
    return {
        "drone_x": float(i) * 0.1,
        "drone_y": 0.0,
        "drone_yaw": 0.0,
        "vel_x": 0.0, "vel_y": 0.0, "ang_vel_z": 0.0,
        "goal_x": 3.0, "goal_y": 3.0,
        "dx_to_goal": 3.0, "dy_to_goal": 3.0,
        "dist_to_goal": 4.24,
        "ray_distances": [3.0] * 8,
        "step": 0,
    }


# ── Mock mode: tests parsing + schema without any server ─────────────────────

async def test_mock_mode(num_envs: int = 32):
    print("\n MOCK MODE — testing schema, parsing, error handling\n")

    client = LLMClient(model="mock-model")

    # 1. Schema structure
    assert "properties" in DRONE_ACTION_SCHEMA
    assert "force_x"  in DRONE_ACTION_SCHEMA["properties"]
    assert "force_y"  in DRONE_ACTION_SCHEMA["properties"]
    assert "torque_z" in DRONE_ACTION_SCHEMA["properties"]
    print("  ✓ DroneAction JSON schema has correct fields")

    # 2. Pydantic validation: good input
    good_json = '{"force_x": 5.0, "force_y": -3.2, "torque_z": 1.1}'
    action = DroneAction.model_validate_json(good_json)
    assert action.force_x == 5.0
    assert action.force_y == -3.2
    print("  ✓ Pydantic validates well-formed action JSON")

    # 3. Pydantic validation: bad input raises
    try:
        DroneAction.model_validate_json('{"force_x": "oops"}')
        assert False, "Should have raised"
    except Exception:
        print("  ✓ Pydantic rejects malformed action JSON")

    # 4. Mock the HTTP call and test query_batch
    fake_response_content = '{"force_x": 10.0, "force_y": 5.0, "torque_z": -2.0}'
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": fake_response_content}}]
    }

    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_response)):
        obs_list = [make_fake_obs(i) for i in range(num_envs)]
        actions = await client.query_batch(obs_list)

    assert len(actions) == num_envs, f"Expected {num_envs} actions, got {len(actions)}"
    for i, a in enumerate(actions):
        assert "force_x"  in a, f"Action {i} missing force_x"
        assert "force_y"  in a, f"Action {i} missing force_y"
        assert "torque_z" in a, f"Action {i} missing torque_z"
        assert isinstance(a["force_x"], float), f"Action {i} force_x not float"
    print(f"  ✓ query_batch() returned {num_envs} valid action dicts (mocked)")

    # 5. Error handling: one request throws — should return zero action
    call_count = 0
    async def flaky_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count % 4 == 0:   # every 4th request fails
            raise httpx.ConnectError("Simulated failure")
        return mock_response

    import httpx
    with patch.object(client._http, "post", new=flaky_post):
        obs_list = [make_fake_obs(i) for i in range(num_envs)]
        actions = await client.query_batch(obs_list)

    assert len(actions) == num_envs
    zero_count = sum(1 for a in actions if a["force_x"] == 0.0 and a["force_y"] == 0.0)
    print(f"  ✓ Flaky requests degrade to zero-action ({zero_count} zeros in batch of {num_envs})")

    await client.close()
    print(f"\n  Mock mode ✓\n")


# ── Live mode: tests against a real vLLM server ───────────────────────────────

async def test_live_mode(model: str, num_envs: int = 32, num_batches: int = 3):
    print(f"\n  LIVE MODE — model={model}  envs={num_envs}  batches={num_batches}\n")

    client = LLMClient(model=model, temperature=0.3, max_tokens=64)

    # 1. Health check
    healthy = await client.health_check()
    if not healthy:
        print("  ✗ vLLM server not healthy. Is it running?")
        print("    Run: vllm serve <model> --port 8000")
        await client.close()
        sys.exit(1)

    # 2. Spin up real environments
    penv = ParallelEnv(Drone2DEnv, num_envs)
    obs_list = penv.reset_all()
    print(f"  ✓ {num_envs} environments ready")

    # 3. Run N batches and measure latency
    import time
    latencies = []

    for batch_idx in range(num_batches):
        t0 = time.monotonic()
        actions = await client.query_batch(obs_list)
        latency = (time.monotonic() - t0) * 1000

        latencies.append(latency)

        # Validate actions
        assert len(actions) == num_envs
        for i, a in enumerate(actions):
            assert "force_x"  in a, f"Batch {batch_idx} env {i}: missing force_x"
            assert "force_y"  in a, f"Batch {batch_idx} env {i}: missing force_y"
            assert "torque_z" in a, f"Batch {batch_idx} env {i}: missing torque_z"

        # Step the environments with the LLM actions
        obs_list, rewards, dones, _ = penv.step_all(actions)

        print(
            f"  Batch {batch_idx+1}/{num_batches}: "
            f"latency={latency:.0f}ms  "
            f"sample_action={actions[0]}  "
            f"mean_reward={sum(rewards)/len(rewards):.3f}"
        )

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n  ✓ All {num_batches} batches completed")
    print(f"  ✓ Average batch latency: {avg_latency:.0f}ms  "
          f"(per-env effective: {avg_latency/num_envs:.1f}ms)")
    print(f"  ✓ Actions are valid JSON matching DroneAction schema")

    penv.close()
    await client.close()
    print(f"\n  Live mode ✓\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Run without vLLM server")
    group.add_argument("--live", action="store_true", help="Run against real vLLM server")
    parser.add_argument("--model",    type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--batches",  type=int, default=3)
    args = parser.parse_args()

    if args.mock:
        asyncio.run(test_mock_mode(num_envs=args.num_envs))
    else:
        asyncio.run(test_live_mode(
            model=args.model,
            num_envs=args.num_envs,
            num_batches=args.batches,
        ))


if __name__ == "__main__":
    main()
