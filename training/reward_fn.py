"""
Reward function bridge between TRL's GRPOTrainer and our PyBullet simulator.

How GRPOTrainer calls reward_funcs:
  reward_fn(prompts, completions, **kwargs) -> list[float]

  - prompts:     list of N prompt strings (the JSON observation we sent)
  - completions: list of N completion strings (what the model generated)
  - **kwargs:    any extra columns from the dataset row (we pass env_idx here)
  - returns:     list of N scalar floats

The function must be SYNCHRONOUS — TRL does not support async reward funcs
in server mode. We run a dedicated ParallelEnv just for reward computation,
separate from any rollout envs.

IMPORTANT: This reward env is NOT the same as the runner's env.
TRL manages its own generation loop; it gives us completions and we
return rewards. We maintain a persistent ParallelEnv that stays alive
for the entire training run.
"""

import json
import os
import re
import time
from typing import Optional

from sim.parallel_env import ParallelEnv
from sim.tasks.drone_2d import Drone2DEnv
from sim.schemas import DroneAction, _ZERO_ACTION


# Path where the training renderer reads live state from.
# Written atomically (tmp → rename) so the watcher never reads a torn file.
RENDER_STATE_FILE = "./checkpoints/drone_grpo/render_state.json"


# Module-level persistent environment 
# Created once, reused across all reward_fn calls.
# GRPOTrainer calls reward_fn many times per training step — we cannot
# afford to spawn/destroy 32 PyBullet clients on every call.

_reward_env: Optional[ParallelEnv] = None
_num_envs:   int = 32


def init_reward_env(num_envs: int = 32):
    """
    Call this ONCE before training starts to pre-warm the reward environments.
    Subsequent calls to reward_fn will reuse these envs.
    """
    global _reward_env, _num_envs
    _num_envs = num_envs
    _reward_env = ParallelEnv(Drone2DEnv, num_envs)
    _reward_env.reset_all()
    print(f"[reward_fn] Reward env ready: {num_envs} × Drone2DEnv")


def close_reward_env():
    """Call this after training.train() finishes."""
    global _reward_env
    if _reward_env is not None:
        _reward_env.close()
        _reward_env = None


# Core reward function 

_debug_call_count  = 0
_t_last_exit: float | None = None   # timestamp of previous reward_fn return


def sim_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    TRL-compatible reward function.

    Flow per call:
      1. Parse each completion string → DroneAction dict (with fallback)
      2. Parse the observation JSON out of each prompt string
      3. Teleport each env to the state described in its prompt (set_state_all)
      4. step_all(actions) on the persistent reward ParallelEnv
      5. Return the scalar rewards

    Each call is now a proper single-step counterfactual evaluation:
    "from THIS state (as described in the prompt), THIS action → THIS reward."
    This eliminates the prompt↔env mismatch that caused reward drift.

    Args:
        prompts:     list of chat-formatted prompt strings containing the obs JSON
        completions: list of model output strings (the "answers")

    Returns:
        list of float rewards, one per completion
    """
    global _reward_env, _debug_call_count, _t_last_exit
    t_enter = time.perf_counter()
    _debug_call_count += 1

    # ── Timing breakdown (printed every 5 calls) ──────────────────────────────
    # gap  = time since last reward_fn return
    #      ≈ vLLM generation + GRPO advantage compute + backward passes + optimizer + weight-sync
    # physics = time spent inside step_all()
    # total   = full time inside reward_fn
    if _debug_call_count % 5 == 1:
        sep = "-" * 60
        print(f"\n{sep}")
        print(f"  reward_fn call #{_debug_call_count}  |  {len(completions)} completions")
        if _t_last_exit is not None:
            gap = t_enter - _t_last_exit
            print(f"  time outside reward_fn (gen+train): {gap:.2f}s")
        print(sep)
        for i, c in enumerate(completions[:4]):
            action = _parse_completion(c)
            print(f"\n  ── completion [{i}] ──────────────────────────────────")
            print(f"  {c!r}")
            print(f"  → parsed action: {action}")
        print(f"{sep}\n")

    if _reward_env is None:
        # Lazy init if someone forgot to call init_reward_env()
        init_reward_env(_num_envs)
    assert _reward_env is not None   # narrow Optional[ParallelEnv] → ParallelEnv

    n = len(completions)

    # 1. Parse completions → actions 
    actions = []
    for i, completion in enumerate(completions):
        action = _parse_completion(completion)
        actions.append(action)

    # 2. Pad or truncate to match num_envs 
    # TRL's batch size may differ from num_envs. We handle this gracefully.
    if n < _num_envs:
        # Pad with zero actions for unused envs
        padded = actions + [dict(_ZERO_ACTION)] * (_num_envs - n)
    else:
        padded = actions[:_num_envs]

    # 3. Teleport each env to the state described in its prompt ───────────────
    # This is the core fix for the prompt↔env mismatch bug: before we step the
    # physics we reset every env to the exact drone position / velocity / goal
    # that the LLM saw when it generated its completion.  Each reward evaluation
    # then becomes a proper single-step counterfactual:
    #   "from THIS state (as in the prompt), THIS action → THIS outcome."
    obs_from_prompts = [_parse_obs_from_prompt(pr) for pr in prompts]
    if n < _num_envs:
        obs_padded = obs_from_prompts + [None] * (_num_envs - n)
    else:
        obs_padded = obs_from_prompts[:_num_envs]
    _reward_env.set_state_all(obs_padded)

    # 4. Step the physics — K steps from the set state, accumulating reward ──
    # K=80 (1.33 s of physics at 60 Hz).
    #   Run-9 diagnosis: 200 steps caused ~10 steps/rollout of wall contact
    #   on average, accumulating -10 of proximity penalty that completely
    #   drowned the navigation signal.  The wall proximity penalty has now
    #   been REMOVED from the reward function, so wall bouncing no longer
    #   ruins the reward.  With goals in the curriculum range 2–3 m and the
    #   4× progress multiplier, 80 steps is sufficient:
    #     goal at 2m → reachable in ~27 best-case steps
    #     goal at 3m → reachable in ~36 best-case steps
    #   Even at the ~35% directional efficiency of an early-stage policy,
    #   a 2m goal is reachable in ~80 steps.
    #   Shorter rollouts mean less accumulation of any per-step penalties and
    #   faster reward_fn calls (each training step is ~0.5 s lighter).
    #
    # Early termination per env: once an episode ends (goal reached → auto-
    # reset), the drone is now in a freshly-spawned random episode with a
    # different goal.  Continuing to accumulate would contaminate the signal,
    # so we mask out terminated envs and stop adding their rewards.
    _ROLLOUT_STEPS = 120

    t_phys_start = time.perf_counter()

    accumulated  = [0.0] * _num_envs
    done_mask    = [False] * _num_envs
    done_at_step = [-1] * _num_envs
    final_dist   = [999.0] * _num_envs

    # First step
    _, rewards, dones, _ = _reward_env.step_all(padded)
    for i in range(_num_envs):
        accumulated[i] += rewards[i]
        if dones[i] and not done_mask[i]:
            done_mask[i]    = True
            done_at_step[i] = 0

    # Remaining steps
    for step_idx in range(1, _ROLLOUT_STEPS):
        _, rewards, dones, _ = _reward_env.step_all(padded)
        for i in range(_num_envs):
            if not done_mask[i]:
                accumulated[i] += rewards[i]
                if dones[i]:
                    done_mask[i]    = True
                    done_at_step[i] = step_idx

    # Get final distances from live physics state
    try:
        snapshots = _reward_env.get_state_snapshot()
        for i, snap in enumerate(snapshots):
            if snap is not None and not done_mask[i]:
                dx = snap.get("drone_x", 0.0) - snap.get("goal_x", 0.0)
                dy = snap.get("drone_y", 0.0) - snap.get("goal_y", 0.0)
                final_dist[i] = (dx**2 + dy**2) ** 0.5
    except Exception:
        pass

    # Post-rollout reward shaping
    for i in range(_num_envs):
        if done_mask[i] and done_at_step[i] >= 0:
            efficiency_bonus = ((_ROLLOUT_STEPS - 1 - done_at_step[i]) / _ROLLOUT_STEPS) * 2.0
            accumulated[i] += efficiency_bonus
        else:
            terminal_bonus = max(0.0, (3.0 - final_dist[i]) * 0.5)
            accumulated[i] += terminal_bonus

    rewards = accumulated
    t_phys_end = time.perf_counter()

    # 5. Export state snapshot for the live training renderer (non-blocking)
    _export_render_state(_debug_call_count)

    # Log the timing breakdown every 5 calls
    if _debug_call_count % 5 == 1:
        phys_ms  = (t_phys_end - t_phys_start) * 1000
        total_ms = (t_phys_end - t_enter) * 1000
        n_parsed = sum(1 for o in obs_from_prompts if o is not None)
        print(f"  [timing] reward_fn total: {total_ms:.0f}ms  |  physics: {phys_ms:.0f}ms  ({_ROLLOUT_STEPS} steps/eval)")
        print(f"  [state-reset] parsed {n_parsed}/{n} prompt observations\n")

    _t_last_exit = time.perf_counter()
    # Return only the rewards corresponding to real completions
    return list(rewards[:n])


def _parse_completion(text: str) -> dict:
    """
    Robustly parse a model completion string into an action dict.

    Stages (in order):
      1. Direct JSON parse of the whole text
      2. Find any JSON object inside the text that has the right keys
      3. Extract any 3 numbers anywhere in the text and use them as forces
      4. Zero fallback (drone drifts, naturally penalised)

    Stage 3 is crucial during early training when the model hasn't yet learned
    to output JSON: even if the model writes "apply 5 Newtons forward and -3
    sideways with torque 0", we extract [5, -3, 0] and give the drone real
    forces.  This means every completion gets a non-trivial reward signal,
    so GRPO has something to differentiate on from the very first step.
    """
    text = text.strip()

    # Stage 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if all(k in obj for k in ("force_x", "force_y", "torque_z")):
                return DroneAction.model_validate(obj).model_dump()
            # Unwrap {"action": {...}}
            for v in obj.values():
                if isinstance(v, dict) and all(k in v for k in ("force_x", "force_y", "torque_z")):
                    return DroneAction.model_validate(v).model_dump()
    except Exception:
        pass

    # Stage 2: find a JSON object with the right keys anywhere in the text
    candidates = re.findall(r'\{[^{}]+\}', text)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and all(k in obj for k in ("force_x", "force_y", "torque_z")):
                return DroneAction.model_validate(obj).model_dump()
        except Exception:
            pass

    # Stage 3: extract any 3 numbers from the text
    # This handles free-form outputs like "apply force 5.0 in x, -3.0 in y, torque 0"
    # Skips numbers inside the observation JSON (which the model echoes back)
    # by looking for numbers NOT immediately preceded by a JSON key pattern.
    nums = [float(m) for m in re.findall(r'-?\d+(?:\.\d+)?', text)]
    if len(nums) >= 3:
        # Clamp to valid DroneAction range
        fx = max(-20.0, min(20.0, nums[0]))
        fy = max(-20.0, min(20.0, nums[1]))
        tz = max(-20.0, min(20.0, nums[2]))
        return {"force_x": fx, "force_y": fy, "torque_z": tz}

    # Stage 4: zero fallback
    return dict(_ZERO_ACTION)


def _parse_obs_from_prompt(prompt: str) -> dict | None:
    """
    Extract the observation JSON dict from a chat-formatted prompt string.

    The prompt is built in train.py as:
        <|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n
        <|im_start|>user\\n{json.dumps(obs)}<|im_end|>\\n
        <|im_start|>assistant\\n

    Strategy:
      1. Preferred: extract the content between the <|im_start|>user tag and
         the following <|im_end|>, then JSON-parse it directly.  This is fast
         and exact.
      2. Fallback: scan the full prompt text for the first JSON object that
         contains "drone_x" — handles any future chat-template changes.
         Uses bracket-depth counting to tolerate nested arrays (ray_distances).

    Returns the obs dict on success, None if parsing fails.  Callers that
    receive None simply skip the set_state call for that env, which is safe
    (the env stays wherever it was from the previous call).
    """
    # ── Strategy 1: extract user-turn content ──────────────────────────────
    m = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', prompt, re.DOTALL)
    if m:
        user_content = m.group(1).strip()
        try:
            obj = json.loads(user_content)
            if isinstance(obj, dict) and "drone_x" in obj:
                return obj
        except Exception:
            pass

    # ── Strategy 2: bracket-depth scan for any JSON object with drone_x ───
    for match in re.finditer(r'\{', prompt):
        start = match.start()
        depth = 0
        end = None
        for i, ch in enumerate(prompt[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break
        if end is None:
            continue
        candidate = prompt[start:end]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "drone_x" in obj:
                return obj
        except Exception:
            pass

    return None


# ── Live render state export ───────────────────────────────────────────────────

def _export_render_state(step: int) -> None:
    """
    Write the current reward-env physics state to RENDER_STATE_FILE so that
    training/watch_training.py can render it in a separate terminal.

    Uses write-to-temp + atomic rename so the watcher never reads a partial file.
    Failures are silently swallowed — a broken render export must never crash training.
    """
    global _reward_env
    if _reward_env is None:
        return
    try:
        snapshot = _reward_env.get_state_snapshot()
        payload = {
            "step":      step,
            "timestamp": time.time(),
            "snapshot":  snapshot,   # list of dicts (or None for uninitialised envs)
        }
        os.makedirs(os.path.dirname(os.path.abspath(RENDER_STATE_FILE)), exist_ok=True)
        tmp = RENDER_STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, RENDER_STATE_FILE)   # atomic on Linux/macOS
    except Exception:
        pass
