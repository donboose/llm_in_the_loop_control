"""
Live renderer for the GRPO training run.

Run this in a SEPARATE terminal while training is running:

    PYTHONPATH=. uv run python training/watch_training.py

What you see:
  - 8×4 grid of 32 environments, each showing:
      - Grey walls
      - Brown rectangular obstacles
      - Green circle = goal
      - Blue circle = drone + yellow heading line
  - The window title shows the current training step and time since last update.
  - The grid updates every time the reward function is called during training
    (~once every 7 seconds per training step).
  - If training hasn't started yet the grid shows empty cells with "waiting" in
    the terminal.

The renderer runs entirely in this process — no GPU memory is shared with
training.  OpenGL only needs a few MB for the shader and framebuffer.
"""

import json
import os
import time

from rendering.renderer import Renderer
from training.reward_fn import RENDER_STATE_FILE

NUM_ENVS       = 32
POLL_INTERVAL  = 0.08   # seconds between file-change checks (≈12 Hz UI loop)
PRINT_INTERVAL = 5.0    # seconds between "waiting…" console prints


def _load_state(path: str) -> tuple[list[dict], int] | None:
    """
    Read and parse the render state file.
    Returns (snapshot, step) or None if the file is absent / not yet valid JSON.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data["snapshot"], data.get("step", 0)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def main():
    print("=" * 60)
    print("  Drone2D — Live Training Renderer")
    print(f"  Watching: {RENDER_STATE_FILE}")
    print("  Close the window to exit.")
    print("=" * 60)

    renderer = Renderer(num_envs=NUM_ENVS, title="Drone2D — Live Training View")

    snapshot:   list[dict] | None = None
    last_mtime: float = 0.0
    last_step:  int   = 0
    t_last_print = time.time() - PRINT_INTERVAL   # print immediately on first loop

    while not renderer.should_close():
        t_now = time.time()

        # ── Poll state file ───────────────────────────────────────────────────
        try:
            mtime = os.path.getmtime(RENDER_STATE_FILE)
            if mtime > last_mtime:
                result = _load_state(RENDER_STATE_FILE)
                if result is not None:
                    snapshot, last_step = result
                    last_mtime = mtime
                    # Update window title with current step
                    # (moderngl-window doesn't expose a title setter, so print instead)
        except FileNotFoundError:
            pass

        # ── Draw ─────────────────────────────────────────────────────────────
        if snapshot is not None:
            renderer.draw(snapshot)
            # Show step info every PRINT_INTERVAL seconds
            if t_now - t_last_print >= PRINT_INTERVAL:
                age = t_now - last_mtime
                print(
                    f"[watch] step={last_step:,}  "
                    f"last update {age:.1f}s ago  "
                    f"(update rate ~{age:.0f}s/step)"
                )
                t_last_print = t_now
        else:
            # Training hasn't started yet — render a blank grid so the window
            # stays responsive.
            renderer.draw([None] * NUM_ENVS)
            if t_now - t_last_print >= PRINT_INTERVAL:
                print(f"[watch] Waiting for training to write {RENDER_STATE_FILE} …")
                t_last_print = t_now

        time.sleep(POLL_INTERVAL)

    renderer.destroy()
    print("\n[watch] Window closed — exiting.")


if __name__ == "__main__":
    main()
