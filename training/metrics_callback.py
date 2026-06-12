"""
CSVMetricsCallback: writes every TRL training log dict to a JSONL file.

One JSON object per line, one per training step.  This is the file that
plot_live.py reads and plots.  JSONL is chosen over CSV because:
  - TRL adds/removes keys between steps (keys are not fixed)
  - Each line is independently valid JSON, so partial writes don't corrupt
  - A tail -f of the file is human-readable
"""

import json
import os

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class JSONLMetricsCallback(TrainerCallback):
    """
    Appends each on_log event to <output_file> as a single JSON line.

    Args:
        output_file: path to the .jsonl file (created / appended to automatically).
    """

    def __init__(self, output_file: str):
        self.output_file = output_file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        # Truncate on start so restarted runs don't mix with old data
        open(self.output_file, "w").close()
        print(f"[metrics] Logging to {self.output_file}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if not logs:
            return
        row = {
            "step":  state.global_step,
            "epoch": round(state.epoch or 0.0, 5),
            **{k: _safe_float(v) for k, v in logs.items()},
        }
        with open(self.output_file, "a") as f:
            f.write(json.dumps(row) + "\n")


def _safe_float(v) -> float | str:
    """Convert tensor/string metric values to plain Python types."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)
