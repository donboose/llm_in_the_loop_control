from pydantic import BaseModel, Field
import json

_ZERO_ACTION = {"force_x": 0.0, "force_y": 0.0, "torque_z": 0.0}

class DroneAction(BaseModel):
    force_x:  float = Field(description="Thrust along world X axis in Newtons. Range: -20 to 20.")
    force_y:  float = Field(description="Thrust along world Y axis in Newtons. Range: -20 to 20.")
    torque_z: float = Field(description="Rotational torque around Z axis in Newtons. Range: -20 to 20.")


# Base schema from Pydantic
_base_schema = DroneAction.model_json_schema()

# Hardened schema for guided_json:
# - additionalProperties: false → model cannot add extra keys like "action"
# - required lists all three fields → grammar sampler must include all of them
# - No $defs, no nesting — flat object only
DRONE_ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "force_x":  {"type": "number"},
        "force_y":  {"type": "number"},
        "torque_z": {"type": "number"},
    },
    "required": ["force_x", "force_y", "torque_z"],
    "additionalProperties": False,
}


# Regex used by vLLM's constrained decoding during GRPO training.
# Forces the model to emit exactly {"force_x": N, "force_y": N, "torque_z": N}
# so completions are ~25 tokens and always parseable.
# Key order is fixed; optional decimal part; values bounded to 4 digits.
DRONE_ACTION_REGEX = (
    r'\{"force_x":\s*-?[0-9]{1,3}(\.[0-9]{1,4})?'
    r',\s*"force_y":\s*-?[0-9]{1,3}(\.[0-9]{1,4})?'
    r',\s*"torque_z":\s*-?[0-9]{1,3}(\.[0-9]{1,4})?\}'
)


SYSTEM_PROMPT = """You are a drone flight controller in a 2D top-down arena.

Arena: 10×10 metres, solid walls on all sides, rectangular obstacles inside.
Your drone starts near the origin. Navigate to the goal position.

You receive a JSON observation. You MUST respond with ONLY this exact JSON format:
{"force_x": <number>, "force_y": <number>, "torque_z": <number>}

Rules:
- force_x: thrust in X direction, range -20 to 20 Newtons
- force_y: thrust in Y direction, range -20 to 20 Newtons  
- torque_z: spin torque, range -20 to 20 Newtons
- Apply force toward (dx_to_goal, dy_to_goal) to move toward the goal
- Reduce force when dist_to_goal < 1.0 to avoid overshooting
- If any ray_distance < 0.5, steer away from that direction
- Do NOT wrap your answer. Do NOT add explanation. Output ONLY the JSON object.

Hint:
- To move toward the goal, set force_x = dx_to_goal * k and force_y = dy_to_goal * k where k is between 1.0 and 5.0. Adjust based on current velocity to avoid overshoot. Reach the goal and stay there to gain maximum reward.

Example correct output:
{"force_x": 8.5, "force_y": -3.2, "torque_z": 0.0}"""
