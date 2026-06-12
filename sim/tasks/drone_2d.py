import math
import random
import pybullet as p
import numpy as np

from sim.env_base import BaseEnv
from sim.utils.constrain_2d import lock_body_to_2d_plane


# ── World constants ────────────────────────────────────────────────
WORLD_SIZE   = 10.0      # the arena is a square: [-5, 5] x [-5, 5] metres
WALL_T       = 0.2       # wall thickness in metres
DRONE_RADIUS = 0.2       # collision radius of the drone disc
DRONE_MASS   = 1.0       # kg
GOAL_RADIUS  = 0.8       # how close the drone must get to the goal centre
MAX_FORCE    = 20.0      # Newtons, maximum thrust in any direction
DRAG         = 0.85      # linear velocity damping (mimics air drag)
ANG_DRAG     = 0.85      # angular damping
DT           = 1 / 60   # physics timestep (60 Hz)
MAX_STEPS    = 1000      # episode length cap
NUM_OBSTACLES = 2        # randomly placed rectangular obstacles


class Drone2DEnv(BaseEnv):
    """
    Top-down 2D drone navigation with:
      - A disc-shaped drone (cylinder, Z locked)
      - Solid walls on all four sides
      - Random rectangular obstacles
      - A circular goal region
      - Continuous force/torque action space
    """

    def __init__(self, max_goal_dist: float = 3.0):
        """
        Args:
            max_goal_dist: Upper bound on the distance from the drone spawn
                (origin) to the randomly-placed goal.  Set to a small value
                (e.g. 3.0) during curriculum training to bias the dataset
                toward close goals the early policy can realistically reach.
                Default 7.1 covers the full 10×10 arena.
        """
        # Each env gets its own isolated physics client in DIRECT mode.
        # pybullet.DIRECT = no GUI, no window, pure CPU physics.
        self.client = p.connect(p.DIRECT)
        self._max_goal_dist = max_goal_dist
        self._step_count = 0
        self._drone_id = None
        self._obstacle_ids = []
        self._goal_pos = None
        self._prev_dist: float | None = None   # for progress reward
        # One-time proximity milestones — reset each rollout via set_state()
        self._milestone_1_5_reached: bool = False
        self._milestone_1_0_reached: bool = False
        self.reset()

    # ── Internal helpers ───────────────────────────────────────────

    def _setup_physics(self):
        """Configure the physics world. Called once per reset."""
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        # 60 Hz timestep, 10 substeps for stability
        p.setPhysicsEngineParameter(
            fixedTimeStep=DT,
            numSubSteps=10,
            physicsClientId=self.client,
        )

    def _create_walls(self):
        """
        Build four solid walls around the arena using thin boxes.
        Walls are static (mass=0), so they never move.

        Arena: [-HALF, HALF] x [-HALF, HALF]
        Wall layout:
          Bottom: along -Y edge
          Top:    along +Y edge
          Left:   along -X edge
          Right:  along +X edge
        """
        half = WORLD_SIZE / 2
        t = WALL_T / 2  # half-thickness
        h = 0.1          # half-height (we're 2D, keep this small)

        wall_specs = [
            # (half-extents,          position)
            ([half + t, t, h],       [0,           -(half + t), 0]),  # bottom
            ([half + t, t, h],       [0,            (half + t), 0]),  # top
            ([t, half + t, h],       [-(half + t),  0,          0]),  # left
            ([t, half + t, h],       [ (half + t),  0,          0]),  # right
        ]

        for extents, pos in wall_specs:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=extents,
                physicsClientId=self.client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=extents,
                rgbaColor=[0.3, 0.3, 0.3, 1],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0,            # mass=0 → static, immovable
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
                physicsClientId=self.client,
            )

    def _create_obstacles(self):
        """
        Spawn NUM_OBSTACLES random rectangular boxes inside the arena.
        Obstacles are static.  Placement rules (all must pass or the
        obstacle is skipped entirely rather than placed in a bad spot):

          1. At least 2.0 m from the drone spawn point (origin).
          2. At least 2.0 m from the goal centre.
          3. At least 1.5 m from every already-placed obstacle centre.
          4. Fully inside the inner arena boundary (wall-clearance margin).

        With NUM_OBSTACLES = 2 and a 10×10 m arena these constraints are
        almost always satisfiable; the 50-attempt retry gives enough tries.
        """
        half = WORLD_SIZE / 2 - 1.5   # inner boundary — keep clear of walls
        self._obstacle_ids = []
        placed: list[tuple[float, float]] = []   # centres of obstacles placed so far

        for _ in range(NUM_OBSTACLES):
            valid_pos: tuple[float, float, float, float] | None = None

            for _ in range(50):
                ox = random.uniform(-half, half)
                oy = random.uniform(-half, half)
                ow = random.uniform(0.3, 0.8)   # smaller max — less arena blockage
                oh = random.uniform(0.3, 0.8)

                # Rule 1: clear of drone spawn
                if math.hypot(ox, oy) < 2.0:
                    continue
                # Rule 2: clear of goal
                if self._goal_pos and math.hypot(
                    ox - self._goal_pos[0], oy - self._goal_pos[1]
                ) < 2.0:
                    continue
                # Rule 3: clear of already-placed obstacles
                if any(math.hypot(ox - px, oy - py) < 1.5 for px, py in placed):
                    continue

                valid_pos = (ox, oy, ow, oh)
                break

            if valid_pos is None:
                # Could not find a valid position — skip this obstacle entirely
                # rather than placing it in a logically bad spot.
                continue

            ox, oy, ow, oh = valid_pos
            placed.append((ox, oy))

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[ow / 2, oh / 2, 0.1],
                physicsClientId=self.client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[ow / 2, oh / 2, 0.1],
                rgbaColor=[0.6, 0.3, 0.1, 1],
                physicsClientId=self.client,
            )
            obs_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[ox, oy, 0],
                physicsClientId=self.client,
            )
            self._obstacle_ids.append(obs_id)

    def _create_drone(self):
        """
        The drone is a thin cylinder (disc) with mass DRONE_MASS.
        We model it as a circle for clean top-down collision.
        High damping mimics air resistance — without it the drone
        would slide forever on the frictionless 2D plane.
        """
        col = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=DRONE_RADIUS,
            height=0.05,
            physicsClientId=self.client,
        )
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=DRONE_RADIUS,
            length=0.05,
            rgbaColor=[0.1, 0.5, 0.9, 1],
            physicsClientId=self.client,
        )
        self._drone_id = p.createMultiBody(
            baseMass=DRONE_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.0, 0.0, 0.0],   # spawn at origin
            physicsClientId=self.client,
        )
        # High damping = air drag; this is essential in 2D
        p.changeDynamics(
            self._drone_id, -1,
            linearDamping=DRAG,
            angularDamping=ANG_DRAG,
            physicsClientId=self.client,
        )

    def _pick_goal(self) -> list:
        """
        Place the goal at a random position satisfying:
          - Inside the inner arena boundary
          - Between 2.0 m and self._max_goal_dist from the drone spawn (origin)

        self._max_goal_dist enables curriculum training: setting it to 3.0
        during early training generates only close goals (2–3 m) that the
        policy can reliably reach, making the goal bonus fire early.
        The default (7.1) covers the full arena.
        """
        half = WORLD_SIZE / 2 - 1.0
        for _ in range(200):
            gx = random.uniform(-half, half)
            gy = random.uniform(-half, half)
            d  = math.hypot(gx, gy)
            if 2.0 < d <= self._max_goal_dist:
                return [gx, gy]
        # Fallback: a point guaranteed to be in the valid range
        return [2.5, 0.0]

    def _remove_all_bodies(self):
        """Wipe every body from the physics world before re-populating."""
        for body_id in range(p.getNumBodies(physicsClientId=self.client)):
            try:
                p.removeBody(body_id, physicsClientId=self.client)
            except Exception:
                pass
        self._obstacle_ids = []
        self._drone_id = None

    # ── BaseEnv interface ──────────────────────────────────────────

    def reset(self) -> dict:
        """
        Full episode reset:
          1. Wipe all bodies
          2. Re-configure physics
          3. Pick a random goal
          4. Spawn walls, obstacles, drone
        """
        p.resetSimulation(physicsClientId=self.client)
        self._setup_physics()
        self._step_count = 0
        self._prev_dist = None   # reset baseline for progress reward
        self._milestone_1_5_reached = False
        self._milestone_1_0_reached = False

        self._goal_pos = self._pick_goal()
        self._create_walls()
        self._create_obstacles()
        self._create_drone()

        # Run a few settling steps so the drone isn't in a weird initial state
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)
            lock_body_to_2d_plane(self.client, self._drone_id)

        obs = self.get_obs()
        self._prev_dist = obs["dist_to_goal"]   # establish baseline before first step
        return obs

    def step(self, action: dict) -> tuple[dict, float, bool]:
        """
        Apply one control step.

        Action schema:
          {
            "force_x": float,   # thrust in world X  [-MAX_FORCE, MAX_FORCE]
            "force_y": float,   # thrust in world Y  [-MAX_FORCE, MAX_FORCE]
            "torque_z": float,  # spin torque        [-MAX_FORCE, MAX_FORCE]
          }

        Forces are clipped to MAX_FORCE before application.
        """
        fx = float(np.clip(action.get("force_x", 0.0), -MAX_FORCE, MAX_FORCE))
        fy = float(np.clip(action.get("force_y", 0.0), -MAX_FORCE, MAX_FORCE))
        tz = float(np.clip(action.get("torque_z", 0.0), -MAX_FORCE, MAX_FORCE))

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self.client
        )

        # Apply force at the drone's current centre, in world frame
        p.applyExternalForce(
            self._drone_id, -1,
            forceObj=[fx, fy, 0.0],
            posObj=pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        # Apply torque (spin the drone in-plane)
        p.applyExternalTorque(
            self._drone_id, -1,
            torqueObj=[0.0, 0.0, tz],
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        p.stepSimulation(physicsClientId=self.client)

        # ← The 2D lock: zero out Z drift and off-plane rotation
        lock_body_to_2d_plane(self.client, self._drone_id)

        self._step_count += 1
        obs = self.get_obs()
        reward, done = self._compute_reward(obs)
        return obs, reward, done

    def get_obs(self) -> dict:
        """
        Serializable observation dict.
        All values are plain Python floats/lists — ready for JSON.

        Includes:
          - drone position and heading
          - drone linear and angular velocity
          - vector and distance to goal
          - nearest obstacle distances (8 raycasts at 45° intervals)
          - goal position (absolute, for LLM context)
        """
        pos, orn = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self.client
        )
        lin_vel, ang_vel = p.getBaseVelocity(
            self._drone_id, physicsClientId=self.client
        )

        # Heading: yaw angle from quaternion
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]

        dx = self._goal_pos[0] - pos[0]
        dy = self._goal_pos[1] - pos[1]
        dist_to_goal = math.hypot(dx, dy)

        # 8-directional raycasts for obstacle sensing
        ray_len = 3.0
        ray_distances = []
        for i in range(8):
            angle = yaw + i * (math.pi / 4)
            ray_end = [
                pos[0] + ray_len * math.cos(angle),
                pos[1] + ray_len * math.sin(angle),
                0.0,
            ]
            hit = p.rayTest(
                [pos[0], pos[1], 0.0],
                ray_end,
                physicsClientId=self.client,
            )
            # hit[0][2] is the hit fraction (0=hit at origin, 1=no hit)
            ray_distances.append(round(float(hit[0][2]) * ray_len, 3))

        return {
            "drone_x":        round(float(pos[0]), 4),
            "drone_y":        round(float(pos[1]), 4),
            "drone_yaw":      round(float(yaw), 4),
            "vel_x":          round(float(lin_vel[0]), 4),
            "vel_y":          round(float(lin_vel[1]), 4),
            "ang_vel_z":      round(float(ang_vel[2]), 4),
            "goal_x":         round(float(self._goal_pos[0]), 4),
            "goal_y":         round(float(self._goal_pos[1]), 4),
            "dx_to_goal":     round(float(dx), 4),
            "dy_to_goal":     round(float(dy), 4),
            "dist_to_goal":   round(float(dist_to_goal), 4),
            "ray_distances":  ray_distances,    # list of 8 floats
            "step":           self._step_count,
        }

    def _compute_reward(self, obs: dict) -> tuple[float, bool]:
        """
        Navigation-focused reward shaping.

        Run-10 redesign rationale:
          Runs 8 and 9 showed that the wall proximity penalty (-0.3/-0.7 per
          step) accumulated to ~-10 per 200-step rollout, completely drowning
          the navigation signal (~0 for a random policy).  The model optimised
          for "avoid walls by applying tiny forces" rather than "navigate to
          goal".  The proximity penalty has been REMOVED; the physics already
          penalises wall contact through reduced progress reward (bouncing off
          a wall moves the drone away from the goal).

        Components:
          (1) Progress reward  [dominant signal — 4× multiplier]
              (prev_dist - curr_dist) × 4.0
              0.1 m closer → +0.40  (doubled from previous 0.20)
              Makes the navigation signal loud enough to dominate the loss.

          (2) Velocity alignment bonus  [secondary signal]
              alignment × 0.05 per step (up from 0.03).
              Rewards the drone for pointing its velocity vector at the goal,
              giving a per-step signal even before distance changes.

          (3) Alive penalty  [efficiency pressure]
              -0.005 per step.  Small enough to not crowd out (1) and (2).

          (4) One-time proximity milestones  [bridge to goal bonus]
              +1.0 first time dist < 1.5 m  (up from +0.5)
              +2.0 first time dist < 1.0 m  (up from +1.0)
              With goals at 2–3 m (curriculum), these fire when the policy
              is 50–67 % of the way home — providing a strong intermediate
              signal before the full +10 goal bonus.

          (5) Goal bonus  [terminal reward]
              +10.0 when dist < GOAL_RADIUS (0.5 m).  Raised from +5 so the
              advantage for a goal-reaching completion is always the largest
              signal in the group, regardless of other reward variance.

          (6) Episode timeout  [episode termination]
        """
        dist = obs["dist_to_goal"]
        done = False
        reward = 0.0

        # (1) Progress: louder multiplier so navigation dominates
        if self._prev_dist is not None:
            progress = self._prev_dist - dist    # positive → drone got closer
            reward += progress * 4.0             # 0.1 m closer → +0.40
        self._prev_dist = dist

        # (2) Velocity alignment with goal direction
        vel_x, vel_y = obs["vel_x"], obs["vel_y"]
        vel_mag = math.hypot(vel_x, vel_y)
        if vel_mag > 0.05 and dist > GOAL_RADIUS:
            goal_ux = obs["dx_to_goal"] / dist
            goal_uy = obs["dy_to_goal"] / dist
            alignment = (vel_x / vel_mag) * goal_ux + (vel_y / vel_mag) * goal_uy
            reward += alignment * 0.05           # fully aligned → +0.05/step

        # (3) Alive penalty
        reward -= 0.005

        # (4) One-time proximity milestones — fire at most once per rollout.
        # Flags reset in set_state() at the start of every reward rollout.
        if not self._milestone_1_5_reached and dist < 1.5:
            reward += 1.0
            self._milestone_1_5_reached = True
        if not self._milestone_1_0_reached and dist < 1.0:
            reward += 2.0
            self._milestone_1_0_reached = True

        # (5) Goal reached
        if dist < GOAL_RADIUS:
            reward += 10.0
            done = True

        # (6) Episode timeout
        if self._step_count >= MAX_STEPS:
            done = True

        return round(reward, 4), done

    def set_state(self, obs: dict) -> None:
        """
        Teleport the drone to the state described in *obs* without running
        physics.  Used by reward_fn so every reward evaluation starts from
        exactly the state described in the corresponding prompt.

        The call uses PyBullet's reset functions (no force integration),
        so the change takes effect instantly — no simulation steps are run.

        Args:
            obs: Observation dict as returned by get_obs(), containing at
                 minimum: drone_x, drone_y, drone_yaw, vel_x, vel_y,
                 ang_vel_z, goal_x, goal_y, dist_to_goal.
        """
        pos = [float(obs["drone_x"]), float(obs["drone_y"]), 0.0]
        orn = p.getQuaternionFromEuler([0.0, 0.0, float(obs.get("drone_yaw", 0.0))])
        p.resetBasePositionAndOrientation(
            self._drone_id, pos, orn, physicsClientId=self.client
        )

        lin_vel = [float(obs["vel_x"]), float(obs["vel_y"]), 0.0]
        ang_vel = [0.0, 0.0, float(obs.get("ang_vel_z", 0.0))]
        p.resetBaseVelocity(
            self._drone_id, lin_vel, ang_vel, physicsClientId=self.client
        )

        self._goal_pos  = [float(obs["goal_x"]), float(obs["goal_y"])]
        self._prev_dist = float(obs["dist_to_goal"])
        # Reset step counter so episode-timeout logic starts fresh for this
        # single-step counterfactual evaluation.
        self._step_count = 0
        # Reset proximity milestones so they can fire once in this rollout.
        self._milestone_1_5_reached = False
        self._milestone_1_0_reached = False

    def close(self):
        """Disconnect this environment's physics client."""
        p.disconnect(physicsClientId=self.client)
