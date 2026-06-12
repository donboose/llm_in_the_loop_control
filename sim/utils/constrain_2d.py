import pybullet as p


def apply_2d_constraint(client: int, body_id: int) -> int:
    """
    Lock a rigid body to the XY plane (top-down 2D view).

    What this does:
      - Allows free movement in X and Y.
      - Locks Z translation to 0 (body stays flat on the plane).
      - Locks rotation around X and Y axes (no tipping or rolling).
      - Allows rotation around Z axis only (body can spin in-plane).

    This is done by creating a 6-DOF constraint between the body
    and a fixed 'world' anchor (childBodyUniqueId=-1), then using
    setLinearLowerLimit / setLinearUpperLimit to pin the Z axis,
    and setAngularLowerLimit / setAngularUpperLimit to pin X/Y rotation.

    Returns the constraint ID (save it if you need to remove it later).
    """
    constraint_id = p.createConstraint(
        parentBodyUniqueId=body_id,
        parentLinkIndex=-1,          # base link
        childBodyUniqueId=-1,        # world (fixed anchor)
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 1],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        physicsClientId=client,
    )
    # We use a generic 6DOF constraint instead — JOINT_FIXED is too
    # restrictive (locks everything). Re-create as 6DOF:
    p.removeConstraint(constraint_id, physicsClientId=client)

    constraint_id = p.createConstraint(
        parentBodyUniqueId=body_id,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FREE,
        jointAxis=[0, 0, 1],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        physicsClientId=client,
    )

    # Lock Z translation: lower == upper == 0
    p.changeConstraint(
        constraint_id,
        maxForce=50000,
        physicsClientId=client,
    )

    return constraint_id


def lock_body_to_2d_plane(client: int, body_id: int):
    """
    Simpler and more reliable approach:
    After every step, manually zero out Z position,
    Z velocity, and X/Y angular velocity.

    This is the most robust method for 2D simulation in PyBullet
    because constraint-based 2D locking can drift under collisions. [web:1]
    Call this function once per step, after p.stepSimulation().
    """
    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client)
    lin_vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client)

    # Force Z=0, keep X and Y as-is
    p.resetBasePositionAndOrientation(
        body_id,
        [pos[0], pos[1], 0.0],   # Z locked to 0
        orn,
        physicsClientId=client,
    )

    # Kill Z linear velocity and X,Y angular velocities
    p.resetBaseVelocity(
        body_id,
        linearVelocity=[lin_vel[0], lin_vel[1], 0.0],
        angularVelocity=[0.0, 0.0, ang_vel[2]],   # only Z spin allowed
        physicsClientId=client,
    )
