"""PyBullet pushing task with a deliberately buggy reward function.

A sphere "pusher" is velocity-controlled in the table plane to push a cube
toward a target zone near the table's front edge. The reward shapes distance
to target but never penalizes the object falling off the table -- that
omission is the injected bug the rest of the pipeline is built to diagnose.
"""

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces

TABLE_HALF_EXTENTS = (0.5, 0.5, 0.3)
TABLE_TOP_Z = TABLE_HALF_EXTENTS[2] * 2  # table sits on the ground plane

OBJECT_HALF_EXTENT = 0.03
OBJECT_START_XY = (0.0, -0.3)
OBJECT_MASS = 0.05

PUSHER_RADIUS = 0.04
PUSHER_START_XY = (0.0, -0.42)
PUSHER_MASS = 0.2
PUSHER_BOUND = 0.55  # xy clamp, slightly beyond the table so it can trail the object near the edge

TARGET_XY = (0.0, 0.35)  # deliberately close to the y=+0.5 edge

MAX_SPEED = 0.5  # m/s
PHYSICS_HZ = 240
SUBSTEPS_PER_ACTION = 8  # -> ~30 Hz control
MAX_EPISODE_STEPS = 200

DIST_WEIGHT = 2.0
ALIGN_WEIGHT = 0.5  # keeps the pusher lined up behind the object in x so contact isn't lost mid-push
ACTION_COST = 0.01
SUCCESS_RADIUS = 0.05
SUCCESS_BONUS = 50.0
DROP_MARGIN = 0.1  # object counts as "dropped" once its z falls this far below the table top

CAMERA_EYE = (0.0, -1.1, 1.3)
CAMERA_TARGET = (0.0, 0.15, TABLE_TOP_Z)
CAMERA_UP = (0.0, 0.0, 1.0)
CAMERA_FOV = 50.0
IMAGE_SIZE = 320


class PushEnv(Env):
    """Gymnasium env: push a cube toward a target near a table edge."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._client = p.connect(p.DIRECT)
        self._plane_id = None
        self._table_id = None
        self._object_id = None
        self._pusher_id = None
        self._step_count = 0

        self._view_matrix = p.computeViewMatrix(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV, aspect=1.0, nearVal=0.05, farVal=5.0
        )

    def _build_scene(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.8, physicsClientId=self._client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._plane_id = p.loadURDF("plane.urdf")

        table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS)
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=TABLE_HALF_EXTENTS, rgbaColor=[0.55, 0.4, 0.25, 1.0]
        )
        self._table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0, 0, TABLE_HALF_EXTENTS[2]],
        )

        obj_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[OBJECT_HALF_EXTENT] * 3)
        obj_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[OBJECT_HALF_EXTENT] * 3, rgbaColor=[0.85, 0.15, 0.15, 1.0]
        )
        self._object_id = p.createMultiBody(
            baseMass=OBJECT_MASS,
            baseCollisionShapeIndex=obj_col,
            baseVisualShapeIndex=obj_vis,
            basePosition=[
                OBJECT_START_XY[0],
                OBJECT_START_XY[1],
                TABLE_TOP_Z + OBJECT_HALF_EXTENT + 0.001,
            ],
        )
        p.changeDynamics(self._object_id, -1, lateralFriction=0.6)

        pusher_col = p.createCollisionShape(p.GEOM_SPHERE, radius=PUSHER_RADIUS)
        pusher_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=PUSHER_RADIUS, rgbaColor=[0.15, 0.35, 0.85, 1.0]
        )
        self._pusher_id = p.createMultiBody(
            baseMass=PUSHER_MASS,
            baseCollisionShapeIndex=pusher_col,
            baseVisualShapeIndex=pusher_vis,
            basePosition=[
                PUSHER_START_XY[0],
                PUSHER_START_XY[1],
                TABLE_TOP_Z + PUSHER_RADIUS + 0.001,
            ],
        )
        p.changeDynamics(self._pusher_id, -1, lateralFriction=0.6)

        # Visual-only target marker (no collision shape).
        target_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=SUCCESS_RADIUS, length=0.002, rgbaColor=[0.1, 0.75, 0.1, 0.6]
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=target_vis,
            basePosition=[TARGET_XY[0], TARGET_XY[1], TABLE_TOP_Z + 0.002],
        )

    def _get_obs(self):
        obj_pos, _ = p.getBasePositionAndOrientation(self._object_id)
        obj_vel, _ = p.getBaseVelocity(self._object_id)
        pusher_pos, _ = p.getBasePositionAndOrientation(self._pusher_id)
        return np.array(
            [
                pusher_pos[0],
                pusher_pos[1],
                obj_pos[0],
                obj_pos[1],
                obj_pos[2],
                obj_vel[0],
                obj_vel[1],
                TARGET_XY[0],
                TARGET_XY[1],
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._build_scene()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        vx, vy = action * MAX_SPEED
        p.resetBaseVelocity(self._pusher_id, linearVelocity=[vx, vy, 0])

        for _ in range(SUBSTEPS_PER_ACTION):
            p.stepSimulation()

        # Pusher is an idealized xy-only actuator: clamp height and workspace
        # bounds so it never itself falls off -- only the object's fate
        # (governed by the reward bug) is the thing under test.
        pos, orn = p.getBasePositionAndOrientation(self._pusher_id)
        clamped_xy = np.clip(pos[:2], -PUSHER_BOUND, PUSHER_BOUND)
        fixed_z = TABLE_TOP_Z + PUSHER_RADIUS + 0.001
        p.resetBasePositionAndOrientation(
            self._pusher_id, [clamped_xy[0], clamped_xy[1], fixed_z], orn
        )

        self._step_count += 1
        obs = self._get_obs()
        pusher_xy = obs[0:2]
        obj_xy = obs[2:4]
        obj_z = obs[4]

        dist_xy = float(np.linalg.norm(obj_xy - np.array(TARGET_XY)))
        align_penalty = abs(float(pusher_xy[0]) - float(obj_xy[0]))
        reward = (
            -DIST_WEIGHT * dist_xy
            - ALIGN_WEIGHT * align_penalty
            - ACTION_COST * float(np.sum(action**2))
        )

        terminated = False
        info = {"dropped": False, "success": False}

        if dist_xy < SUCCESS_RADIUS and obj_z > TABLE_TOP_Z - DROP_MARGIN:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True

        # BUG: dropping the object ends the episode but is never penalized.
        if obj_z < TABLE_TOP_Z - DROP_MARGIN:
            terminated = True
            info["dropped"] = True

        truncated = self._step_count >= MAX_EPISODE_STEPS
        return obs, reward, terminated, truncated, info

    def get_camera_image(self):
        """Render the current scene to an (IMAGE_SIZE, IMAGE_SIZE, 3) uint8 RGB array."""
        _, _, rgba, _, _ = p.getCameraImage(
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            viewMatrix=self._view_matrix,
            projectionMatrix=self._proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgba = np.reshape(rgba, (IMAGE_SIZE, IMAGE_SIZE, 4))
        return rgba[:, :, :3].astype(np.uint8)

    def close(self):
        if p.isConnected(self._client):
            p.disconnect(self._client)
