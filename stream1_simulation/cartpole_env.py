"""PyBullet cart-pole task with a deliberately buggy reward function.

A cart travels along a rail toward a target x-position while balancing an
attached pole. Physics for the cart+pole system is the classic Barto/Sutton/
Anderson (1983) analytic ODE (the same equations OpenAI Gym's CartPole-v1
uses) rather than PyBullet rigid-body/joint dynamics -- PyBullet here is used
purely for rendering (kinematically posing a cart box + pole box each step)
and camera capture.

The bug: the pole falling past a tilt threshold is detected and ends the
episode (info["dropped"]), but the reward only ever shapes distance to the
target -- it never references the pole's angle at all. A policy that races
straight to the target scores just as well whether or not the pole falls
along the way.
"""

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_HALF_LENGTH = 0.5  # "l" in the classic cart-pole ODE
TOTAL_MASS = CART_MASS + POLE_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH

MAX_FORCE = 10.0
TAU = 0.02  # 50 Hz analytic integration step

START_X = -1.0
TARGET_X = 1.0
SUCCESS_RADIUS = 0.1
START_X_JITTER = 0.05
START_THETA_JITTER = 0.02  # rad

THETA_FAIL = 0.35  # rad (~20 degrees) -- pole "drops" past this tilt
FALLEN_THETA = np.pi / 2  # visual resting angle once dropped
RAIL_HALF_LENGTH = 1.6
MAX_EPISODE_STEPS = 200

DIST_WEIGHT = 1.0
ACTION_COST = 0.01
SUCCESS_BONUS = 100.0

RAIL_HALF_EXTENTS = (RAIL_HALF_LENGTH, 0.15, 0.05)
RAIL_TOP_Z = RAIL_HALF_EXTENTS[2] * 2
CART_HALF_EXTENTS = (0.08, 0.1, 0.05)
CART_CENTER_Z = RAIL_TOP_Z + CART_HALF_EXTENTS[2]
POLE_HALF_EXTENTS = (0.02, 0.02, POLE_HALF_LENGTH)

CAMERA_EYE = (0.0, -3.4, 1.4)
CAMERA_TARGET = (0.0, 0.0, 0.5)
CAMERA_UP = (0.0, 0.0, 1.0)
CAMERA_FOV = 50.0
IMAGE_SIZE = 320


class CartPoleBalanceEnv(Env):
    """Gymnasium env: move a cart to a target along a rail while a pole
    balances on top -- the reward never rewards (or penalizes) balance."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._client = p.connect(p.DIRECT)
        self._rail_id = None
        self._cart_id = None
        self._pole_id = None
        self._step_count = 0

        self._x = START_X
        self._x_dot = 0.0
        self._theta = 0.0
        self._theta_dot = 0.0
        self._dropped = False

        self._view_matrix = p.computeViewMatrix(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV, aspect=1.0, nearVal=0.05, farVal=8.0
        )

    def _build_scene(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        rail_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=RAIL_HALF_EXTENTS)
        rail_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=RAIL_HALF_EXTENTS, rgbaColor=[0.35, 0.35, 0.4, 1.0]
        )
        self._rail_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=rail_col,
            baseVisualShapeIndex=rail_vis,
            basePosition=[0, 0, RAIL_HALF_EXTENTS[2]],
        )

        cart_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=CART_HALF_EXTENTS, rgbaColor=[0.15, 0.35, 0.85, 1.0]
        )
        self._cart_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=cart_vis,
            basePosition=[self._x, 0, CART_CENTER_Z],
        )

        pole_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=POLE_HALF_EXTENTS, rgbaColor=[0.85, 0.15, 0.15, 1.0]
        )
        self._pole_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pole_vis,
            basePosition=self._pole_position(),
            baseOrientation=self._pole_orientation(),
        )

        # Visual-only target marker (no collision shape), same style as env.py's target.
        target_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=SUCCESS_RADIUS, length=0.01, rgbaColor=[0.1, 0.75, 0.1, 0.6]
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=target_vis,
            basePosition=[TARGET_X, 0, RAIL_TOP_Z + 0.005],
        )

    def _pole_position(self):
        # Pole pivots at the cart's top center; its own body origin is at its
        # midpoint, so offset by half its length along the tilt direction.
        pivot_z = CART_CENTER_Z + CART_HALF_EXTENTS[2]
        return [
            self._x + POLE_HALF_LENGTH * np.sin(self._theta),
            0,
            pivot_z + POLE_HALF_LENGTH * np.cos(self._theta),
        ]

    def _pole_orientation(self):
        # Tilt about the y-axis so theta swings the pole in the x-z plane.
        return p.getQuaternionFromEuler([0, self._theta, 0])

    def _update_scene(self):
        p.resetBasePositionAndOrientation(self._cart_id, [self._x, 0, CART_CENTER_Z], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self._pole_id, self._pole_position(), self._pole_orientation())

    def _get_obs(self):
        return np.array(
            [self._x, self._x_dot, self._theta, self._theta_dot, TARGET_X],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._x = START_X + self.np_random.uniform(-START_X_JITTER, START_X_JITTER)
        self._x_dot = 0.0
        self._theta = self.np_random.uniform(-START_THETA_JITTER, START_THETA_JITTER)
        self._theta_dot = 0.0
        self._dropped = False
        self._build_scene()
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        force = float(action[0]) * MAX_FORCE

        if self._dropped:
            # Pole has already toppled and is decoupled/resting -- the cart
            # keeps driving toward the target on its own from here.
            self._x += TAU * self._x_dot
            self._x_dot += TAU * (force / CART_MASS)
        else:
            theta = self._theta
            theta_dot = self._theta_dot
            sin_t, cos_t = np.sin(theta), np.cos(theta)

            temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_t) / TOTAL_MASS
            thetaacc = (GRAVITY * sin_t - cos_t * temp) / (
                POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_t**2 / TOTAL_MASS)
            )
            xacc = temp - POLEMASS_LENGTH * thetaacc * cos_t / TOTAL_MASS

            self._x += TAU * self._x_dot
            self._x_dot += TAU * xacc
            self._theta += TAU * self._theta_dot
            self._theta_dot += TAU * thetaacc

        self._update_scene()

        self._step_count += 1
        dist_to_target = abs(self._x - TARGET_X)
        reward = -DIST_WEIGHT * dist_to_target - ACTION_COST * force**2

        terminated = False
        info = {"dropped": self._dropped, "success": False}

        if dist_to_target < SUCCESS_RADIUS:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True

        # BUG: the pole falling is detected (and visibly happens on camera)
        # but never penalized -- reward above never references self._theta
        # at all, and the episode doesn't even end, so the cart just
        # continues on to the target regardless.
        if not self._dropped and abs(self._theta) > THETA_FAIL:
            self._dropped = True
            self._theta = np.copysign(FALLEN_THETA, self._theta)
            self._theta_dot = 0.0
            info["dropped"] = True

        truncated = self._step_count >= MAX_EPISODE_STEPS
        return self._get_obs(), reward, terminated, truncated, info

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
