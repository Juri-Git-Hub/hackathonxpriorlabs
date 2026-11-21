"""2nd-order rigid-body drone dynamics, wind drag, attitude, obstacle avoidance, and battery."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from shared.schemas import Vec3
from sim.world import World

DroneStatus = Literal["searching", "tracking", "locked", "depleted"]


class Drone:
    """Quadrotor translational dynamics with aerodynamic drag and thrust-derived attitude."""

    MASS = 1.5          # kg
    GRAVITY = 9.81      # m/s^2
    K_DRAG = 0.035      # realistic aerodynamic drag (0.5 * rho * Cd * A) for 1.5 kg quadrotor
    MAX_SPEED_XY = 8.0  # m/s
    MAX_SPEED_Z = 3.0   # m/s
    MAX_TILT_RAD = math.radians(28.0)
    MIN_ALTITUDE = 1.5  # m
    BUILDING_CLEARANCE = 2.0  # m

    # Battery model tuned for ~3.5 minutes of active flight (100% -> 0%)
    P_BASE = 25.0
    K_T = 1.8
    K_V = 0.35
    BATTERY_CAPACITY_JOULES = 27000.0

    def __init__(self, drone_id: str, start_pos: Vec3) -> None:
        self.id = drone_id
        self.start_pos = start_pos

        self.pos = np.array(start_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.rot_rpy = np.zeros(3, dtype=np.float64)
        self.target_pos = np.array(start_pos, dtype=np.float64)

        self.battery_pct: float = 100.0
        self.samples_taken: int = 0
        self.last_ppm: float = 0.0
        self.status: DroneStatus = "searching"

    def reset(self, start_pos: Vec3 | None = None) -> None:
        if start_pos is not None:
            self.start_pos = start_pos
        self.pos = np.array(self.start_pos, dtype=np.float64)
        self.vel[:] = 0.0
        self.rot_rpy[:] = 0.0
        self.target_pos = np.array(self.start_pos, dtype=np.float64)
        self.battery_pct = 100.0
        self.samples_taken = 0
        self.last_ppm = 0.0
        self.status = "searching"

    def set_target(self, target_pos: Vec3) -> None:
        self.target_pos[0] = float(np.clip(target_pos[0], -58.0, 58.0))
        self.target_pos[1] = float(np.clip(target_pos[1], -58.0, 58.0))
        self.target_pos[2] = float(np.clip(target_pos[2], self.MIN_ALTITUDE, 28.0))

    def _repulsive_acceleration(self, world: World) -> np.ndarray:
        """Repulsive potential field keeping drone >= 2.0m from buildings and ground."""
        a_rep = np.zeros(3, dtype=np.float64)
        influence_dist = self.BUILDING_CLEARANCE + 2.5
        to_target = self.target_pos - self.pos
        to_target_xy = to_target[:2]

        for i in range(len(world.buildings)):
            lo = world._bldg_min[i]
            hi = world._bldg_max[i]
            closest = np.clip(self.pos, lo, hi)
            diff = self.pos - closest
            dist = float(np.linalg.norm(diff))

            if dist < influence_dist:
                if dist < 1e-4:
                    # Push outward from building center
                    away = self.pos - world._bldg_center[i]
                    n_away = float(np.linalg.norm(away))
                    direction = away / n_away if n_away > 1e-4 else np.array([0.0, 0.0, 1.0])
                    mag = 24.0
                else:
                    direction = diff / dist
                    penetration = max(0.0, influence_dist - dist)
                    mag = 6.0 * (penetration / max(dist - self.BUILDING_CLEARANCE * 0.5, 0.3))
                clamped_mag = min(mag, 24.0)
                a_rep += direction * clamped_mag

                # Tangential slide around vertical walls to avoid head-on local minima
                horiz_norm = math.hypot(direction[0], direction[1])
                if horiz_norm > 0.3:
                    t1 = np.array([-direction[1] / horiz_norm, direction[0] / horiz_norm, 0.0])
                    # Choose tangent side that aligns better with target (default +t1)
                    if float(np.dot(t1[:2], to_target_xy)) < 0.0:
                        t1 = -t1
                    a_rep += t1 * (0.85 * clamped_mag)
                    # Gentle upward lift to clear low obstacles like pipe racks
                    if self.pos[2] < hi[2] + 1.5:
                        a_rep[2] += 0.45 * clamped_mag

        # Ground repulsion below MIN_ALTITUDE + 0.5
        if self.pos[2] < self.MIN_ALTITUDE + 0.5:
            a_rep[2] += 6.0 * (self.MIN_ALTITUDE + 0.5 - self.pos[2])

        return a_rep

    def step(self, dt: float, wind_local: np.ndarray, world: World) -> None:
        """Advance drone dynamics by dt seconds."""
        if self.battery_pct <= 0.0:
            self.battery_pct = 0.0
            self.status = "depleted"
            self.vel *= 0.85
            self.rot_rpy[:2] *= 0.85
            return

        # 1. Cascaded PD position controller
        err = self.target_pos - self.pos
        v_des = 1.2 * err

        # Clamp horizontal and vertical desired speeds
        v_xy_norm = math.hypot(v_des[0], v_des[1])
        if v_xy_norm > self.MAX_SPEED_XY:
            scale = self.MAX_SPEED_XY / v_xy_norm
            v_des[0] *= scale
            v_des[1] *= scale
        v_des[2] = float(np.clip(v_des[2], -self.MAX_SPEED_Z, self.MAX_SPEED_Z))

        a_cmd = 2.6 * (v_des - self.vel) + self._repulsive_acceleration(world)

        # 2. Aerodynamic drag from relative wind
        v_rel = self.vel - wind_local
        v_rel_norm = float(np.linalg.norm(v_rel))
        f_drag = -self.K_DRAG * v_rel_norm * v_rel

        # 3. Required thrust vector T = m * (a_cmd + g) - F_drag
        g_vec = np.array([0.0, 0.0, self.GRAVITY], dtype=np.float64)
        thrust = self.MASS * (a_cmd + g_vec) - f_drag

        # Ensure positive upward thrust and clamp tilt to MAX_TILT_RAD
        thrust[2] = max(thrust[2], 0.25 * self.MASS * self.GRAVITY)
        t_xy = math.hypot(thrust[0], thrust[1])
        max_t_xy = thrust[2] * math.tan(self.MAX_TILT_RAD)
        if t_xy > max_t_xy and t_xy > 1e-6:
            scale = max_t_xy / t_xy
            thrust[0] *= scale
            thrust[1] *= scale

        # 4. Integrate translational motion
        accel = (thrust + f_drag) / self.MASS - g_vec
        self.vel += accel * dt

        # Soft velocity cap for numerical stability
        v_xy = math.hypot(self.vel[0], self.vel[1])
        if v_xy > self.MAX_SPEED_XY * 1.25:
            s = (self.MAX_SPEED_XY * 1.25) / v_xy
            self.vel[0] *= s
            self.vel[1] *= s
        self.vel[2] = float(np.clip(self.vel[2], -self.MAX_SPEED_Z * 1.5, self.MAX_SPEED_Z * 1.5))

        self.pos += self.vel * dt

        # Hard safety enforcement: bounds, altitude, and building clearance
        self.pos[0] = float(np.clip(self.pos[0], world.bounds_min[0] + 1.5, world.bounds_max[0] - 1.5))
        self.pos[1] = float(np.clip(self.pos[1], world.bounds_min[1] + 1.5, world.bounds_max[1] - 1.5))
        self.pos[2] = float(np.clip(self.pos[2], self.MIN_ALTITUDE, world.bounds_max[2] - 1.5))

        if world.is_inside_building(self.pos_tuple, margin=self.BUILDING_CLEARANCE):
            pushed = world.push_outside_buildings(self.pos[None, :], margin=self.BUILDING_CLEARANCE)[0]
            normal = pushed - self.pos
            n_norm = float(np.linalg.norm(normal))
            self.pos = pushed
            if n_norm > 1e-6:
                n_hat = normal / n_norm
                v_into = float(np.dot(self.vel, n_hat))
                if v_into < 0.0:
                    self.vel -= v_into * n_hat

        # 5. Visual attitude from thrust vector & velocity yaw
        if v_xy > 0.4:
            desired_yaw = math.atan2(self.vel[1], self.vel[0])
            yaw_err = (desired_yaw - self.rot_rpy[2] + math.pi) % (2.0 * math.pi) - math.pi
            self.rot_rpy[2] += float(np.clip(yaw_err, -2.5 * dt, 2.5 * dt))

        t_norm = float(np.linalg.norm(thrust))
        if t_norm > 1e-6:
            tz = thrust / t_norm
            cy = math.cos(self.rot_rpy[2])
            sy = math.sin(self.rot_rpy[2])
            # Rotate thrust unit vector into yaw-aligned body frame
            tx_b = cy * tz[0] + sy * tz[1]
            ty_b = -sy * tz[0] + cy * tz[1]
            tz_b = max(tz[2], 1e-3)
            pitch = math.atan2(tx_b, tz_b)
            roll = math.atan2(-ty_b, math.hypot(tx_b, tz_b))
            self.rot_rpy[0] = float(np.clip(roll, -self.MAX_TILT_RAD, self.MAX_TILT_RAD))
            self.rot_rpy[1] = float(np.clip(pitch, -self.MAX_TILT_RAD, self.MAX_TILT_RAD))

        # 6. Battery drain
        power = self.P_BASE + self.K_T * (t_norm ** 1.5) + self.K_V * (v_rel_norm ** 2)
        self.battery_pct = max(0.0, self.battery_pct - 100.0 * (power * dt) / self.BATTERY_CAPACITY_JOULES)
        if self.battery_pct <= 0.0:
            self.status = "depleted"

    @property
    def pos_tuple(self) -> Vec3:
        return (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]))

    @property
    def vel_tuple(self) -> Vec3:
        return (float(self.vel[0]), float(self.vel[1]), float(self.vel[2]))

    @property
    def rot_rpy_tuple(self) -> Vec3:
        return (float(self.rot_rpy[0]), float(self.rot_rpy[1]), float(self.rot_rpy[2]))

    @property
    def target_pos_tuple(self) -> Vec3:
        return (float(self.target_pos[0]), float(self.target_pos[1]), float(self.target_pos[2]))
