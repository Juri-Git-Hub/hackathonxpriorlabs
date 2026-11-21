"""MuJoCo-grade 6-DOF rigid-body quadrotor dynamics with 4-motor ESC lag,
Lee et al. geometric SE(3) attitude controller, rotor gyroscopic precession,
anisotropic body + blade-flapping drag, Cheeseman-Bennett ground effect,
soft-penalty contact physics, and physical power/battery model.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from shared.schemas import Vec3
from sim.world import World

DroneStatus = Literal["searching", "tracking", "locked", "depleted"]


class Drone:
    """Full 6-DOF rigid-body quadrotor (X-frame, 200 Hz internal sub-stepping)."""

    # Rigid-body & geometric parameters (1.5 kg industrial inspection quadrotor)
    MASS = 1.5                # kg
    GRAVITY = 9.81            # m/s^2
    ARM_LENGTH = 0.225        # m (center-to-rotor, 450mm wheelbase)
    L_EFF = ARM_LENGTH / math.sqrt(2.0)
    I_XX = 0.022              # kg*m^2
    I_YY = 0.022              # kg*m^2
    I_ZZ = 0.041              # kg*m^2
    J_ROTOR = 6.0e-5          # kg*m^2 (per rotor+propeller for gyroscopic precession)
    ROTOR_RADIUS = 0.127      # m (10-inch propellers, for ground effect)

    # Brushless motor & propeller aerodynamics
    K_F = 1.4e-5              # N / (rad/s)^2  (hover ~512.5 rad/s = 4895 RPM)
    K_M = 2.4e-7              # N*m / (rad/s)^2
    C_M = K_M / K_F           # torque-to-thrust ratio (m)
    TAU_MOTOR = 0.035         # s (35 ms ESC + motor 1st-order lag)
    OMEGA_MIN = 80.0          # rad/s (armed idle)
    OMEGA_MAX = 960.0         # rad/s (~9170 RPM, ~3.5x TWR)
    OMEGA_HOVER = math.sqrt((MASS * GRAVITY) / (4.0 * K_F))

    # Aerodynamic drag & rotational damping
    K_DRAG = 0.035            # nominal scalar drag reference (kept for compatibility)
    C_DRAG_BX = 0.042         # body-X quadratic parasitic drag (N/(m/s)^2)
    C_DRAG_BY = 0.042         # body-Y quadratic parasitic drag
    C_DRAG_BZ = 0.095         # body-Z quadratic parasitic drag (larger planform)
    K_ROTOR_H = 1.4e-4        # rotor blade-flapping / H-force linear drag (N*s/(m*rad/s))
    C_OMEGA_XY = 0.008        # rotational aerodynamic damping (N*m*s/rad)
    C_OMEGA_Z = 0.012

    # Flight envelope & safety clearances
    MAX_SPEED_XY = 8.0        # m/s
    MAX_SPEED_Z = 3.0         # m/s
    MAX_TILT_RAD = math.radians(28.0)
    MIN_ALTITUDE = 1.5        # m
    BUILDING_CLEARANCE = 2.0  # m
    SUBSTEPS = 8              # 8x sub-stepping => 160 Hz rigid-body integration at 20 Hz tick

    # Geometric SE(3) attitude controller gains (critically damped for J)
    K_R_XY = 1.65
    K_R_Z = 0.65
    K_W_XY = 0.28
    K_W_Z = 0.16

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

        # Full 6-DOF internal state
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # [w, x, y, z]
        self.omega = np.zeros(3, dtype=np.float64)                     # body [p, q, r] rad/s
        self.rotor_speeds = np.full(4, self.OMEGA_HOVER, dtype=np.float64)
        self.accel_world = np.zeros(3, dtype=np.float64)
        self._int_err = np.zeros(3, dtype=np.float64)
        self._desired_yaw: float = 0.0

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
        self.quat[:] = (1.0, 0.0, 0.0, 0.0)
        self.omega[:] = 0.0
        self.rotor_speeds[:] = self.OMEGA_HOVER
        self.accel_world[:] = 0.0
        self._int_err[:] = 0.0
        self._desired_yaw = 0.0
        self.battery_pct = 100.0
        self.samples_taken = 0
        self.last_ppm = 0.0
        self.status = "searching"

    def set_target(self, target_pos: Vec3) -> None:
        self.target_pos[0] = float(np.clip(target_pos[0], -58.0, 58.0))
        self.target_pos[1] = float(np.clip(target_pos[1], -58.0, 58.0))
        self.target_pos[2] = float(np.clip(target_pos[2], self.MIN_ALTITUDE, 28.0))

    def _repulsive_and_contact(
        self,
        px: float,
        py: float,
        pz: float,
        vx: float,
        vy: float,
        vz: float,
        tx: float,
        ty: float,
        world: World,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute smooth avoidance acceleration (ax, ay, az) + MuJoCo-style contact force (fx, fy, fz)."""
        ax = 0.0
        ay = 0.0
        az = 0.0
        fx = 0.0
        fy = 0.0
        fz = 0.0

        influence = self.BUILDING_CLEARANCE + 2.5
        to_tx = tx - px
        to_ty = ty - py

        for x0, y0, z0, x1, y1, z1, cx, cy, cz in world._bldg_bounds_scalar:
            clx = x0 if px < x0 else (x1 if px > x1 else px)
            cly = y0 if py < y0 else (y1 if py > y1 else py)
            clz = z0 if pz < z0 else (z1 if pz > z1 else pz)

            dx = px - clx
            dy = py - cly
            dz = pz - clz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < influence:
                if dist < 1e-4:
                    awx = px - cx
                    awy = py - cy
                    awz = pz - cz
                    naw = math.sqrt(awx * awx + awy * awy + awz * awz)
                    if naw > 1e-4:
                        nx, ny, nz = awx / naw, awy / naw, awz / naw
                    else:
                        nx, ny, nz = 0.0, 0.0, 1.0
                    mag = 22.0
                else:
                    nx, ny, nz = dx / dist, dy / dist, dz / dist
                    pen = influence - dist
                    mag = 5.5 * (pen / max(dist - self.BUILDING_CLEARANCE * 0.5, 0.35))
                    if mag > 22.0:
                        mag = 22.0

                ax += nx * mag
                ay += ny * mag
                az += nz * mag

                # Tangential slide around vertical walls to avoid head-on local minima
                hnorm = math.hypot(nx, ny)
                if hnorm > 0.3:
                    t1x = -ny / hnorm
                    t1y = nx / hnorm
                    if t1x * to_tx + t1y * to_ty < 0.0:
                        t1x = -t1x
                        t1y = -t1y
                    ax += t1x * (0.85 * mag)
                    ay += t1y * (0.85 * mag)
                    if pz < z1 + 1.5:
                        az += 0.45 * mag

                # Soft-penalty spring-damper contact force inside safety clearance buffer
                if dist < self.BUILDING_CLEARANCE:
                    overlap = self.BUILDING_CLEARANCE - dist
                    vn = vx * nx + vy * ny + vz * nz
                    # Stiff spring + viscous damper (MuJoCo-style contact impedance)
                    fn = 120.0 * overlap - 18.0 * min(0.0, vn)
                    fx += fn * nx
                    fy += fn * ny
                    fz += fn * nz

        # Ground repulsion below MIN_ALTITUDE + 0.5
        if pz < self.MIN_ALTITUDE + 0.5:
            az += 6.0 * (self.MIN_ALTITUDE + 0.5 - pz)

        return ax, ay, az, fx, fy, fz

    def step(self, dt: float, wind_local: np.ndarray, world: World) -> None:
        """Advance 6-DOF rigid-body quadrotor dynamics by dt seconds (200 Hz sub-stepped)."""
        wx = float(wind_local[0])
        wy = float(wind_local[1])
        wz = float(wind_local[2])

        px = float(self.pos[0])
        py = float(self.pos[1])
        pz = float(self.pos[2])
        vx = float(self.vel[0])
        vy = float(self.vel[1])
        vz = float(self.vel[2])

        qw = float(self.quat[0])
        qx = float(self.quat[1])
        qy = float(self.quat[2])
        qz = float(self.quat[3])

        ox = float(self.omega[0])
        oy = float(self.omega[1])
        oz = float(self.omega[2])

        w0 = float(self.rotor_speeds[0])
        w1 = float(self.rotor_speeds[1])
        w2 = float(self.rotor_speeds[2])
        w3 = float(self.rotor_speeds[3])

        tx = float(self.target_pos[0])
        ty = float(self.target_pos[1])
        tz = float(self.target_pos[2])

        ie_x = float(self._int_err[0])
        ie_y = float(self._int_err[1])
        ie_z = float(self._int_err[2])

        depleted = self.battery_pct <= 0.0
        if depleted:
            self.battery_pct = 0.0
            self.status = "depleted"

        sub_dt = dt / self.SUBSTEPS
        mass = self.MASS
        inv_mass = 1.0 / mass
        g = self.GRAVITY
        k_f = self.K_F
        c_m = self.C_M
        l_eff = self.L_EFF
        inv_ixx = 1.0 / self.I_XX
        inv_iyy = 1.0 / self.I_YY
        inv_izz = 1.0 / self.I_ZZ
        motor_alpha = min(1.0, sub_dt / self.TAU_MOTOR)
        tan_max_tilt = math.tan(self.MAX_TILT_RAD)
        om_min_sq = self.OMEGA_MIN * self.OMEGA_MIN
        om_max_sq = self.OMEGA_MAX * self.OMEGA_MAX

        # Update desired yaw smoothly toward horizontal velocity when moving
        v_xy_tick = math.hypot(vx, vy)
        if not depleted and v_xy_tick > 0.45:
            tgt_yaw = math.atan2(vy, vx)
            yaw_diff = (tgt_yaw - self._desired_yaw + math.pi) % (2.0 * math.pi) - math.pi
            max_dyaw = 2.5 * dt
            if yaw_diff > max_dyaw:
                yaw_diff = max_dyaw
            elif yaw_diff < -max_dyaw:
                yaw_diff = -max_dyaw
            self._desired_yaw = (self._desired_yaw + yaw_diff + math.pi) % (2.0 * math.pi) - math.pi

        cos_yd = math.cos(self._desired_yaw)
        sin_yd = math.sin(self._desired_yaw)

        energy_j = 0.0
        last_ax = 0.0
        last_ay = 0.0
        last_az = 0.0

        for _ in range(self.SUBSTEPS):
            # 1) Body-to-world rotation matrix R from unit quaternion (qw, qx, qy, qz)
            xx = qx * qx
            yy = qy * qy
            zz = qz * qz
            xy = qx * qy
            xz = qx * qz
            yz = qy * qz
            wxq = qw * qx
            wyq = qw * qy
            wzq = qw * qz

            r00 = 1.0 - 2.0 * (yy + zz)
            r01 = 2.0 * (xy - wzq)
            r02 = 2.0 * (xz + wyq)
            r10 = 2.0 * (xy + wzq)
            r11 = 1.0 - 2.0 * (xx + zz)
            r12 = 2.0 * (yz - wxq)
            r20 = 2.0 * (xz - wyq)
            r21 = 2.0 * (yz + wxq)
            r22 = 1.0 - 2.0 * (xx + yy)

            # 2) Obstacle avoidance + soft-penalty contact forces
            rep_ax, rep_ay, rep_az, c_fx, c_fy, c_fz = self._repulsive_and_contact(
                px, py, pz, vx, vy, vz, tx, ty, world
            )

            # 3) Control law -> desired motor speeds (w0_cmd..w3_cmd)
            if depleted:
                # Controlled power-off spin-down & gentle descent
                w0_cmd = w1_cmd = w2_cmd = w3_cmd = 0.0
            else:
                ex = tx - px
                ey = ty - py
                ez = tz - pz

                # Outer position-to-velocity PD
                vdx = 1.25 * ex
                vdy = 1.25 * ey
                vdz = 1.35 * ez

                vd_xy = math.hypot(vdx, vdy)
                if vd_xy > self.MAX_SPEED_XY:
                    sc = self.MAX_SPEED_XY / vd_xy
                    vdx *= sc
                    vdy *= sc
                if vdz > self.MAX_SPEED_Z:
                    vdz = self.MAX_SPEED_Z
                elif vdz < -self.MAX_SPEED_Z:
                    vdz = -self.MAX_SPEED_Z

                # Integral wind-drag rejection (clamped anti-windup)
                ie_x = max(-2.5, min(2.5, ie_x + 0.35 * (vdx - vx) * sub_dt))
                ie_y = max(-2.5, min(2.5, ie_y + 0.35 * (vdy - vy) * sub_dt))
                ie_z = max(-2.0, min(2.0, ie_z + 0.40 * (vdz - vz) * sub_dt))

                ac_x = 2.8 * (vdx - vx) + ie_x + rep_ax
                ac_y = 2.8 * (vdy - vy) + ie_y + rep_ay
                ac_z = 3.2 * (vdz - vz) + ie_z + rep_az

                # Desired world thrust vector F_des = m * (a_cmd + g)
                f_des_x = mass * ac_x
                f_des_y = mass * ac_y
                f_des_z = mass * (ac_z + g)

                min_fz = 0.25 * mass * g
                if f_des_z < min_fz:
                    f_des_z = min_fz

                f_xy = math.hypot(f_des_x, f_des_y)
                max_f_xy = f_des_z * tan_max_tilt
                if f_xy > max_f_xy and f_xy > 1e-6:
                    sc = max_f_xy / f_xy
                    f_des_x *= sc
                    f_des_y *= sc

                # Construct desired rotation matrix R_d = [b1d, b2d, b3d]
                f_norm = math.sqrt(f_des_x * f_des_x + f_des_y * f_des_y + f_des_z * f_des_z)
                b3d_x = f_des_x / f_norm
                b3d_y = f_des_y / f_norm
                b3d_z = f_des_z / f_norm

                # b2d = normalize(b3d x b1c), where b1c = [cos(yaw_d), sin(yaw_d), 0]
                b2d_x = -b3d_z * sin_yd
                b2d_y = b3d_z * cos_yd
                b2d_z = b3d_x * sin_yd - b3d_y * cos_yd
                b2_norm = math.sqrt(b2d_x * b2d_x + b2d_y * b2d_y + b2d_z * b2d_z)
                if b2_norm > 1e-6:
                    inv_b2 = 1.0 / b2_norm
                    b2d_x *= inv_b2
                    b2d_y *= inv_b2
                    b2d_z *= inv_b2
                else:
                    b2d_x, b2d_y, b2d_z = -sin_yd, cos_yd, 0.0

                # b1d = b2d x b3d
                b1d_x = b2d_y * b3d_z - b2d_z * b3d_y
                b1d_y = b2d_z * b3d_x - b2d_x * b3d_z
                b1d_z = b2d_x * b3d_y - b2d_y * b3d_x

                # E = R_d^T * R  (only need off-diagonals for vee(0.5 * (E - E^T)))
                e01 = b1d_x * r01 + b1d_y * r11 + b1d_z * r21
                e02 = b1d_x * r02 + b1d_y * r12 + b1d_z * r22
                e10 = b2d_x * r00 + b2d_y * r10 + b2d_z * r20
                e12 = b2d_x * r02 + b2d_y * r12 + b2d_z * r22
                e20 = b3d_x * r00 + b3d_y * r10 + b3d_z * r20
                e21 = b3d_x * r01 + b3d_y * r11 + b3d_z * r21

                er_x = 0.5 * (e21 - e12)
                er_y = 0.5 * (e02 - e20)
                er_z = 0.5 * (e10 - e01)

                # Geometric SE(3) torque command with gyroscopic feedforward
                tau_cx = -self.K_R_XY * er_x - self.K_W_XY * ox + (self.I_ZZ - self.I_YY) * oy * oz
                tau_cy = -self.K_R_XY * er_y - self.K_W_XY * oy + (self.I_XX - self.I_ZZ) * ox * oz
                tau_cz = -self.K_R_Z * er_z - self.K_W_Z * oz + (self.I_YY - self.I_XX) * ox * oy

                # Collective thrust projected onto actual body +Z axis
                t_cmd = f_des_x * r02 + f_des_y * r12 + f_des_z * r22
                if t_cmd < min_fz:
                    t_cmd = min_fz

                # Exact orthogonal X-quad control allocation -> rotor squared speeds
                tx_l = tau_cx / l_eff
                ty_l = tau_cy / l_eff
                tz_c = tau_cz / c_m

                f0_c = 0.25 * (t_cmd + tx_l - ty_l + tz_c)
                f1_c = 0.25 * (t_cmd - tx_l - ty_l - tz_c)
                f2_c = 0.25 * (t_cmd - tx_l + ty_l + tz_c)
                f3_c = 0.25 * (t_cmd + tx_l + ty_l - tz_c)

                inv_kf = 1.0 / k_f
                w0_cmd = math.sqrt(max(om_min_sq, min(om_max_sq, f0_c * inv_kf)))
                w1_cmd = math.sqrt(max(om_min_sq, min(om_max_sq, f1_c * inv_kf)))
                w2_cmd = math.sqrt(max(om_min_sq, min(om_max_sq, f2_c * inv_kf)))
                w3_cmd = math.sqrt(max(om_min_sq, min(om_max_sq, f3_c * inv_kf)))

            # 4) 1st-order brushless motor + ESC lag
            w0 += motor_alpha * (w0_cmd - w0)
            w1 += motor_alpha * (w1_cmd - w1)
            w2 += motor_alpha * (w2_cmd - w2)
            w3 += motor_alpha * (w3_cmd - w3)

            f0 = k_f * w0 * w0
            f1 = k_f * w1 * w1
            f2 = k_f * w2 * w2
            f3 = k_f * w3 * w3

            t_raw = f0 + f1 + f2 + f3
            tau_mx = l_eff * (f0 - f1 - f2 + f3)
            tau_my = l_eff * (-f0 - f1 + f2 + f3)
            tau_mz = c_m * (f0 - f1 + f2 - f3)

            # 5) Cheeseman-Bennett Ground Effect (near ground or building rooftops)
            h_agl = world.height_above_surface(px, py, pz)
            ge_factor = 1.0
            if h_agl < 4.0 * self.ROTOR_RADIUS:
                ratio = self.ROTOR_RADIUS / (4.0 * max(h_agl, 0.15))
                denom = max(0.80, 1.0 - ratio * ratio)
                ge_factor = min(1.25, 1.0 / denom)
            t_eff = t_raw * ge_factor

            # 6) Aerodynamic forces: anisotropic body drag + rotor blade-flapping H-force
            vrx = vx - wx
            vry = vy - wy
            vrz = vz - wz
            vrel_norm = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)

            # Relative velocity in body frame: v_b = R^T * v_rel
            vbx = r00 * vrx + r10 * vry + r20 * vrz
            vby = r01 * vrx + r11 * vry + r21 * vrz
            vbz = r02 * vrx + r12 * vry + r22 * vrz

            omega_sum = w0 + w1 + w2 + w3
            # Body-frame drag force
            fb_x = -self.C_DRAG_BX * abs(vbx) * vbx - self.K_ROTOR_H * omega_sum * vbx
            fb_y = -self.C_DRAG_BY * abs(vby) * vby - self.K_ROTOR_H * omega_sum * vby
            fb_z = -self.C_DRAG_BZ * abs(vbz) * vbz

            # Rotate drag back to world frame: F_drag = R * F_b
            fd_x = r00 * fb_x + r01 * fb_y + r02 * fb_z
            fd_y = r10 * fb_x + r11 * fb_y + r12 * fb_z
            fd_z = r20 * fb_x + r21 * fb_y + r22 * fb_z

            # 7) Rotational dynamics: motor torque + aero damping + gyroscopic precession + wind torque
            # Rotor net angular momentum along body Z: J_r * (-w0 + w1 - w2 + w3)
            h_rot_z = self.J_ROTOR * (-w0 + w1 - w2 + w3)
            # Subtle wind-gust pitching/rolling disturbance from relative airspeed
            tau_wx = 0.0022 * vby * abs(vby)
            tau_wy = -0.0022 * vbx * abs(vbx)

            # Euler's rigid-body rotation equation: J * d(omega)/dt = tau - omega x (J*omega + [0,0,h_rot_z])
            net_tx = tau_mx - self.C_OMEGA_XY * ox + tau_wx - (self.I_ZZ - self.I_YY) * oy * oz - oy * h_rot_z
            net_ty = tau_my - self.C_OMEGA_XY * oy + tau_wy - (self.I_XX - self.I_ZZ) * ox * oz + ox * h_rot_z
            net_tz = tau_mz - self.C_OMEGA_Z * oz - (self.I_YY - self.I_XX) * ox * oy

            ox += (net_tx * inv_ixx) * sub_dt
            oy += (net_ty * inv_iyy) * sub_dt
            oz += (net_tz * inv_izz) * sub_dt

            # Clamp extreme angular rates for numerical safety
            ox = max(-12.0, min(12.0, ox))
            oy = max(-12.0, min(12.0, oy))
            oz = max(-8.0, min(8.0, oz))

            # Integrate unit quaternion: dq/dt = 0.5 * q * [0, ox, oy, oz]
            dqw = 0.5 * (-qx * ox - qy * oy - qz * oz)
            dqx = 0.5 * (qw * ox + qy * oz - qz * oy)
            dqy = 0.5 * (qw * oy - qx * oz + qz * ox)
            dqz = 0.5 * (qw * oz + qx * oy - qy * ox)

            qw += dqw * sub_dt
            qx += dqx * sub_dt
            qy += dqy * sub_dt
            qz += dqz * sub_dt
            q_inv = 1.0 / math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            qw *= q_inv
            qx *= q_inv
            qy *= q_inv
            qz *= q_inv

            # 8) Translational Newton's law (Symplectic Euler)
            last_ax = (t_eff * r02 + fd_x + c_fx) * inv_mass
            last_ay = (t_eff * r12 + fd_y + c_fy) * inv_mass
            last_az = (t_eff * r22 + fd_z + c_fz) * inv_mass - g

            vx += last_ax * sub_dt
            vy += last_ay * sub_dt
            vz += last_az * sub_dt

            # Soft aerodynamic terminal cap for numerical stability
            vxy_now = math.hypot(vx, vy)
            max_vxy = self.MAX_SPEED_XY * 1.35
            if vxy_now > max_vxy:
                sc = max_vxy / vxy_now
                vx *= sc
                vy *= sc
            max_vz = self.MAX_SPEED_Z * 1.6
            if vz > max_vz:
                vz = max_vz
            elif vz < -max_vz:
                vz = -max_vz

            px += vx * sub_dt
            py += vy * sub_dt
            pz += vz * sub_dt

            # Floor landing contact at MIN_ALTITUDE
            if pz < self.MIN_ALTITUDE:
                pz = self.MIN_ALTITUDE
                if vz < 0.0:
                    vz = -0.15 * vz  # damped ground bounce
                if depleted:
                    vx *= 0.85
                    vy *= 0.85

            # Accumulate physical power draw
            if not depleted:
                power = self.P_BASE + self.K_T * (t_raw ** 1.5) + self.K_V * (vrel_norm * vrel_norm)
                energy_j += power * sub_dt

        # Hard facility bounds & safety clearance backstop
        px = max(world.bounds_min[0] + 1.5, min(world.bounds_max[0] - 1.5, px))
        py = max(world.bounds_min[1] + 1.5, min(world.bounds_max[1] - 1.5, py))
        pz = max(self.MIN_ALTITUDE, min(world.bounds_max[2] - 1.5, pz))

        self.pos[0] = px
        self.pos[1] = py
        self.pos[2] = pz
        self.vel[0] = vx
        self.vel[1] = vy
        self.vel[2] = vz
        self.quat[0] = qw
        self.quat[1] = qx
        self.quat[2] = qy
        self.quat[3] = qz
        self.omega[0] = ox
        self.omega[1] = oy
        self.omega[2] = oz
        self.rotor_speeds[0] = w0
        self.rotor_speeds[1] = w1
        self.rotor_speeds[2] = w2
        self.rotor_speeds[3] = w3
        self.accel_world[0] = last_ax
        self.accel_world[1] = last_ay
        self.accel_world[2] = last_az
        self._int_err[0] = ie_x
        self._int_err[1] = ie_y
        self._int_err[2] = ie_z

        if world.is_inside_building(self.pos_tuple, margin=self.BUILDING_CLEARANCE):
            pushed = world.push_outside_buildings(self.pos[None, :], margin=self.BUILDING_CLEARANCE)[0]
            normal = pushed - self.pos
            n_norm = float(np.linalg.norm(normal))
            self.pos = pushed
            if n_norm > 1e-6:
                n_hat = normal / n_norm
                v_into = float(np.dot(self.vel, n_hat))
                if v_into < 0.0:
                    # Restitution 0.2 bounce + tangential sliding
                    self.vel -= 1.2 * v_into * n_hat

        # Extract true physical ZYX Euler angles (roll, pitch, yaw) from quaternion
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.rot_rpy[0] = roll
        self.rot_rpy[1] = pitch
        self.rot_rpy[2] = yaw

        # Battery drain update
        if not depleted:
            self.battery_pct = max(0.0, self.battery_pct - 100.0 * energy_j / self.BATTERY_CAPACITY_JOULES)
            if self.battery_pct <= 0.0:
                self.status = "depleted"

    @property
    def tilt_rad(self) -> float:
        """Total tilt angle of body +Z axis from vertical (radians)."""
        qw, qx, qy, qz = self.quat
        r22 = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
        return math.acos(r22)

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
