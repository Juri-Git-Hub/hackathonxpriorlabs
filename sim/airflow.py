"""Mass-consistent 3D diagnostic wind solver (QUIC-URB style) + 3D divergence-free eddies.

Architecture:
- Staggered MAC 3D grid (48 x 48 x 12, 2.5 m voxels) over the 120 x 120 x 30 m refinery
- Exact solid masks for rectangular buildings and cylindrical storage tanks (zero normal flow)
- Sub-grid porous drag & wake turbulence for open-truss pipe rack and standalone pipe runs
- Empirical ABL power-law shear + Röckle/QUIC-URB lee-side recirculation cavities & far wakes
- Fast vectorized Red-Black SOR 3D Poisson pressure projection enforcing div(u) = 0
  (cached per base wind direction; ~20-35 ms on wind-angle change, 0 ms on speed change)
- Fast vectorized trilinear interpolation + strictly divergence-free 3D Fourier eddies
  + cylindrical Strouhal vortex shedding + Ornstein-Uhlenbeck temporal gusts
"""

from __future__ import annotations

import math

import numpy as np

from shared.schemas import Vec2, WindFieldPayload
from sim.world import World


class Airflow:
    """Mass-consistent 3D airflow with ABL shear, Poisson obstacle flow, and 3D turbulence."""

    # Atmospheric Boundary Layer (ABL) industrial power-law parameters
    Z_REF: float = 10.0
    ABL_ALPHA: float = 0.18
    N_EDDY_MODES: int = 6

    # 3D MAC grid resolution (120 x 120 x 30 m -> 2.5 m cubic voxels)
    NX: int = 48
    NY: int = 48
    NZ: int = 12
    SOR_ITERS: int = 75
    SOR_OMEGA: float = 1.72

    def __init__(
        self,
        world: World,
        base_wind_angle_deg: float = 15.0,
        base_wind_speed: float = 3.5,
        tau_w: float = 6.0,
        sigma_w: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.world = world
        self.tau_w = tau_w
        self.sigma_w = sigma_w
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        self.w0 = np.zeros(2, dtype=np.float64)
        self.w_turb = np.zeros(2, dtype=np.float64)
        self._time: float = 0.0
        self._eddy_offset = np.zeros(2, dtype=np.float64)

        # Grid geometry
        self._x_min = float(world.bounds_min[0])
        self._y_min = float(world.bounds_min[1])
        self._z_min = float(world.bounds_min[2])
        self._x_max = float(world.bounds_max[0])
        self._y_max = float(world.bounds_max[1])
        self._z_max = float(world.bounds_max[2])
        self._dx = (self._x_max - self._x_min) / self.NX
        self._dy = (self._y_max - self._y_min) / self.NY
        self._dz = (self._z_max - self._z_min) / self.NZ
        self._inv_dx = 1.0 / self._dx
        self._inv_dy = 1.0 / self._dy
        self._inv_dz = 1.0 / self._dz

        # Precompute static solid / porous / pipe masks and SOR checkerboards
        self._build_static_grid()

        # Cached unit-speed solved MAC velocity fields + cell-center turbulence field
        self._solved_angle_rad: float | None = None
        self._u_unit = np.zeros((self.NX + 1, self.NY, self.NZ), dtype=np.float64)
        self._v_unit = np.zeros((self.NX, self.NY + 1, self.NZ), dtype=np.float64)
        self._w_unit = np.zeros((self.NX, self.NY, self.NZ + 1), dtype=np.float64)
        self._u_samp = self._u_unit
        self._v_samp = self._v_unit
        self._w_samp = self._w_unit
        self._turb_grid = np.ones((self.NX, self.NY, self.NZ), dtype=np.float64)
        self._field_version: int = 0

        # Spatial 3D divergence-free Fourier eddy modes (advected via integrated wind offset)
        self._k_vecs = np.zeros((self.N_EDDY_MODES, 3), dtype=np.float64)
        self._e_vecs = np.zeros((self.N_EDDY_MODES, 3), dtype=np.float64)
        self._phases = np.zeros(self.N_EDDY_MODES, dtype=np.float64)
        self._eddy_weights = np.zeros(self.N_EDDY_MODES, dtype=np.float64)
        self._init_eddy_modes()

        self.set_wind(base_wind_angle_deg, base_wind_speed)

    def _build_static_grid(self) -> None:
        """Precompute solid masks, open MAC faces, porous pipe drag, and SOR stencils."""
        nx, ny, nz = self.NX, self.NY, self.NZ
        dx, dy, dz = self._dx, self._dy, self._dz

        self._xc = self._x_min + (np.arange(nx, dtype=np.float64) + 0.5) * dx
        self._yc = self._y_min + (np.arange(ny, dtype=np.float64) + 0.5) * dy
        self._zc = self._z_min + (np.arange(nz, dtype=np.float64) + 0.5) * dz

        self._xf = self._x_min + np.arange(nx + 1, dtype=np.float64) * dx
        self._yf = self._y_min + np.arange(ny + 1, dtype=np.float64) * dy
        self._zf = self._z_min + np.arange(nz + 1, dtype=np.float64) * dz

        X, Y, Z = np.meshgrid(self._xc, self._yc, self._zc, indexing="ij")
        self._grid_X = X
        self._grid_Y = Y
        self._grid_Z = Z
        self._abl_zc = self._abl_factor(self._zc)

        solid = np.zeros((nx, ny, nz), dtype=bool)
        porous_drag = np.zeros((nx, ny, nz), dtype=np.float64)
        static_turb = np.zeros((nx, ny, nz), dtype=np.float64)

        # 1) Buildings, cylindrical storage tanks, and open-truss pipe rack
        for i in range(len(self.world.buildings)):
            bmin = self.world._bldg_min[i]
            bmax = self.world._bldg_max[i]
            if self.world._bldg_is_porous[i]:
                in_pr = (
                    (X >= bmin[0])
                    & (X <= bmax[0])
                    & (Y >= bmin[1])
                    & (Y <= bmax[1])
                    & (Z >= bmin[2])
                    & (Z <= bmax[2])
                )
                porous_drag = np.maximum(porous_drag, np.where(in_pr, 0.34, 0.0))
                static_turb = np.maximum(static_turb, np.where(in_pr, 0.55, 0.0))
            elif self.world._bldg_is_cylinder[i]:
                cx = float(self.world._bldg_center[i, 0])
                cy = float(self.world._bldg_center[i, 1])
                r_cyl = float(self.world._bldg_cyl_radius[i])
                in_cyl = (
                    ((X - cx) ** 2 + (Y - cy) ** 2 <= r_cyl * r_cyl)
                    & (Z >= bmin[2])
                    & (Z <= bmax[2])
                )
                solid |= in_cyl
            else:
                in_box = (
                    (X >= bmin[0])
                    & (X <= bmax[0])
                    & (Y >= bmin[1])
                    & (Y <= bmax[1])
                    & (Z >= bmin[2])
                    & (Z <= bmax[2])
                )
                solid |= in_box

        # 2) Sub-grid fractional porous drag & localized turbulence for standalone pipe runs
        if self.world._pipe_seg_a.shape[0] > 0:
            # Only visit cells close enough in height to intersect a pipe's influence.
            pipe_influence_top = self.world._pipe_top + 0.5 * dx
            kz_max = int(min(nz, math.ceil((pipe_influence_top - self._z_min) * self._inv_dz)))
            pts_low = np.stack(
                [X[:, :, :kz_max], Y[:, :, :kz_max], Z[:, :, :kz_max]], axis=-1
            ).reshape(-1, 3)

            pipe_drag_flat = np.zeros(pts_low.shape[0], dtype=np.float64)
            pipe_turb_flat = np.zeros(pts_low.shape[0], dtype=np.float64)
            half_cell = 0.5 * dx
            for s in range(self.world._pipe_seg_a.shape[0]):
                a = self.world._pipe_seg_a[s]
                ab = self.world._pipe_seg_ab[s]
                len2 = float(self.world._pipe_seg_len2[s])
                r_p = float(self.world._pipe_seg_r[s])

                ap = pts_low - a[None, :]
                t = np.clip((ap @ ab) / len2, 0.0, 1.0)
                closest = a[None, :] + t[:, None] * ab[None, :]
                dist = np.linalg.norm(pts_low - closest, axis=1)

                influence = half_cell + r_p
                hit = dist < influence
                if np.any(hit):
                    # Fractional blockage proportional to pipe diameter / voxel size
                    w_frac = np.clip(1.0 - dist[hit] / influence, 0.0, 1.0)
                    area_ratio = min(0.32, (2.0 * r_p) / dx)
                    pipe_drag_flat[hit] = np.maximum(pipe_drag_flat[hit], area_ratio * (0.45 + 0.55 * w_frac))
                    pipe_turb_flat[hit] = np.maximum(pipe_turb_flat[hit], 0.42 * w_frac)

            pipe_drag_3d = pipe_drag_flat.reshape(nx, ny, kz_max)
            pipe_turb_3d = pipe_turb_flat.reshape(nx, ny, kz_max)
            porous_drag[:, :, :kz_max] = np.maximum(porous_drag[:, :, :kz_max], pipe_drag_3d)
            static_turb[:, :, :kz_max] = np.maximum(static_turb[:, :, :kz_max], pipe_turb_3d)

        porous_drag[solid] = 0.0
        static_turb[solid] = 0.0

        self._solid = solid
        self._fluid = ~solid
        self._porous_drag = porous_drag
        self._static_turb = static_turb

        # MAC face openness (False if touching a solid cell or impermeable ground z=0)
        u_open = np.ones((nx + 1, ny, nz), dtype=bool)
        u_open[0, :, :] = ~solid[0, :, :]
        u_open[nx, :, :] = ~solid[nx - 1, :, :]
        u_open[1:nx, :, :] = (~solid[:-1, :, :]) & (~solid[1:, :, :])

        v_open = np.ones((nx, ny + 1, nz), dtype=bool)
        v_open[:, 0, :] = ~solid[:, 0, :]
        v_open[:, ny, :] = ~solid[:, ny - 1, :]
        v_open[:, 1:ny, :] = (~solid[:, :-1, :]) & (~solid[:, 1:, :])

        w_open = np.ones((nx, ny, nz + 1), dtype=bool)
        w_open[:, :, 0] = False  # Impermeable ground plane z = 0
        w_open[:, :, nz] = False  # Free-slip top lid z = 30 m (conserves horizontal flux)
        w_open[:, :, 1:nz] = (~solid[:, :, :-1]) & (~solid[:, :, 1:])

        self._u_open = u_open
        self._v_open = v_open
        self._w_open = w_open

        # Precompute open-face count per fluid cell for 3D Poisson stencil
        n_open = (
            u_open[:-1, :, :].astype(np.float64)
            + u_open[1:, :, :].astype(np.float64)
            + v_open[:, :-1, :].astype(np.float64)
            + v_open[:, 1:, :].astype(np.float64)
            + w_open[:, :, :-1].astype(np.float64)
            + w_open[:, :, 1:].astype(np.float64)
        )
        self._inv_n_open = np.where(self._fluid & (n_open > 0.0), 1.0 / np.maximum(n_open, 1.0), 0.0)

        # Interior neighbor openness masks for fast vectorized Red-Black SOR
        self._xm_open = u_open[1:nx, :, :]
        self._ym_open = v_open[:, 1:ny, :]
        self._zm_open = w_open[:, :, 1:nz]

        # 3D Red-Black checkerboard masks (only fluid cells updated)
        ii, jj, kk = np.indices((nx, ny, nz))
        parity = (ii + jj + kk) & 1
        self._red_mask = (parity == 0) & self._fluid
        self._black_mask = (parity == 1) & self._fluid

    def _solve_unit_wind_field(self, ux: float, uy: float) -> None:
        """Build QUIC-URB empirical wakes + run 3D MAC Poisson projection for unit wind (ux, uy)."""
        nx, ny, nz = self.NX, self.NY, self.NZ
        h = self._dx  # Uniform cubic voxels (dx == dy == dz == 2.5 m)
        px_u = -uy
        py_u = ux
        abs_ux = abs(ux)
        abs_uy = abs(uy)

        X, Y, Z = self._grid_X, self._grid_Y, self._grid_Z
        abl_3d = self._abl_zc[None, None, :]

        # 1) Cell-centered empirical initial velocity (ABL shear + porous drag + lee wakes)
        uc = np.broadcast_to(ux * abl_3d, (nx, ny, nz)).copy()
        vc = np.broadcast_to(uy * abl_3d, (nx, ny, nz)).copy()
        wc = np.zeros((nx, ny, nz), dtype=np.float64)
        turb = 1.0 + self._static_turb.copy()

        # Apply sub-grid porous drag (pipe rack + standalone pipes)
        retain = 1.0 - self._porous_drag
        uc *= retain
        vc *= retain

        # Empirical Röckle / QUIC-URB lee-side recirculation cavity & far-wake deficit
        for i in range(len(self.world.buildings)):
            cx = float(self.world._bldg_center[i, 0])
            cy = float(self.world._bldg_center[i, 1])
            hx = float(self.world._bldg_half[i, 0])
            hy = float(self.world._bldg_half[i, 1])
            height = float(self.world._bldg_max[i, 2])
            is_cyl = bool(self.world._bldg_is_cylinder[i])
            is_porous = bool(self.world._bldg_is_porous[i])

            if is_cyl:
                half_along = float(self.world._bldg_cyl_radius[i])
                half_cross = half_along
            else:
                half_along = hx * abs_ux + hy * abs_uy
                half_cross = hx * abs_uy + hy * abs_ux

            dx_b = X - cx
            dy_b = Y - cy
            along = dx_b * ux + dy_b * uy
            cross = dx_b * px_u + dy_b * py_u
            abs_cross = np.abs(cross)

            wake_len = 2.5 * height
            cavity_len = 1.05 * height
            wake_mask = (
                self._fluid
                & (along > half_along)
                & (along <= half_along + wake_len * 1.1)
                & (abs_cross <= half_cross * 1.05)
                & (Z <= height * 1.15)
            )
            if not np.any(wake_mask):
                continue

            a_w = np.maximum(0.0, along[wake_mask] - half_along)
            c_w = cross[wake_mask]
            z_w = Z[wake_mask]

            frac_along = np.clip(a_w / wake_len, 0.0, 1.2)
            env_cross = np.exp(
                -0.5
                * (
                    np.maximum(0.0, np.abs(c_w) - half_cross * 0.65)
                    / max(half_cross * 0.22, 0.8)
                )
                ** 2
            )
            env_z = 0.5 * (1.0 - np.tanh((z_w - height * 0.98) / 1.2))
            env_tail = 0.5 * (1.0 - np.tanh((frac_along - 0.90) / 0.14))
            wake_env = env_cross * env_z * env_tail

            if is_porous:
                # Porous pipe rack: mild wake deficit, no reverse-flow recirculation bubble
                core_factor = np.clip(0.68 + 0.32 * (frac_along * frac_along), 0.68, 1.0)
                eff_scale = 1.0 - wake_env * (1.0 - core_factor)
                uc[wake_mask] *= eff_scale
                vc[wake_mask] *= eff_scale
                boost = 0.65 * wake_env * np.sin(math.pi * np.clip(frac_along, 0.0, 1.0))
                turb[wake_mask] = np.maximum(turb[wake_mask], 1.0 + boost)
            else:
                # Solid building/tank: near-wake reverse recirculation (-0.22U) -> far-wake recovery
                frac_cav = a_w / max(cavity_len, 1.0)
                near_rev = -0.22 * np.sin(math.pi * np.clip(frac_cav, 0.0, 1.0))
                far_rec = np.clip(
                    0.18 + 0.82 * ((np.maximum(0.0, a_w - 0.70 * cavity_len)) / max(wake_len - 0.70 * cavity_len, 1.0)) ** 1.35,
                    0.18,
                    1.0,
                )
                target_along_mult = np.where(frac_cav < 0.75, near_rev, far_rec)
                eff_along = 1.0 - wake_env * (1.0 - target_along_mult)

                u_cur = uc[wake_mask]
                v_cur = vc[wake_mask]
                cur_along = u_cur * ux + v_cur * uy
                cur_cross = u_cur * px_u + v_cur * py_u
                new_along = cur_along * eff_along

                frac_c = np.clip(frac_along, 0.0, 1.0)
                swirl = -0.20 * wake_env * np.sin(math.pi * frac_c) * np.tanh(c_w / max(half_cross, 1.0))
                new_cross = cur_cross + swirl

                uc[wake_mask] = new_along * ux + new_cross * px_u
                vc[wake_mask] = new_along * uy + new_cross * py_u
                wc[wake_mask] -= 0.14 * wake_env * (1.0 - frac_c)

                boost = 1.65 * wake_env * np.sin(math.pi * frac_c)
                turb[wake_mask] = np.maximum(turb[wake_mask], 1.0 + boost)

        uc[self._solid] = 0.0
        vc[self._solid] = 0.0
        wc[self._solid] = 0.0

        # 2) Transfer cell-centered initial field to staggered MAC faces
        u = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        v = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        w = np.zeros((nx, ny, nz + 1), dtype=np.float64)

        u[0, :, :] = ux * abl_3d[0, 0, :][None, :]
        u[nx, :, :] = ux * abl_3d[0, 0, :][None, :]
        u[1:nx, :, :] = 0.5 * (uc[:-1, :, :] + uc[1:, :, :])

        v[:, 0, :] = uy * abl_3d[0, 0, :][None, :]
        v[:, ny, :] = uy * abl_3d[0, 0, :][None, :]
        v[:, 1:ny, :] = 0.5 * (vc[:, :-1, :] + vc[:, 1:, :])

        w[:, :, 1:nz] = 0.5 * (wc[:, :, :-1] + wc[:, :, 1:])
        w[:, :, nz] = wc[:, :, nz - 1]

        # Enforce exact zero normal flow at all solid walls & ground
        u[~self._u_open] = 0.0
        v[~self._v_open] = 0.0
        w[~self._w_open] = 0.0

        # Direction-aware domain boundary conditions:
        # Inflow faces: fixed Dirichlet velocity (Neumann dp/dn = 0) so ambient wind never slows down.
        # Outflow & top faces: open Dirichlet p = 0 so displaced air exits freely.
        u_p_open = self._u_open.copy()
        v_p_open = self._v_open.copy()
        if ux > 1e-6:
            u_p_open[0, :, :] = False
        elif ux < -1e-6:
            u_p_open[nx, :, :] = False
        if uy > 1e-6:
            v_p_open[:, 0, :] = False
        elif uy < -1e-6:
            v_p_open[:, ny, :] = False

        n_open_dir = (
            u_p_open[:-1, :, :].astype(np.float64)
            + u_p_open[1:, :, :].astype(np.float64)
            + v_p_open[:, :-1, :].astype(np.float64)
            + v_p_open[:, 1:, :].astype(np.float64)
            + self._w_open[:, :, :-1].astype(np.float64)
            + self._w_open[:, :, 1:].astype(np.float64)
        )
        inv_n = np.where(self._fluid & (n_open_dir > 0.0), 1.0 / np.maximum(n_open_dir, 1.0), 0.0)

        # 3) 3D Poisson pressure projection: div(u - grad(p)) = 0 via Red-Black SOR
        rhs = (u[1:, :, :] - u[:-1, :, :]) + (v[:, 1:, :] - v[:, :-1, :]) + (w[:, :, 1:] - w[:, :, :-1])
        rhs *= h
        rhs[self._solid] = 0.0

        p = np.zeros((nx, ny, nz), dtype=np.float64)
        omega = self.SOR_OMEGA
        xm = self._xm_open
        ym = self._ym_open
        zm = self._zm_open
        red = self._red_mask
        black = self._black_mask

        sum_nbr = np.zeros_like(p)
        for _ in range(self.SOR_ITERS):
            # Red update
            sum_nbr.fill(0.0)
            sum_nbr[1:, :, :] += np.where(xm, p[:-1, :, :], 0.0)
            sum_nbr[:-1, :, :] += np.where(xm, p[1:, :, :], 0.0)
            sum_nbr[:, 1:, :] += np.where(ym, p[:, :-1, :], 0.0)
            sum_nbr[:, :-1, :] += np.where(ym, p[:, 1:, :], 0.0)
            sum_nbr[:, :, 1:] += np.where(zm, p[:, :, :-1], 0.0)
            sum_nbr[:, :, :-1] += np.where(zm, p[:, :, 1:], 0.0)
            p_target = (sum_nbr - rhs) * inv_n
            p[red] += omega * (p_target[red] - p[red])

            # Black update
            sum_nbr.fill(0.0)
            sum_nbr[1:, :, :] += np.where(xm, p[:-1, :, :], 0.0)
            sum_nbr[:-1, :, :] += np.where(xm, p[1:, :, :], 0.0)
            sum_nbr[:, 1:, :] += np.where(ym, p[:, :-1, :], 0.0)
            sum_nbr[:, :-1, :] += np.where(ym, p[:, 1:, :], 0.0)
            sum_nbr[:, :, 1:] += np.where(zm, p[:, :, :-1], 0.0)
            sum_nbr[:, :, :-1] += np.where(zm, p[:, :, 1:], 0.0)
            p_target = (sum_nbr - rhs) * inv_n
            p[black] += omega * (p_target[black] - p[black])

        # Subtract pressure gradient from all pressure-open MAC faces
        inv_h = 1.0 / h
        u[1:nx, :, :] -= np.where(u_p_open[1:nx, :, :], (p[1:, :, :] - p[:-1, :, :]) * inv_h, 0.0)
        u[0, :, :] -= np.where(u_p_open[0, :, :], p[0, :, :] * inv_h, 0.0)
        u[nx, :, :] -= np.where(u_p_open[nx, :, :], -p[nx - 1, :, :] * inv_h, 0.0)

        v[:, 1:ny, :] -= np.where(v_p_open[:, 1:ny, :], (p[:, 1:, :] - p[:, :-1, :]) * inv_h, 0.0)
        v[:, 0, :] -= np.where(v_p_open[:, 0, :], p[:, 0, :] * inv_h, 0.0)
        v[:, ny, :] -= np.where(v_p_open[:, ny, :], -p[:, ny - 1, :] * inv_h, 0.0)

        w[:, :, 1:nz] -= np.where(self._w_open[:, :, 1:nz], (p[:, :, 1:] - p[:, :, :-1]) * inv_h, 0.0)
        w[:, :, nz] -= np.where(self._w_open[:, :, nz], -p[:, :, nz - 1] * inv_h, 0.0)

        u[~self._u_open] = 0.0
        v[~self._v_open] = 0.0
        w[~self._w_open] = 0.0

        self._u_unit = u
        self._v_unit = v
        self._w_unit = w
        self._turb_grid = turb

        # Free-slip ghost fill for trilinear sampling near curved/flat walls:
        # prevents 2.5 m voxel stair-stepping from artificially dragging tangential flank flow to 0,
        # while exact analytical _inside_solid_mask keeps interior velocity strictly 0.
        self._u_samp = self._ghost_fill_tangential(u, self._u_open, axis=0)
        self._v_samp = self._ghost_fill_tangential(v, self._v_open, axis=1)
        self._w_samp = self._ghost_fill_tangential(w, self._w_open, axis=2)
        self._field_version += 1

    @staticmethod
    def _ghost_fill_tangential(field: np.ndarray, open_mask: np.ndarray, axis: int) -> np.ndarray:
        """Copy neighboring open-face tangential velocities into 1-cell solid ghost faces."""
        out = field.copy()
        closed = ~open_mask
        if not np.any(closed):
            return out
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for ax in (0, 1, 2):
            if ax == axis:
                continue  # Only extrapolate along tangential axes, never across normal wall face
            for shift in (-1, 1):
                src_slice = [slice(None)] * 3
                dst_slice = [slice(None)] * 3
                if shift == 1:
                    src_slice[ax] = slice(1, None)
                    dst_slice[ax] = slice(0, -1)
                else:
                    src_slice[ax] = slice(0, -1)
                    dst_slice[ax] = slice(1, None)
                s_t, d_t = tuple(src_slice), tuple(dst_slice)
                valid = open_mask[s_t]
                acc[d_t] += np.where(valid, field[s_t], 0.0)
                cnt[d_t] += valid.astype(np.float64)
        fill = closed & (cnt > 0.0)
        out[fill] = acc[fill] / cnt[fill]
        return out

    def _init_eddy_modes(self) -> None:
        """Build K strictly divergence-free 3D Kolmogorov-like spatial eddy modes."""
        wavelengths = np.geomspace(14.0, 48.0, self.N_EDDY_MODES)
        for m, lam in enumerate(wavelengths):
            k_mag = 2.0 * math.pi / float(lam)
            dir_vec = self._rng.normal(0.0, 1.0, size=3)
            dir_vec[2] *= 0.6  # Slightly flatter atmospheric eddies
            n_dir = float(np.linalg.norm(dir_vec))
            if n_dir < 1e-6:
                dir_vec = np.array([1.0, 0.0, 0.0])
            else:
                dir_vec /= n_dir
            self._k_vecs[m] = k_mag * dir_vec

            # Polarization orthogonal to k_vec with vertical damping re-projected so k . e == 0 exactly
            rand_v = self._rng.normal(0.0, 1.0, size=3)
            pol = np.cross(dir_vec, rand_v)
            if float(np.linalg.norm(pol)) < 1e-6:
                pol = np.cross(dir_vec, np.array([0.0, 0.0, 1.0]))
                if float(np.linalg.norm(pol)) < 1e-6:
                    pol = np.array([0.0, 1.0, 0.0])
            pol /= float(np.linalg.norm(pol))
            pol[2] *= 0.45
            pol = pol - float(np.dot(pol, dir_vec)) * dir_vec
            n_pol = float(np.linalg.norm(pol))
            if n_pol < 1e-6:
                pol = np.cross(dir_vec, np.array([0.0, 0.0, 1.0]))
                pol /= max(float(np.linalg.norm(pol)), 1e-6)
            else:
                pol /= n_pol
            self._e_vecs[m] = pol
            self._phases[m] = float(self._rng.uniform(0.0, 2.0 * math.pi))
            self._eddy_weights[m] = (float(lam) / 28.0) ** 0.55

        rms = float(np.sqrt(np.sum(0.5 * (self._eddy_weights ** 2))))
        if rms > 1e-6:
            self._eddy_weights /= rms

    def set_wind(self, angle_deg: float, speed: float) -> None:
        """Set base wind direction (degrees, 0 = +X, 90 = +Y) and speed (m/s)."""
        rad = math.radians(angle_deg)
        spd = max(0.0, float(speed))
        ux = math.cos(rad)
        uy = math.sin(rad)
        self.w0[0] = spd * ux
        self.w0[1] = spd * uy

        # Clamp existing gust magnitude when wind speed is lowered so stale gusts
        # don't overpower a low setpoint speed.
        max_gust = max(0.05, 0.28 * spd)
        gust_mag = math.hypot(float(self.w_turb[0]), float(self.w_turb[1]))
        if gust_mag > max_gust:
            self.w_turb *= max_gust / gust_mag

        # Re-solve 3D Poisson wind field only if wind direction changed meaningfully
        if (
            self._solved_angle_rad is None
            or abs(math.atan2(math.sin(rad - self._solved_angle_rad), math.cos(rad - self._solved_angle_rad))) > 1e-3
        ):
            self._solved_angle_rad = rad
            self._solve_unit_wind_field(ux, uy)
        else:
            self._field_version += 1

    def reset(self, seed: int | None = None) -> None:
        """Reset turbulent gust state and eddy advection clock."""
        if seed is not None:
            self._seed = seed
            self._rng = np.random.default_rng(seed)
            self._init_eddy_modes()
        self.w_turb[:] = 0.0
        self._time = 0.0
        self._eddy_offset[:] = 0.0

    def step(self, dt: float) -> None:
        """Advance Ornstein-Uhlenbeck turbulent gust vector and integrated eddy offset."""
        self._time += dt
        self._eddy_offset += self.w0 * dt
        if self.sigma_w > 0.0:
            base_speed = math.hypot(float(self.w0[0]), float(self.w0[1]))
            # Scale turbulence intensity with mean wind speed (~14% stationary std)
            # so gusts add natural meandering without overpowering low wind speeds.
            eff_sigma = min(self.sigma_w, (0.14 * base_speed) / math.sqrt(0.5 * self.tau_w))
            noise = self._rng.normal(0.0, 1.0, size=2)
            dw = -(1.0 / self.tau_w) * self.w_turb * dt + eff_sigma * math.sqrt(dt) * noise
            self.w_turb += dw
        else:
            self.w_turb[:] = 0.0

    def current_wind_vec(self) -> Vec2:
        """Return spatially uniform reference-height ambient wind vector (wx, wy)."""
        w = self.w0 + self.w_turb
        return (float(w[0]), float(w[1]))

    def _abl_factor(self, z: np.ndarray) -> np.ndarray:
        """Atmospheric boundary layer power-law shear profile U(z) / U(z_ref)."""
        z_eff = np.maximum(z, 0.5)
        return np.clip(np.power(z_eff / self.Z_REF, self.ABL_ALPHA), 0.55, 1.35)

    @staticmethod
    def _trilinear_sample(
        grid: np.ndarray, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
    ) -> np.ndarray:
        """Fast vectorized trilinear interpolation on a 3D array."""
        nx, ny, nz = grid.shape
        x = np.clip(gx, 0.0, nx - 1.000001)
        y = np.clip(gy, 0.0, ny - 1.000001)
        z = np.clip(gz, 0.0, nz - 1.000001)

        i0 = x.astype(np.intp)
        j0 = y.astype(np.intp)
        k0 = z.astype(np.intp)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        fx = x - i0
        fy = y - j0
        fz = z - k0
        omx = 1.0 - fx
        omy = 1.0 - fy
        omz = 1.0 - fz

        c000 = grid[i0, j0, k0]
        c100 = grid[i1, j0, k0]
        c010 = grid[i0, j1, k0]
        c110 = grid[i1, j1, k0]
        c001 = grid[i0, j0, k1]
        c101 = grid[i1, j0, k1]
        c011 = grid[i0, j1, k1]
        c111 = grid[i1, j1, k1]

        c00 = c000 * omx + c100 * fx
        c10 = c010 * omx + c110 * fx
        c01 = c001 * omx + c101 * fx
        c11 = c011 * omx + c111 * fx

        c0 = c00 * omy + c10 * fy
        c1 = c01 * omy + c11 * fy
        return c0 * omz + c1 * fz

    def _inside_solid_mask(self, pts: np.ndarray) -> np.ndarray:
        """Vectorized check for points strictly inside solid buildings or cylindrical tanks."""
        inside = np.zeros(pts.shape[0], dtype=bool)
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        for i in range(len(self.world.buildings)):
            if self.world._bldg_is_porous[i]:
                continue
            z0 = float(self.world._bldg_min[i, 2])
            z1 = float(self.world._bldg_max[i, 2])
            if self.world._bldg_is_cylinder[i]:
                cx = float(self.world._bldg_center[i, 0])
                cy = float(self.world._bldg_center[i, 1])
                r = float(self.world._bldg_cyl_radius[i])
                inside |= ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) & (z >= z0) & (z <= z1)
            else:
                bmin = self.world._bldg_min[i]
                bmax = self.world._bldg_max[i]
                inside |= (
                    (x >= bmin[0])
                    & (x <= bmax[0])
                    & (y >= bmin[1])
                    & (y <= bmax[1])
                    & (z >= z0)
                    & (z <= z1)
                )
        return inside

    def sample_turbulence_intensity(self, points_nx3: np.ndarray) -> np.ndarray:
        """Return local turbulent diffusivity multiplier (>= 1.0; elevated in wakes & near pipes)."""
        _, turb = self.sample_wind_and_turbulence(points_nx3, compute_turb=True)
        return turb

    def sample_wind(self, points_nx3: np.ndarray) -> np.ndarray:
        """Vectorized local 3D wind query at N points from the mass-consistent MAC grid."""
        wind, _ = self.sample_wind_and_turbulence(points_nx3, compute_turb=False)
        return wind

    def sample_wind_and_turbulence(
        self, points_nx3: np.ndarray, compute_turb: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass vectorized local 3D wind and optional wake turbulence multiplier."""
        pts = np.asarray(points_nx3, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        n = pts.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64), np.ones(0, dtype=np.float64)

        x_all = pts[:, 0]
        y_all = pts[:, 1]
        z_all = pts[:, 2]

        # Continuous MAC grid coordinates
        fx = (x_all - self._x_min) * self._inv_dx
        fy = (y_all - self._y_min) * self._inv_dy
        fz = (z_all - self._z_min) * self._inv_dz
        cx_g = fx - 0.5
        cy_g = fy - 0.5
        cz_g = fz - 0.5

        w_amb = self.w0 + self.w_turb
        base_speed = math.hypot(float(self.w0[0]), float(self.w0[1]))
        amb_speed = math.hypot(float(w_amb[0]), float(w_amb[1]))

        out = np.zeros((n, 3), dtype=np.float64)
        if base_speed >= 1e-4:
            # Project live gusted wind onto solved base direction + perpendicular gust residual
            ux0 = float(self.w0[0]) / base_speed
            uy0 = float(self.w0[1]) / base_speed
            s_par = float(w_amb[0]) * ux0 + float(w_amb[1]) * uy0
            u_samp = self._trilinear_sample(self._u_samp, fx, cy_g, cz_g)
            v_samp = self._trilinear_sample(self._v_samp, cx_g, fy, cz_g)
            w_samp = self._trilinear_sample(self._w_samp, cx_g, cy_g, fz)

            out[:, 0] = u_samp * s_par
            out[:, 1] = v_samp * s_par
            out[:, 2] = w_samp * s_par

            # Add small perpendicular OU gust perturbation (damped smoothly in wakes/near solids)
            perp_x = float(w_amb[0]) - s_par * ux0
            perp_y = float(w_amb[1]) - s_par * uy0
            if abs(perp_x) + abs(perp_y) > 1e-6:
                local_mag = np.hypot(u_samp, v_samp)
                abl_loc = self._abl_factor(z_all)
                flow_weight = np.clip(local_mag / np.maximum(abl_loc, 0.2), 0.0, 1.0)
                out[:, 0] += perp_x * abl_loc * flow_weight
                out[:, 1] += perp_y * abl_loc * flow_weight
        elif amb_speed >= 1e-4:
            abl_loc = self._abl_factor(z_all)
            out[:, 0] = float(w_amb[0]) * abl_loc
            out[:, 1] = float(w_amb[1]) * abl_loc

        if compute_turb:
            turb_mult = self._trilinear_sample(self._turb_grid, cx_g, cy_g, cz_g)
            turb_mult = np.maximum(1.0, turb_mult)
        else:
            turb_mult = np.empty(0, dtype=np.float64)

        # 2) Strictly divergence-free 3D Fourier turbulent eddies (advected via integrated offset)
        if self.sigma_w > 0.0:
            adv_x = x_all - self._eddy_offset[0]
            adv_y = y_all - self._eddy_offset[1]
            phase = (
                adv_x[:, None] * self._k_vecs[None, :, 0]
                + adv_y[:, None] * self._k_vecs[None, :, 1]
                + z_all[:, None] * self._k_vecs[None, :, 2]
                + self._phases[None, :]
            )
            eddy_amp = min(0.45 * self.sigma_w, 0.12 * base_speed)
            sin_p = np.sin(phase) * self._eddy_weights[None, :]
            eddy_vel = (sin_p @ self._e_vecs) * eddy_amp
            # Gently damp vertical eddies right next to the ground (z < 2m)
            ground_z_damp = np.clip(z_all / 2.0, 0.0, 1.0)
            eddy_vel[:, 2] *= ground_z_damp
            out += eddy_vel

            # 3) Cylindrical Strouhal vortex shedding in lee wakes of storage tanks
            if amb_speed > 0.2:
                ux = float(w_amb[0]) / amb_speed
                uy = float(w_amb[1]) / amb_speed
                px_u = -uy
                py_u = ux
                for i in range(len(self.world.buildings)):
                    if not self.world._bldg_is_cylinder[i]:
                        continue
                    cx = float(self.world._bldg_center[i, 0])
                    cy = float(self.world._bldg_center[i, 1])
                    r_cyl = float(self.world._bldg_cyl_radius[i])
                    height = float(self.world._bldg_max[i, 2])
                    dx_b = x_all - cx
                    dy_b = y_all - cy
                    along = dx_b * ux + dy_b * uy
                    cross = dx_b * px_u + dy_b * py_u
                    wake_len = 2.5 * height
                    shed_mask = (
                        (along > r_cyl)
                        & (along <= r_cyl + wake_len)
                        & (np.abs(cross) <= r_cyl * 1.35)
                        & (z_all <= height * 1.15)
                    )
                    if np.any(shed_mask):
                        a_w = along[shed_mask] - r_cyl
                        c_w = cross[shed_mask]
                        z_w = z_all[shed_mask]
                        env = (
                            np.exp(-0.5 * (c_w / max(r_cyl * 0.85, 1.0)) ** 2)
                            * np.sin(math.pi * np.clip(a_w / wake_len, 0.0, 1.0))
                            * 0.5 * (1.0 - np.tanh((z_w - height) / 1.5))
                        )
                        strouhal_omega = 2.0 * math.pi * 0.20 * amb_speed / max(2.0 * r_cyl, 4.0)
                        swirl = 0.18 * amb_speed * env * np.sin(strouhal_omega * self._time - 0.35 * a_w)
                        out[shed_mask, 0] += swirl * px_u
                        out[shed_mask, 1] += swirl * py_u

        # Enforce zero wind strictly inside solid buildings & cylindrical tanks
        solid_pts = self._inside_solid_mask(pts)
        if np.any(solid_pts):
            out[solid_pts] = 0.0

        if single:
            return out[0], (turb_mult[0] if compute_turb else turb_mult)
        return out, turb_mult

    def get_wind_field_payload(self, stride: int = 1) -> WindFieldPayload:
        """Export full-resolution lower-troposphere 3D wind grid (z in [0, 20m]) for visualization."""
        base_speed = math.hypot(float(self.w0[0]), float(self.w0[1]))
        # Use free-slip ghost-filled velocity for crisp flank speedup around tanks/buildings,
        # while strictly zeroing solid interiors.
        uc = 0.5 * (self._u_samp[:-1, :, :] + self._u_samp[1:, :, :]) * base_speed
        vc = 0.5 * (self._v_samp[:, :-1, :] + self._v_samp[:, 1:, :]) * base_speed
        wc = 0.5 * (self._w_samp[:, :, :-1] + self._w_samp[:, :, 1:]) * base_speed

        uc[self._solid] = 0.0
        vc[self._solid] = 0.0
        wc[self._solid] = 0.0

        # Cover z in [0, 20m] (first 8 of 12 vertical layers -> 48x48x8 at full 2.5m resolution)
        kz_vis = min(self.NZ, 8)
        uc_s = uc[::stride, ::stride, :kz_vis]
        vc_s = vc[::stride, ::stride, :kz_vis]
        wc_s = wc[::stride, ::stride, :kz_vis]

        stacked = np.stack([uc_s, vc_s, wc_s], axis=-1)
        nx_s, ny_s, nz_s = uc_s.shape
        vectors = np.round(stacked.ravel(), 2).tolist()
        z_max_vis = self._z_min + kz_vis * self._dz

        return WindFieldPayload(
            bounds_min=self.world.bounds_min,
            bounds_max=(self._x_max, self._y_max, float(z_max_vis)),
            dims=(int(nx_s), int(ny_s), int(nz_s)),
            base_speed=round(base_speed, 2),
            vectors=vectors,
        )
