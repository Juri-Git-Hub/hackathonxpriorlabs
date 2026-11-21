"""Deterministic 3D industrial refinery world geometry, assets, occlusion, and candidate grid."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from shared.schemas import (
    BuildingBox,
    FacilityAsset,
    GridPointMeta,
    PipeRun,
    Vec2,
    Vec3,
    WorldInitPayload,
)

NearestAssetLiteral = Literal["open_air", "flange", "valve", "storage_tank", "wall"]

_ASSET_TYPE_MAP: dict[str, NearestAssetLiteral] = {
    "valve": "valve",
    "flange": "flange",
    "storage_tank": "storage_tank",
    "pipe_junction": "flange",
}


_MAPS = json.loads(
    (Path(__file__).resolve().parents[1] / "frontend" / "src" / "industrialMaps.json").read_text()
)


class World:
    """120 x 120 x 30 m refinery layout (Right-handed Z-up, ground at z = 0)."""

    def __init__(self, layout_seed: int | None = None, *, map_id: str = "refinery") -> None:
        is_generated = map_id == "generated" and layout_seed is not None
        if map_id not in _MAPS and not is_generated:
            raise ValueError("Choose a valid facility map.")
        self.map_id = map_id
        self.bounds_min: Vec3 = (-60.0, -60.0, 0.0)
        self.bounds_max: Vec3 = (60.0, 60.0, 30.0)

        self.buildings: list[BuildingBox] = [
            BuildingBox(
                id="bldg_control",
                kind="building",
                min_xyz=(-38.0, -28.0, 0.0),
                max_xyz=(-20.0, -12.0, 12.0),
            ),
            BuildingBox(
                id="tank_north",
                kind="storage_tank",
                model="fractionator",
                min_xyz=(-10.0, 14.0, 0.0),
                max_xyz=(10.0, 34.0, 16.0),
            ),
            BuildingBox(
                id="tank_east",
                kind="storage_tank",
                model="horton_sphere",
                min_xyz=(22.0, -16.0, 0.0),
                max_xyz=(40.0, 2.0, 14.0),
            ),
            BuildingBox(
                id="pipe_rack_center",
                kind="pipe_rack",
                min_xyz=(-16.0, -6.0, 0.0),
                max_xyz=(14.0, 4.0, 8.0),
            ),
            BuildingBox(
                id="bldg_compressor",
                kind="building",
                min_xyz=(-35.0, 16.0, 0.0),
                max_xyz=(-18.0, 32.0, 10.0),
            ),
        ]

        self.assets: list[FacilityAsset] = [
            FacilityAsset(id="valve_01", asset_type="valve", pos=(-17.5, -2.0, 3.5), age_years=18.5),
            FacilityAsset(id="flange_01", asset_type="flange", pos=(-6.0, -7.2, 4.0), age_years=22.0),
            FacilityAsset(id="pipe_junc_01", asset_type="pipe_junction", pos=(4.0, -7.2, 5.0), age_years=14.0),
            FacilityAsset(id="valve_02", asset_type="valve", pos=(15.5, -1.0, 4.5), age_years=9.5),
            FacilityAsset(id="flange_02", asset_type="flange", pos=(-10.0, 5.5, 4.0), age_years=19.0),
            FacilityAsset(id="pipe_junc_02", asset_type="pipe_junction", pos=(8.0, 5.5, 6.0), age_years=7.0),
            FacilityAsset(id="tank_port_n1", asset_type="storage_tank", pos=(0.0, 12.5, 5.0), age_years=16.0),
            FacilityAsset(id="tank_port_n2", asset_type="storage_tank", pos=(11.5, 24.0, 8.0), age_years=21.5),
            FacilityAsset(id="tank_port_e1", asset_type="storage_tank", pos=(20.5, -7.0, 6.0), age_years=24.0),
            FacilityAsset(id="valve_03", asset_type="valve", pos=(31.0, -17.5, 3.0), age_years=11.0),
            FacilityAsset(id="flange_03", asset_type="flange", pos=(-18.5, -20.0, 4.0), age_years=5.5),
            FacilityAsset(id="valve_04", asset_type="valve", pos=(-29.0, -10.5, 3.5), age_years=15.0),
            FacilityAsset(id="flange_04", asset_type="flange", pos=(-16.5, 24.0, 4.5), age_years=20.0),
            FacilityAsset(id="pipe_junc_03", asset_type="pipe_junction", pos=(-26.0, 14.5, 5.0), age_years=3.0),
            FacilityAsset(id="valve_05", asset_type="valve", pos=(0.0, -25.0, 2.5), age_years=12.5),
        ]

        # Single source of truth for elevated industrial piping network (shared with frontend)
        self.pipes: list[PipeRun] = [
            PipeRun(
                id="line_gas_main",
                name="Main Methane Header",
                service="High-Pressure Gas",
                pressure_bar=18.4,
                radius=0.34,
                material="gas",
                points=[
                    (-20.0, -14.0, 3.5),
                    (-17.5, -14.0, 3.5),
                    (-17.5, -2.0, 3.5),
                    (-14.0, -2.0, 4.4),
                    (14.0, -2.0, 4.4),
                    (15.5, -1.0, 4.5),
                    (20.5, -1.0, 4.5),
                    (20.5, -7.0, 6.0),
                    (23.0, -7.0, 6.0),
                ],
            ),
            PipeRun(
                id="line_east_transfer",
                name="East Tank Transfer Line",
                service="Process Feed",
                pressure_bar=11.2,
                radius=0.28,
                material="steel",
                points=[
                    (31.0, -15.0, 3.0),
                    (31.0, -17.5, 3.0),
                    (14.0, -17.5, 3.0),
                    (4.0, -17.5, 3.0),
                    (4.0, -7.2, 5.0),
                    (4.0, -3.0, 5.0),
                ],
            ),
            PipeRun(
                id="line_north_gas",
                name="North Tank Gas Feed",
                service="High-Pressure Gas",
                pressure_bar=16.0,
                radius=0.32,
                material="gas",
                points=[
                    (-10.0, 3.0, 4.0),
                    (-10.0, 5.5, 4.0),
                    (0.0, 5.5, 4.0),
                    (0.0, 12.5, 5.0),
                    (0.0, 15.0, 5.0),
                ],
            ),
            PipeRun(
                id="line_north_coolant",
                name="North Cooling Return",
                service="Coolant Loop",
                pressure_bar=6.5,
                radius=0.30,
                material="coolant",
                points=[
                    (8.0, 3.0, 6.0),
                    (8.0, 5.5, 6.0),
                    (11.5, 5.5, 6.0),
                    (11.5, 24.0, 8.0),
                    (9.0, 24.0, 8.0),
                ],
            ),
            PipeRun(
                id="line_comp_discharge",
                name="Compressor Discharge",
                service="Compressed Vapor",
                pressure_bar=22.0,
                radius=0.28,
                material="steel",
                points=[
                    (-19.0, 24.0, 4.5),
                    (-16.5, 24.0, 4.5),
                    (-9.0, 24.0, 4.5),
                ],
            ),
            PipeRun(
                id="line_comp_coolant",
                name="Compressor Coolant",
                service="Coolant Loop",
                pressure_bar=5.8,
                radius=0.26,
                material="coolant",
                points=[
                    (-26.0, 17.0, 5.0),
                    (-26.0, 14.5, 5.0),
                    (-26.0, 2.0, 4.0),
                    (-15.0, 2.0, 4.0),
                ],
            ),
            PipeRun(
                id="line_south_header",
                name="South Manifold Gas Line",
                service="High-Pressure Gas",
                pressure_bar=17.5,
                radius=0.30,
                material="gas",
                points=[
                    (-6.0, -3.5, 4.0),
                    (-6.0, -7.2, 4.0),
                    (-6.0, -25.0, 2.5),
                    (0.0, -25.0, 2.5),
                    (0.0, -25.0, 0.2),
                ],
            ),
            PipeRun(
                id="line_control_bypass",
                name="Control Unit Bypass",
                service="Instrument Line",
                pressure_bar=9.0,
                radius=0.26,
                material="steel",
                points=[
                    (-29.0, -13.0, 3.5),
                    (-29.0, -10.5, 3.5),
                    (-18.5, -10.5, 3.5),
                    (-18.5, -20.0, 4.0),
                    (-21.0, -20.0, 4.0),
                ],
            ),
        ]

        self.layout_seed = layout_seed if map_id == "refinery" or is_generated else None
        self.theme: str | None = _MAPS.get(map_id, {}).get("presentation", {}).get("theme")
        self.pipe_runs = self.pipes
        if self.layout_seed is not None:
            from sim.layout import generate_layout, layout_theme
            self.buildings, self.assets, self.pipe_runs = generate_layout(self.buildings, self.assets, self.layout_seed)
            self.theme = layout_theme(self.layout_seed)

        if map_id in _MAPS and map_id != "refinery":
            layout = _MAPS[map_id]
            self.buildings = [BuildingBox.model_validate(box) for box in layout["buildings"]]
            self.assets = [FacilityAsset.model_validate(asset) for asset in layout["assets"]]
            self.pipe_runs = [PipeRun.model_validate(run) for run in layout["pipe_runs"]]
            self.default_leak_pos = next(
                asset.pos for asset in self.assets if asset.id == layout["default_leak_asset"]
            )
            for run in self.pipe_runs:
                for index, (start, end) in enumerate(zip(run.points, run.points[1:])):
                    # Use an inscribed box for each straight pipe segment so
                    # the rendered cylindrical surface remains placeable.
                    half_width = run.radius / math.sqrt(2)
                    pad = tuple(half_width if a == b else 0.0 for a, b in zip(start, end))
                    self.buildings.append(BuildingBox(
                        id=f"{run.id}_{index}",
                        kind="pipe_segment",
                        min_xyz=tuple(min(a, b) - p for a, b, p in zip(start, end, pad)),
                        max_xyz=tuple(max(a, b) + p for a, b, p in zip(start, end, pad)),
                    ))

        else:
            self.default_leak_pos = next(asset.pos for asset in self.assets if asset.id == "flange_01")
        self.pipes = self.pipe_runs

        # Precompute flattened pipe segments for fast vectorized distance & drag queries
        seg_a: list[Vec3] = []
        seg_b: list[Vec3] = []
        seg_r: list[float] = []
        for pipe in self.pipes:
            for j in range(len(pipe.points) - 1):
                a = pipe.points[j]
                b = pipe.points[j + 1]
                if math.dist(a, b) > 0.05:
                    seg_a.append(a)
                    seg_b.append(b)
                    seg_r.append(pipe.radius)
        self._pipe_seg_a = np.array(seg_a, dtype=np.float64) if seg_a else np.zeros((0, 3), dtype=np.float64)
        self._pipe_seg_b = np.array(seg_b, dtype=np.float64) if seg_b else np.zeros((0, 3), dtype=np.float64)
        self._pipe_seg_ab = self._pipe_seg_b - self._pipe_seg_a
        self._pipe_seg_len2 = np.maximum(np.sum(self._pipe_seg_ab ** 2, axis=1), 1e-9)
        self._pipe_seg_r = np.array(seg_r, dtype=np.float64) if seg_r else np.zeros(0, dtype=np.float64)

        self._pipe_top = float(np.max(
            np.maximum(self._pipe_seg_a[:, 2], self._pipe_seg_b[:, 2]) + self._pipe_seg_r,
            initial=0.0,
        ))

        # Precomputed NumPy arrays for fast vectorized geometry & physics queries
        self._bldg_min = np.array([b.min_xyz for b in self.buildings], dtype=np.float64)
        self._bldg_max = np.array([b.max_xyz for b in self.buildings], dtype=np.float64)
        self._bldg_center = 0.5 * (self._bldg_min + self._bldg_max)
        self._bldg_half = 0.5 * (self._bldg_max - self._bldg_min)

        # Kind-specific aerodynamic & occlusion properties
        self._bldg_is_cylinder = np.array(
            [b.kind == "storage_tank" for b in self.buildings], dtype=bool
        )
        self._bldg_is_porous = np.array(
            [b.kind == "pipe_rack" for b in self.buildings], dtype=bool
        )
        self._bldg_cyl_radius = np.minimum(self._bldg_half[:, 0], self._bldg_half[:, 1])
        # Gas transmission through obstacle when segment crosses it
        # Solid buildings/tanks strongly block gas (only ~5% corner diffraction);
        # open-truss pipe racks allow ~55% transmission.
        self._bldg_transmission = np.where(self._bldg_is_porous, 0.55, 0.05)

        # Scalar tuples for zero-overhead inner-loop drone physics
        self._bldg_bounds_scalar: list[tuple[float, float, float, float, float, float, float, float, float]] = [
            (
                float(self._bldg_min[i, 0]),
                float(self._bldg_min[i, 1]),
                float(self._bldg_min[i, 2]),
                float(self._bldg_max[i, 0]),
                float(self._bldg_max[i, 1]),
                float(self._bldg_max[i, 2]),
                float(self._bldg_center[i, 0]),
                float(self._bldg_center[i, 1]),
                float(self._bldg_center[i, 2]),
            )
            for i in range(len(self.buildings))
        ]

        self._asset_pos = np.array([a.pos for a in self.assets], dtype=np.float64)
        self._asset_age = np.array([a.age_years for a in self.assets], dtype=np.float64)
        self._asset_types: list[NearestAssetLiteral] = [
            _ASSET_TYPE_MAP[a.asset_type] for a in self.assets
        ]

        self.grid_coords: list[Vec3] = self._generate_candidate_grid()
        self.grid_coords_np = np.array(self.grid_coords, dtype=np.float64)

    def _generate_candidate_grid(self, safety_margin: float = 2.0) -> list[Vec3]:
        """Generate ~800 navigable 3D candidate points outside buildings + margin."""
        xs = np.linspace(-52.0, 52.0, 12)
        ys = np.linspace(-52.0, 52.0, 12)
        zs = np.array([3., 8., 13., 18., 23., 28.])

        coords: list[Vec3] = []
        for z in zs:
            for y in ys:
                for x in xs:
                    pt = (float(x), float(y), float(z))
                    if not self.is_inside_building(pt, margin=safety_margin):
                        coords.append(pt)
        return coords

    def is_inside_building(
        self, pos: Vec3, margin: float = 0.0, skip_porous: bool = False
    ) -> bool:
        """Return True if pos lies inside any solid obstacle (exact cylinder for tanks)."""
        px, py, pz = pos
        for i, (x0, y0, z0, x1, y1, z1, cx, cy, _) in enumerate(self._bldg_bounds_scalar):
            if skip_porous and self._bldg_is_porous[i]:
                continue
            if pz < z0 or pz > z1 + margin:
                continue
            if self._bldg_is_cylinder[i]:
                r = float(self._bldg_cyl_radius[i]) + margin
                dx = px - cx
                dy = py - cy
                if dx * dx + dy * dy <= r * r:
                    return True
            else:
                if (
                    px >= x0 - margin
                    and px <= x1 + margin
                    and py >= y0 - margin
                    and py <= y1 + margin
                ):
                    return True
        return False

    def height_above_surface(self, px: float, py: float, pz: float) -> float:
        """Vertical clearance above ground (z=0) or any building/tank roof directly below."""
        h_agl = pz
        for i, (x0, y0, _, x1, y1, z1, cx, cy, _) in enumerate(self._bldg_bounds_scalar):
            if pz < z1:
                continue
            if self._bldg_is_cylinder[i]:
                r = float(self._bldg_cyl_radius[i])
                if (px - cx) ** 2 + (py - cy) ** 2 <= r * r:
                    h_roof = pz - z1
                    if h_roof < h_agl:
                        h_agl = h_roof
            elif x0 <= px <= x1 and y0 <= py <= y1:
                h_roof = pz - z1
                if h_roof < h_agl:
                    h_agl = h_roof
        return max(0.05, h_agl)

    def push_outside_buildings(
        self, points_nx3: np.ndarray, margin: float = 0.0, skip_porous: bool = False
    ) -> np.ndarray:
        """Vectorized push-out: moves points inside buildings/tanks to nearest surface.

        Uses exact cylinder geometry for storage tanks and skips open-truss pipe racks
        when skip_porous=True (so gas puffs flow through pipe racks instead of teleporting).
        """
        pts = points_nx3.copy()
        if pts.shape[0] == 0:
            return pts

        for i in range(len(self.buildings)):
            if skip_porous and self._bldg_is_porous[i]:
                continue

            if self._bldg_is_cylinder[i]:
                cx = float(self._bldg_center[i, 0])
                cy = float(self._bldg_center[i, 1])
                r_eff = float(self._bldg_cyl_radius[i]) + margin
                z_lo = float(self._bldg_min[i, 2])
                z_hi = float(self._bldg_max[i, 2]) + margin

                dx = pts[:, 0] - cx
                dy = pts[:, 1] - cy
                r_dist = np.hypot(dx, dy)
                inside = (r_dist <= r_eff) & (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
                if not np.any(inside):
                    continue

                sub = pts[inside]
                d_rad = r_eff - r_dist[inside]
                d_top = z_hi - sub[:, 2]
                push_top = d_top < d_rad

                if np.any(push_top):
                    sub[push_top, 2] = z_hi + 0.05
                push_side = ~push_top
                if np.any(push_side):
                    sdx = dx[inside][push_side]
                    sdy = dy[inside][push_side]
                    sr = r_dist[inside][push_side]
                    safe_r = np.where(sr > 1e-6, sr, 1.0)
                    ux = np.where(sr > 1e-6, sdx / safe_r, 1.0)
                    uy = np.where(sr > 1e-6, sdy / safe_r, 0.0)
                    sub[push_side, 0] = cx + ux * (r_eff + 0.05)
                    sub[push_side, 1] = cy + uy * (r_eff + 0.05)

                pts[inside] = sub
                continue

            lo = self._bldg_min[i] - np.array([margin, margin, 0.0])
            hi = self._bldg_max[i] + margin

            inside = (
                (pts[:, 0] >= lo[0])
                & (pts[:, 0] <= hi[0])
                & (pts[:, 1] >= lo[1])
                & (pts[:, 1] <= hi[1])
                & (pts[:, 2] >= lo[2])
                & (pts[:, 2] <= hi[2])
            )
            if not np.any(inside):
                continue

            sub = pts[inside]
            # Distances to the 5 escapable faces (-x, +x, -y, +y, +z)
            d_xmin = sub[:, 0] - lo[0]
            d_xmax = hi[0] - sub[:, 0]
            d_ymin = sub[:, 1] - lo[1]
            d_ymax = hi[1] - sub[:, 1]
            d_zmax = hi[2] - sub[:, 2]

            dists = np.stack([d_xmin, d_xmax, d_ymin, d_ymax, d_zmax], axis=1)
            face = np.argmin(dists, axis=1)

            m0 = face == 0
            m1 = face == 1
            m2 = face == 2
            m3 = face == 3
            m4 = face == 4

            sub[m0, 0] = lo[0] - 0.05
            sub[m1, 0] = hi[0] + 0.05
            sub[m2, 1] = lo[1] - 0.05
            sub[m3, 1] = hi[1] + 0.05
            sub[m4, 2] = hi[2] + 0.05

            pts[inside] = sub

        return pts

    def push_outside_pipes(
        self, points_nx3: np.ndarray, margin: float = 0.05
    ) -> np.ndarray:
        """Vectorized continuous capsule push-out so puffs deflect cleanly around pipe runs."""
        if points_nx3.shape[0] == 0 or self._pipe_seg_a.shape[0] == 0:
            return points_nx3

        pts = points_nx3.copy()
        low_mask = pts[:, 2] <= self._pipe_top + margin
        if not np.any(low_mask):
            return pts

        sub = pts[low_mask]  # (K, 3)
        # Compute closest point on every segment: (K, S, 3)
        ap = sub[:, None, :] - self._pipe_seg_a[None, :, :]
        t = np.sum(ap * self._pipe_seg_ab[None, :, :], axis=2) / self._pipe_seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = self._pipe_seg_a[None, :, :] + t[:, :, None] * self._pipe_seg_ab[None, :, :]
        diff = sub[:, None, :] - closest
        dist2 = np.sum(diff * diff, axis=2)
        min_r = self._pipe_seg_r[None, :] + margin
        hit_mask = dist2 < (min_r * min_r)

        if np.any(hit_mask):
            # Resolve against the deepest penetrating segment per point
            pen = min_r - np.sqrt(np.maximum(dist2, 1e-12))
            pen = np.where(hit_mask, pen, -1.0)
            best_seg = np.argmax(pen, axis=1)
            best_pen = pen[np.arange(sub.shape[0]), best_seg]
            fix = best_pen > 0.0
            if np.any(fix):
                idx = np.nonzero(fix)[0]
                s_idx = best_seg[idx]
                v = diff[idx, s_idx]
                v_norm = np.linalg.norm(v, axis=1, keepdims=True)
                n_hat = np.where(v_norm > 1e-6, v / np.maximum(v_norm, 1e-6), np.array([[0.0, 0.0, 1.0]]))
                sub[idx] = closest[idx, s_idx] + n_hat * (self._pipe_seg_r[s_idx, None] + margin + 0.02)
            pts[low_mask] = sub

        return pts

    def segment_transmission(self, p_from_nx3: np.ndarray, p_to: Vec3) -> np.ndarray:
        """Vectorized gas-transmission factor in (0, 1] from N puff centers to p_to.

        Uses exact ray-cylinder intersection for storage tanks and ray-AABB for
        rectangular buildings and open-truss pipe racks.
        """
        n = p_from_nx3.shape[0]
        trans = np.ones(n, dtype=np.float64)
        if n == 0:
            return trans

        target = np.asarray(p_to, dtype=np.float64)
        d = target[None, :] - p_from_nx3  # (N, 3)

        # Safe inverse direction avoiding division by zero
        eps = 1e-9
        abs_d = np.abs(d)
        parallel = abs_d < eps
        inv_d = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, d))

        for i in range(len(self.buildings)):
            if self._bldg_is_cylinder[i]:
                cx = float(self._bldg_center[i, 0])
                cy = float(self._bldg_center[i, 1])
                r = float(self._bldg_cyl_radius[i])
                z0 = float(self._bldg_min[i, 2])
                z1 = float(self._bldg_max[i, 2])

                # Z-slab interval [tz_enter, tz_exit]
                par_z = parallel[:, 2]
                tz1 = (z0 - p_from_nx3[:, 2]) * inv_d[:, 2]
                tz2 = (z1 - p_from_nx3[:, 2]) * inv_d[:, 2]
                in_z = (p_from_nx3[:, 2] >= z0) & (p_from_nx3[:, 2] <= z1)
                tz_enter = np.where(par_z, np.where(in_z, -1e9, 1e9), np.minimum(tz1, tz2))
                tz_exit = np.where(par_z, np.where(in_z, 1e9, -1e9), np.maximum(tz1, tz2))

                # 2D ray-circle intersection in XY
                ox = p_from_nx3[:, 0] - cx
                oy = p_from_nx3[:, 1] - cy
                dx = d[:, 0]
                dy = d[:, 1]
                a = dx * dx + dy * dy
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - r * r
                disc = b * b - 4.0 * a * c

                vert_ray = a < 1e-9
                inside_cyl_xy = c <= 0.0
                safe_a = np.where(vert_ray, 1.0, a)
                sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
                tc_enter = np.where(
                    vert_ray,
                    np.where(inside_cyl_xy, -1e9, 1e9),
                    np.where(disc >= 0.0, (-b - sqrt_disc) / (2.0 * safe_a), 1e9),
                )
                tc_exit = np.where(
                    vert_ray,
                    np.where(inside_cyl_xy, 1e9, -1e9),
                    np.where(disc >= 0.0, (-b + sqrt_disc) / (2.0 * safe_a), -1e9),
                )

                t_enter = np.maximum(tc_enter, tz_enter)
                t_exit = np.minimum(tc_exit, tz_exit)
                hit = (t_exit >= t_enter) & (t_exit > 0.04) & (t_enter < 0.96)
                if np.any(hit):
                    trans[hit] *= self._bldg_transmission[i]
                continue

            bmin = self._bldg_min[i]
            bmax = self._bldg_max[i]

            t1 = (bmin[None, :] - p_from_nx3) * inv_d
            t2 = (bmax[None, :] - p_from_nx3) * inv_d

            tmin_ax = np.minimum(t1, t2)
            tmax_ax = np.maximum(t1, t2)

            # For axes parallel to the ray, check if origin is inside slab
            in_slab = (p_from_nx3 >= bmin[None, :]) & (p_from_nx3 <= bmax[None, :])
            tmin_ax = np.where(parallel, np.where(in_slab, -1e9, 1e9), tmin_ax)
            tmax_ax = np.where(parallel, np.where(in_slab, 1e9, -1e9), tmax_ax)

            t_enter = np.max(tmin_ax, axis=1)
            t_exit = np.min(tmax_ax, axis=1)

            # Strictly interior segment crossing (avoids self-occluding surface-grazing puffs)
            hit = (t_exit >= t_enter) & (t_exit > 0.04) & (t_enter < 0.96)
            if np.any(hit):
                trans[hit] *= self._bldg_transmission[i]

        return trans

    def is_in_wind_shadow(self, pos: Vec3, wind_vec: Vec2) -> bool:
        """Return True if pos lies in the downwind wake cavity of any building."""
        wx, wy = wind_vec
        speed = math.hypot(wx, wy)
        if speed < 1e-3:
            return False

        ux, uy = wx / speed, wy / speed
        px, py, pz = pos

        for i in range(len(self.buildings)):
            cx, cy, _ = self._bldg_center[i]
            hx, hy, _ = self._bldg_half[i]
            height = float(self._bldg_max[i, 2])
            if pz > height * 1.15:
                continue

            dx = px - cx
            dy = py - cy
            along = dx * ux + dy * uy
            cross = abs(-dx * uy + dy * ux)

            if self._bldg_is_cylinder[i]:
                r = float(self._bldg_cyl_radius[i])
                half_along = r
                half_cross = r
            else:
                half_along = hx * abs(ux) + hy * abs(uy)
                half_cross = hx * abs(uy) + hy * abs(ux)

            if half_along < along <= half_along + 2.5 * height and cross <= half_cross * 1.1:
                return True
        return False

    def query_spatial_context(self, pos: Vec3, wind_vec: Vec2) -> dict:
        """Return contextual tabular features for a 3D point under current wind."""
        p = np.asarray(pos, dtype=np.float64)

        # Nearest facility asset
        diffs = self._asset_pos - p
        dists = np.linalg.norm(diffs, axis=1)
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])

        # Distance to nearest building wall
        clamped = np.clip(p, self._bldg_min, self._bldg_max)
        wall_dists = np.linalg.norm(p - clamped, axis=1)
        min_wall_dist = float(np.min(wall_dists))

        if nearest_dist <= 12.0:
            nearest_asset_type: NearestAssetLiteral = self._asset_types[nearest_idx]
            asset_age_years = float(self._asset_age[nearest_idx])
        elif min_wall_dist <= 5.0:
            nearest_asset_type = "wall"
            asset_age_years = 10.0
        else:
            nearest_asset_type = "open_air"
            asset_age_years = 0.0

        in_shadow = self.is_in_wind_shadow(pos, wind_vec)

        # Upwind pipe/asset score in [0, 1]: weights assets located upwind of pos
        wx, wy = wind_vec
        speed = math.hypot(wx, wy)
        upwind_pipe_score = 0.0
        if speed >= 1e-3:
            ux, uy = wx / speed, wy / speed
            # Vector from asset to pos should align with wind direction (+u)
            from_assets = p[:2] - self._asset_pos[:, :2]
            along = from_assets[:, 0] * ux + from_assets[:, 1] * uy
            cross = np.abs(-from_assets[:, 0] * uy + from_assets[:, 1] * ux)
            mask = (along > 0.0) & (along < 45.0)
            if np.any(mask):
                a_m = along[mask]
                c_m = cross[mask]
                age_w = 0.5 + 0.5 * (self._asset_age[mask] / 25.0)
                cone_sigma = 3.0 + 0.25 * a_m
                contrib = age_w * np.exp(-0.5 * (c_m / cone_sigma) ** 2) * np.exp(-a_m / 30.0)
                upwind_pipe_score = float(np.clip(np.sum(contrib) / 2.5, 0.0, 1.0))

        return {
            "nearest_asset_type": nearest_asset_type,
            "asset_age_years": asset_age_years,
            "in_wind_shadow": in_shadow,
            "upwind_pipe_score": upwind_pipe_score,
        }

    def get_grid_meta(self, wind_vec: Vec2) -> list[GridPointMeta]:
        """Compute GridPointMeta for every candidate grid point under current wind."""
        meta: list[GridPointMeta] = []
        for idx, pt in enumerate(self.grid_coords):
            ctx = self.query_spatial_context(pt, wind_vec)
            meta.append(
                GridPointMeta(
                    idx=idx,
                    pos=pt,
                    nearest_asset_type=ctx["nearest_asset_type"],
                    asset_age_years=ctx["asset_age_years"],
                    in_wind_shadow=ctx["in_wind_shadow"],
                    upwind_pipe_score=ctx["upwind_pipe_score"],
                )
            )
        return meta

    def to_world_init(self) -> WorldInitPayload:
        """Serialize static world geometry for WebSocket init and agent reset."""
        return WorldInitPayload(
            map_id=self.map_id,
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            buildings=self.buildings,
            assets=self.assets,
            pipes=self.pipes,
            grid_coords=self.grid_coords,
            pipe_runs=self.pipe_runs,
            layout_seed=self.layout_seed,
            theme=self.theme,
        )
