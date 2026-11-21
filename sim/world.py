"""Deterministic 3D industrial refinery world geometry, assets, and candidate grid."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from shared.schemas import (
    BuildingBox,
    FacilityAsset,
    GridPointMeta,
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


class World:
    """120 x 120 x 30 m refinery layout (Right-handed Z-up, ground at z = 0)."""

    def __init__(self) -> None:
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
                min_xyz=(-10.0, 14.0, 0.0),
                max_xyz=(10.0, 34.0, 16.0),
            ),
            BuildingBox(
                id="tank_east",
                kind="storage_tank",
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

        # Precomputed NumPy arrays for fast vectorized geometry queries
        self._bldg_min = np.array([b.min_xyz for b in self.buildings], dtype=np.float64)
        self._bldg_max = np.array([b.max_xyz for b in self.buildings], dtype=np.float64)
        self._bldg_center = 0.5 * (self._bldg_min + self._bldg_max)
        self._bldg_half = 0.5 * (self._bldg_max - self._bldg_min)

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

    def is_inside_building(self, pos: Vec3, margin: float = 0.0) -> bool:
        """Return True if pos lies inside any BuildingBox expanded by margin."""
        p = np.asarray(pos, dtype=np.float64)
        lo = self._bldg_min - np.array([margin, margin, 0.0])
        hi = self._bldg_max + margin
        inside = np.all((p >= lo) & (p <= hi), axis=1)
        return bool(np.any(inside))

    def push_outside_buildings(
        self, points_nx3: np.ndarray, margin: float = 0.0
    ) -> np.ndarray:
        """Vectorized push-out: moves any point inside a building to its nearest face."""
        pts = points_nx3.copy()
        if pts.shape[0] == 0:
            return pts

        for i in range(len(self.buildings)):
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

            # Projected building half-extents along and perpendicular to wind
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
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            buildings=self.buildings,
            assets=self.assets,
            grid_coords=self.grid_coords,
        )
