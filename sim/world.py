"""Deterministic 3D industrial refinery world geometry, assets, occlusion, and candidate grid."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from shared.flight_geometry import flight_bounds
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

        self.layout_seed = layout_seed
        self.theme: str | None = _MAPS.get(map_id, {}).get("presentation", {}).get("theme")
        self.pipe_runs = self.pipes
        if self.layout_seed is not None and (map_id == "refinery" or is_generated):
            from sim.layout import generate_layout, layout_theme
            self.buildings, self.assets, self.pipe_runs = generate_layout(self.buildings, self.assets, self.layout_seed)
            if is_generated:
                self.theme = layout_theme(self.layout_seed)

        if map_id in _MAPS and map_id != "refinery":
            layout = _MAPS[map_id]
            self.buildings = [BuildingBox.model_validate(box) for box in layout["buildings"]]
            self.assets = [FacilityAsset.model_validate(asset) for asset in layout["assets"]]
            self.pipe_runs = [PipeRun.model_validate(run) for run in layout["pipe_runs"]]
            if self.layout_seed is not None:
                from sim.layout import generate_layout
                self.buildings, self.assets, self.pipe_runs = generate_layout(
                    self.buildings, self.assets, self.layout_seed, preserve_models=True,
                    large_pipes=map_id == "pipe_loop",
                )
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
        flight_lo, flight_hi = flight_bounds(self.buildings, self.pipes)
        self.flight_bounds = tuple(zip(map(tuple, flight_lo.tolist()), map(tuple, flight_hi.tolist())))

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

        # Precompute per-pipe-run and overall AABBs for fast broad-phase culling in push_outside_pipes
        self._pipe_run_aabbs: list[tuple[float, float, float, float, float, float]] = []
        for pipe in self.pipes:
            if len(pipe.points) < 2:
                continue
            pts_p = np.asarray(pipe.points, dtype=np.float64)
            r_p = float(pipe.radius)
            p_lo = pts_p.min(axis=0) - r_p
            p_hi = pts_p.max(axis=0) + r_p
            self._pipe_run_aabbs.append(
                (float(p_lo[0]), float(p_hi[0]), float(p_lo[1]), float(p_hi[1]), float(p_lo[2]), float(p_hi[2]))
            )
        if self._pipe_seg_a.shape[0] > 0:
            seg_min = np.minimum(self._pipe_seg_a, self._pipe_seg_b) - self._pipe_seg_r[:, None]
            seg_max = np.maximum(self._pipe_seg_a, self._pipe_seg_b) + self._pipe_seg_r[:, None]
            self._pipe_all_min: tuple[float, float, float] = (
                float(seg_min[:, 0].min()),
                float(seg_min[:, 1].min()),
                float(seg_min[:, 2].min()),
            )
            self._pipe_all_max: tuple[float, float, float] = (
                float(seg_max[:, 0].max()),
                float(seg_max[:, 1].max()),
                float(seg_max[:, 2].max()),
            )
        else:
            self._pipe_all_min = (0.0, 0.0, 0.0)
            self._pipe_all_max = (-1.0, -1.0, -1.0)

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

        # Scalar tuples for zero-overhead inner-loop drone & geometry physics
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
        # Extended scalar tuples: (x0, y0, z0, x1, y1, z1, cx, cy, hx, hy, r_cyl, r_cyl2, is_cyl, is_porous, trans)
        self._bldg_geom_scalar: list[
            tuple[float, float, float, float, float, float, float, float, float, float, float, float, bool, bool, float]
        ] = [
            (
                float(self._bldg_min[i, 0]),
                float(self._bldg_min[i, 1]),
                float(self._bldg_min[i, 2]),
                float(self._bldg_max[i, 0]),
                float(self._bldg_max[i, 1]),
                float(self._bldg_max[i, 2]),
                float(self._bldg_center[i, 0]),
                float(self._bldg_center[i, 1]),
                float(self._bldg_half[i, 0]),
                float(self._bldg_half[i, 1]),
                float(self._bldg_cyl_radius[i]),
                float(self._bldg_cyl_radius[i] ** 2),
                bool(self._bldg_is_cylinder[i]),
                bool(self._bldg_is_porous[i]),
                float(self._bldg_transmission[i]),
            )
            for i in range(len(self.buildings))
        ]

        # Precompute 3D spatial broad-phase voxel bitmasks (2.0 m voxels over 120x120x30 m)
        # so push_outside_buildings, push_outside_pipes, and _inside_solid_mask reject
        # >95% of open-air points in a single O(1) array lookup with zero approximation.
        self._bp_nx = 60
        self._bp_ny = 60
        self._bp_nz = 15
        self._bp_inv = 0.5  # 1 / 2.0 m
        vx = -60.0 + (np.arange(self._bp_nx, dtype=np.float64) + 0.5) * 2.0
        vy = -60.0 + (np.arange(self._bp_ny, dtype=np.float64) + 0.5) * 2.0
        vz = (np.arange(self._bp_nz, dtype=np.float64) + 0.5) * 2.0
        VX, VY, VZ = np.meshgrid(vx, vy, vz, indexing="ij")
        # Pad by half-voxel diagonal (sqrt(3) ~= 1.74 m) + max supported query margin (0.5 m) = 2.3 m
        bp_pad = 2.3
        bldg_solid_bp = np.zeros((self._bp_nx, self._bp_ny, self._bp_nz), dtype=bool)
        bldg_all_bp = np.zeros((self._bp_nx, self._bp_ny, self._bp_nz), dtype=bool)
        for x0, y0, z0, x1, y1, z1, cx, cy, _hx, _hy, r_cyl, _r2, is_cyl, is_porous, _tr in self._bldg_geom_scalar:
            if is_cyl:
                r_p = r_cyl + bp_pad
                m = ((VX - cx) ** 2 + (VY - cy) ** 2 <= r_p * r_p) & (VZ >= z0 - bp_pad) & (VZ <= z1 + bp_pad)
            else:
                m = (
                    (VX >= x0 - bp_pad)
                    & (VX <= x1 + bp_pad)
                    & (VY >= y0 - bp_pad)
                    & (VY <= y1 + bp_pad)
                    & (VZ >= z0 - bp_pad)
                    & (VZ <= z1 + bp_pad)
                )
            bldg_all_bp |= m
            if not is_porous:
                bldg_solid_bp |= m
        self._bp_bldg_solid_flat = np.ascontiguousarray(bldg_solid_bp).ravel()
        self._bp_bldg_all_flat = np.ascontiguousarray(bldg_all_bp).ravel()

        pipe_bp = np.zeros((self._bp_nx, self._bp_ny, self._bp_nz), dtype=bool)
        if self._pipe_seg_a.shape[0] > 0:
            # Conservative capsule-to-voxel-center distance check (half-diag 1.74 + margin 0.5 = 2.25 m)
            p_pad = 2.25
            pts_bp = np.stack([VX, VY, VZ], axis=-1).reshape(-1, 3)
            pipe_flat = np.zeros(pts_bp.shape[0], dtype=bool)
            for s in range(self._pipe_seg_a.shape[0]):
                a = self._pipe_seg_a[s]
                ab = self._pipe_seg_ab[s]
                len2 = float(self._pipe_seg_len2[s])
                r_lim = float(self._pipe_seg_r[s]) + p_pad
                ap = pts_bp - a[None, :]
                t = (ap @ ab) / len2
                np.clip(t, 0.0, 1.0, out=t)
                diff = pts_bp - (a[None, :] + t[:, None] * ab[None, :])
                pipe_flat |= np.sum(diff * diff, axis=1) <= r_lim * r_lim
            pipe_bp = pipe_flat.reshape(self._bp_nx, self._bp_ny, self._bp_nz)
        self._bp_pipe_flat = np.ascontiguousarray(pipe_bp).ravel()

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
        for x0, y0, z0, x1, y1, z1, cx, cy, _hx, _hy, r_cyl, _r2, is_cyl, is_porous, _tr in self._bldg_geom_scalar:
            if skip_porous and is_porous:
                continue
            if pz < z0 or pz > z1 + margin:
                continue
            if is_cyl:
                r = r_cyl + margin
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

    def random_pipe_leak_pos(self, rng: np.random.Generator | None = None) -> Vec3:
        """Sample a uniformly random exposed point along the facility's pipe network."""
        if self._pipe_seg_a.shape[0] == 0:
            return self.default_leak_pos

        gen = rng if rng is not None else np.random.default_rng()
        lengths = np.sqrt(self._pipe_seg_len2)
        total_len = float(lengths.sum())
        if total_len <= 0.0:
            return self.default_leak_pos
        probs = lengths / total_len

        solid_obstacles = [
            geom
            for bldg, geom in zip(self.buildings, self._bldg_geom_scalar)
            if bldg.kind != "pipe_segment"
        ]

        def _too_close_to_obstacle(px: float, py: float, pz: float, margin: float = 1.5) -> bool:
            for x0, y0, z0, x1, y1, z1, cx, cy, _hx, _hy, r_cyl, _r_cyl2, is_cyl, is_porous, _tr in solid_obstacles:
                m = 0.75 if is_porous else margin
                if pz < z0 or pz > z1 + m:
                    continue
                if is_cyl:
                    r = r_cyl + m
                    dx = px - cx
                    dy = py - cy
                    if dx * dx + dy * dy <= r * r:
                        return True
                elif (x0 - m) <= px <= (x1 + m) and (y0 - m) <= py <= (y1 + m):
                    return True
            return False

        fallback: Vec3 = self.default_leak_pos
        for _ in range(96):
            seg_idx = int(gen.choice(len(lengths), p=probs))
            t = float(gen.uniform(0.0, 1.0))
            pt = self._pipe_seg_a[seg_idx] + t * self._pipe_seg_ab[seg_idx]
            px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
            fallback = (px, py, max(1.0, pz))
            if (
                pz < 1.0
                or px < self.bounds_min[0] or px > self.bounds_max[0]
                or py < self.bounds_min[1] or py > self.bounds_max[1]
                or pz > self.bounds_max[2]
            ):
                continue
            if not _too_close_to_obstacle(px, py, pz):
                return (px, py, pz)
        return fallback

    def height_above_surface(self, px: float, py: float, pz: float) -> float:
        """Vertical clearance above ground (z=0) or any building/tank roof directly below."""
        h_agl = pz
        for x0, y0, _z0, x1, y1, z1, cx, cy, _hx, _hy, _r, r2, is_cyl, _ip, _tr in self._bldg_geom_scalar:
            if pz < z1:
                continue
            h_roof = pz - z1
            if h_roof >= h_agl:
                continue
            if is_cyl:
                dx = px - cx
                dy = py - cy
                if dx * dx + dy * dy <= r2:
                    h_agl = h_roof
            elif x0 <= px <= x1 and y0 <= py <= y1:
                h_agl = h_roof
        return 0.05 if h_agl < 0.05 else h_agl

    def _bp_lookup(self, flat_mask: np.ndarray, px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
        """O(1) per-point broad-phase lookup in 60x60x15 (2.0 m) voxel bitmask."""
        ix = ((px + 60.0) * 0.5).astype(np.intp)
        iy = ((py + 60.0) * 0.5).astype(np.intp)
        iz = (pz * 0.5).astype(np.intp)
        in_grid = (ix >= 0) & (ix < 60) & (iy >= 0) & (iy < 60) & (iz >= 0) & (iz < 15)
        out = np.zeros(px.shape[0], dtype=bool)
        if in_grid.any():
            out[in_grid] = flat_mask[(ix[in_grid] * 60 + iy[in_grid]) * 15 + iz[in_grid]]
        return out

    def push_outside_buildings(
        self,
        points_nx3: np.ndarray,
        margin: float = 0.0,
        skip_porous: bool = False,
        copy: bool = True,
        _use_bp: bool = True,
    ) -> np.ndarray:
        """Vectorized push-out: moves points inside buildings/tanks to nearest surface.

        Uses exact cylinder geometry for storage tanks and skips open-truss pipe racks
        when skip_porous=True (so gas puffs flow through pipe racks instead of teleporting).
        """
        pts = points_nx3.copy() if copy else points_nx3
        if pts.shape[0] == 0:
            return pts

        # Fast O(1) voxel broad-phase when margin fits within precomputed 0.5m pad
        if _use_bp and margin <= 0.5 and pts.shape[0] > 8:
            bp_mask = self._bp_bldg_solid_flat if skip_porous else self._bp_bldg_all_flat
            near = self._bp_lookup(bp_mask, pts[:, 0], pts[:, 1], pts[:, 2])
            if not near.any():
                return pts
            cand_idx = np.nonzero(near)[0]
            sub_pts = self.push_outside_buildings(
                pts[cand_idx], margin=margin, skip_porous=skip_porous, copy=False, _use_bp=False
            )
            pts[cand_idx] = sub_pts
            return pts

        px = pts[:, 0]
        py = pts[:, 1]
        pz = pts[:, 2]

        for x0, y0, z0, x1, y1, z1, cx, cy, _hx, _hy, r_cyl, _r2, is_cyl, is_porous, _tr in self._bldg_geom_scalar:
            if skip_porous and is_porous:
                continue

            z_hi = z1 + margin
            if is_cyl:
                r_eff = r_cyl + margin
                dx = px - cx
                dy = py - cy
                r2_dist = dx * dx + dy * dy
                inside = (r2_dist <= r_eff * r_eff) & (pz >= z0) & (pz <= z_hi)
                if not inside.any():
                    continue

                sub = pts[inside]
                r_in = np.sqrt(r2_dist[inside])
                d_rad = r_eff - r_in
                d_top = z_hi - sub[:, 2]
                push_top = d_top < d_rad

                if push_top.any():
                    sub[push_top, 2] = z_hi + 0.05
                push_side = ~push_top
                if push_side.any():
                    sdx = dx[inside][push_side]
                    sdy = dy[inside][push_side]
                    sr = r_in[push_side]
                    safe_r = np.where(sr > 1e-6, sr, 1.0)
                    ux = np.where(sr > 1e-6, sdx / safe_r, 1.0)
                    uy = np.where(sr > 1e-6, sdy / safe_r, 0.0)
                    sub[push_side, 0] = cx + ux * (r_eff + 0.05)
                    sub[push_side, 1] = cy + uy * (r_eff + 0.05)

                pts[inside] = sub
                continue

            lo0 = x0 - margin
            lo1 = y0 - margin
            hi0 = x1 + margin
            hi1 = y1 + margin

            inside = (
                (px >= lo0)
                & (px <= hi0)
                & (py >= lo1)
                & (py <= hi1)
                & (pz >= z0)
                & (pz <= z_hi)
            )
            if not inside.any():
                continue

            sub = pts[inside]
            # Distances to the 5 escapable faces (-x, +x, -y, +y, +z)
            d_xmin = sub[:, 0] - lo0
            d_xmax = hi0 - sub[:, 0]
            d_ymin = sub[:, 1] - lo1
            d_ymax = hi1 - sub[:, 1]
            d_zmax = z_hi - sub[:, 2]

            dists = np.stack([d_xmin, d_xmax, d_ymin, d_ymax, d_zmax], axis=1)
            face = np.argmin(dists, axis=1)

            sub[face == 0, 0] = lo0 - 0.05
            sub[face == 1, 0] = hi0 + 0.05
            sub[face == 2, 1] = lo1 - 0.05
            sub[face == 3, 1] = hi1 + 0.05
            sub[face == 4, 2] = z_hi + 0.05

            pts[inside] = sub

        return pts

    def push_outside_pipes(
        self, points_nx3: np.ndarray, margin: float = 0.05, copy: bool = True
    ) -> np.ndarray:
        """Vectorized continuous capsule push-out so puffs deflect cleanly around pipe runs."""
        if points_nx3.shape[0] == 0 or self._pipe_seg_a.shape[0] == 0:
            return points_nx3.copy() if copy else points_nx3

        pts = points_nx3.copy() if copy else points_nx3
        px = pts[:, 0]
        py = pts[:, 1]
        pz = pts[:, 2]

        # O(1) 3D voxel broad-phase lookup (exact: zero false negatives for margin <= 0.5 m)
        if margin <= 0.5:
            near_any = self._bp_lookup(self._bp_pipe_flat, px, py, pz)
            if not near_any.any():
                return pts
            active_idx = np.nonzero(near_any)[0]
        else:
            ax0, ay0, az0 = self._pipe_all_min
            ax1, ay1, az1 = self._pipe_all_max
            in_all = (
                (pz <= az1 + margin)
                & (pz >= az0 - margin)
                & (px >= ax0 - margin)
                & (px <= ax1 + margin)
                & (py >= ay0 - margin)
                & (py <= ay1 + margin)
            )
            if not in_all.any():
                return pts
            active_idx = np.nonzero(in_all)[0]

        sub = pts[active_idx]  # (K, 3) where K is typically 0..10 puffs near pipes

        # Compute closest point on every segment: (K, S, 3)
        ap = sub[:, None, :] - self._pipe_seg_a[None, :, :]
        t = np.sum(ap * self._pipe_seg_ab[None, :, :], axis=2) / self._pipe_seg_len2[None, :]
        np.clip(t, 0.0, 1.0, out=t)
        closest = self._pipe_seg_a[None, :, :] + t[:, :, None] * self._pipe_seg_ab[None, :, :]
        diff = sub[:, None, :] - closest
        dist2 = np.sum(diff * diff, axis=2)
        min_r = self._pipe_seg_r[None, :] + margin
        hit_mask = dist2 < (min_r * min_r)

        if hit_mask.any():
            # Resolve against the deepest penetrating segment per point
            pen = min_r - np.sqrt(np.maximum(dist2, 1e-12))
            pen = np.where(hit_mask, pen, -1.0)
            best_seg = np.argmax(pen, axis=1)
            best_pen = pen[np.arange(sub.shape[0]), best_seg]
            fix = best_pen > 0.0
            if fix.any():
                idx = np.nonzero(fix)[0]
                s_idx = best_seg[idx]
                v = diff[idx, s_idx]
                v_norm = np.linalg.norm(v, axis=1, keepdims=True)
                n_hat = np.where(v_norm > 1e-6, v / np.maximum(v_norm, 1e-6), np.array([[0.0, 0.0, 1.0]]))
                sub[idx] = closest[idx, s_idx] + n_hat * (self._pipe_seg_r[s_idx, None] + margin + 0.02)
            pts[active_idx] = sub

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

        px = p_from_nx3[:, 0]
        py = p_from_nx3[:, 1]
        pz = p_from_nx3[:, 2]
        dx = p_to[0] - px
        dy = p_to[1] - py
        dz = p_to[2] - pz

        eps = 1e-9
        par_x = np.abs(dx) < eps
        par_y = np.abs(dy) < eps
        par_z = np.abs(dz) < eps
        inv_dx = np.where(par_x, 0.0, 1.0 / np.where(par_x, 1.0, dx))
        inv_dy = np.where(par_y, 0.0, 1.0 / np.where(par_y, 1.0, dy))
        inv_dz = np.where(par_z, 0.0, 1.0 / np.where(par_z, 1.0, dz))

        has_par_x = bool(par_x.any())
        has_par_y = bool(par_y.any())
        has_par_z = bool(par_z.any())

        for x0, y0, z0, x1, y1, z1, cx, cy, _hx, _hy, _r, r2, is_cyl, _ip, bldg_trans in self._bldg_geom_scalar:
            # Z-slab interval [tz_enter, tz_exit]
            tz1 = (z0 - pz) * inv_dz
            tz2 = (z1 - pz) * inv_dz
            tz_enter = np.minimum(tz1, tz2)
            tz_exit = np.maximum(tz1, tz2)
            if has_par_z:
                in_z = (pz >= z0) & (pz <= z1)
                tz_enter = np.where(par_z, np.where(in_z, -1e9, 1e9), tz_enter)
                tz_exit = np.where(par_z, np.where(in_z, 1e9, -1e9), tz_exit)

            if is_cyl:
                ox = px - cx
                oy = py - cy
                a = dx * dx + dy * dy
                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - r2
                disc = b * b - 4.0 * a * c

                vert_ray = a < 1e-9
                inside_cyl_xy = c <= 0.0
                safe_a = np.where(vert_ray, 1.0, a)
                sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
                inv_2a = 0.5 / safe_a
                tc_enter = np.where(
                    vert_ray,
                    np.where(inside_cyl_xy, -1e9, 1e9),
                    np.where(disc >= 0.0, (-b - sqrt_disc) * inv_2a, 1e9),
                )
                tc_exit = np.where(
                    vert_ray,
                    np.where(inside_cyl_xy, 1e9, -1e9),
                    np.where(disc >= 0.0, (-b + sqrt_disc) * inv_2a, -1e9),
                )

                t_enter = np.maximum(tc_enter, tz_enter)
                t_exit = np.minimum(tc_exit, tz_exit)
                hit = (t_exit >= t_enter) & (t_exit > 0.04) & (t_enter < 0.96)
                if hit.any():
                    trans[hit] *= bldg_trans
                continue

            tx1 = (x0 - px) * inv_dx
            tx2 = (x1 - px) * inv_dx
            tx_enter = np.minimum(tx1, tx2)
            tx_exit = np.maximum(tx1, tx2)
            if has_par_x:
                in_x = (px >= x0) & (px <= x1)
                tx_enter = np.where(par_x, np.where(in_x, -1e9, 1e9), tx_enter)
                tx_exit = np.where(par_x, np.where(in_x, 1e9, -1e9), tx_exit)

            ty1 = (y0 - py) * inv_dy
            ty2 = (y1 - py) * inv_dy
            ty_enter = np.minimum(ty1, ty2)
            ty_exit = np.maximum(ty1, ty2)
            if has_par_y:
                in_y = (py >= y0) & (py <= y1)
                ty_enter = np.where(par_y, np.where(in_y, -1e9, 1e9), ty_enter)
                ty_exit = np.where(par_y, np.where(in_y, 1e9, -1e9), ty_exit)

            t_enter = np.maximum(np.maximum(tx_enter, ty_enter), tz_enter)
            t_exit = np.minimum(np.minimum(tx_exit, ty_exit), tz_exit)

            # Strictly interior segment crossing (avoids self-occluding surface-grazing puffs)
            hit = (t_exit >= t_enter) & (t_exit > 0.04) & (t_enter < 0.96)
            if hit.any():
                trans[hit] *= bldg_trans

        return trans

    def is_in_wind_shadow(self, pos: Vec3, wind_vec: Vec2) -> bool:
        """Return True if pos lies in the downwind wake cavity of any building."""
        wx, wy = wind_vec
        speed = math.hypot(wx, wy)
        if speed < 1e-3:
            return False

        ux, uy = wx / speed, wy / speed
        abs_ux = abs(ux)
        abs_uy = abs(uy)
        px, py, pz = pos

        for _x0, _y0, _z0, _x1, _y1, height, cx, cy, hx, hy, r_cyl, _r2, is_cyl, _ip, _tr in self._bldg_geom_scalar:
            if pz > height * 1.15:
                continue

            dx = px - cx
            dy = py - cy
            along = dx * ux + dy * uy
            cross = abs(-dx * uy + dy * ux)

            if is_cyl:
                half_along = r_cyl
                half_cross = r_cyl
            else:
                half_along = hx * abs_ux + hy * abs_uy
                half_cross = hx * abs_uy + hy * abs_ux

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
