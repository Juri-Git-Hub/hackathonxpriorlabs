"""Frozen cross-module data contracts (Single Source of Truth).

Coordinate System Convention:
- Python Physics & Agents: Right-handed Z-up, in meters.
  Facility bounds: x in [-60, 60], y in [-60, 60], z in [0, 30] (ground plane at z = 0).
- Three.js Frontend (Y-up): Convert strictly at the rendering boundary using:
  three_pos = (sim_x, sim_z, -sim_y). Never convert coordinates in Python.

Do not modify field names without updating all consumers.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


class BuildingBox(BaseModel):
    id: str
    kind: Literal["building", "storage_tank", "pipe_rack"]
    min_xyz: Vec3
    max_xyz: Vec3


class FacilityAsset(BaseModel):
    id: str
    asset_type: Literal["valve", "flange", "storage_tank", "pipe_junction"]
    pos: Vec3
    age_years: float


class LeakSource(BaseModel):
    id: str
    pos: Vec3
    emission_rate: float = 1.0  # kg/s equivalent scale
    buoyancy_vz: float = 0.25   # >0 rising (methane), <0 heavy gas


class GridPointMeta(BaseModel):
    idx: int
    pos: Vec3
    nearest_asset_type: Literal["open_air", "flange", "valve", "storage_tank", "wall"]
    asset_age_years: float
    in_wind_shadow: bool
    upwind_pipe_score: float


class WorldInitPayload(BaseModel):
    type: Literal["world_init"] = "world_init"
    bounds_min: Vec3 = (-60.0, -60.0, 0.0)
    bounds_max: Vec3 = (60.0, 60.0, 30.0)
    buildings: list[BuildingBox]
    assets: list[FacilityAsset]
    grid_coords: list[Vec3]  # N candidate 3D points (~800) for TabPFN belief cloud


class DroneObservation(BaseModel):
    sim_time: float
    drone_id: str
    pos: Vec3
    vel: Vec3
    wind_measured: tuple[Optional[float], Optional[float]]  # Can be None during NaN fault
    ppm_measured: float
    temp_delta: Optional[float]  # Can be None during NaN fault
    nearest_asset_type: Literal["open_air", "flange", "valve", "storage_tank", "wall"]
    asset_age_years: float
    in_wind_shadow: bool
    upwind_pipe_score: float
    battery_pct: float


class DroneTickState(BaseModel):
    id: str  # "tabpfn" | "gradient" | "lawnmower" | "manual"
    pos: Vec3
    rot_rpy: Vec3  # roll, pitch, yaw in radians (derived from thrust vector)
    target_pos: Vec3
    ppm: float
    battery_pct: float
    samples_taken: int
    dist_to_nearest_leak: float
    status: Literal["searching", "tracking", "locked", "depleted"]


class PuffState(BaseModel):
    pos: Vec3
    radius_xy: float
    radius_z: float
    opacity: float


class SimTickPayload(BaseModel):
    type: Literal["sim_tick"] = "sim_tick"
    sim_time: float
    wind_vec: Vec2
    leaks: list[LeakSource]  # Hidden on frontend unless "Reveal True Plume" is toggled
    drones: list[DroneTickState]
    puffs: list[PuffState]  # Capped at <= 400 active puffs for 60 FPS rendering
    sensor_dropout_active: bool
    paused: bool = False


class BeliefUpdatePayload(BaseModel):
    type: Literal["belief_update"] = "belief_update"
    sim_time: float
    drone_id: str
    q50: list[float]          # Length N matching WorldInitPayload.grid_coords
    uncertainty: list[float]  # (q90 - q10), length N
    estimated_source: Vec3
    next_waypoint: Vec3
    source_scores: list[float] = []
    hazard_probabilities: list[float] = []
    route: list[Vec3] = []
    inference_seconds: float = 0.0
    measurement_count: int = 0
    exposure_ppm_seconds: float = 0.0
    score_kind: str = "legacy_quantile"


class BackendStatusPayload(BaseModel):
    type: Literal["backend_status"] = "backend_status"
    sim_time: float
    mode: str
    mission_state: Literal["searching", "paused", "depleted", "no_leak"] = "searching"
    active_leaks: int = 0
    battery_pct: float = 100.0
    state: Literal["pending", "ready", "error", "mock"]
    error: Optional[str] = None
    inference_seconds: float = 0.0
    measurement_count: int = 0
    exposure_ppm_seconds: float = 0.0
    paused: bool = False


class JudgeCommand(BaseModel):
    type: Literal["judge_cmd"] = "judge_cmd"
    action: Literal[
        "set_wind",
        "place_leak",
        "clear_leaks",
        "toggle_dropout",
        "reset_race",
        "pause",
        "step",
    ]
    wind_angle_deg: Optional[float] = None
    wind_speed: Optional[float] = None
    leak_pos: Optional[Vec3] = None
    dropout_enabled: Optional[bool] = None
    seed: Optional[int] = Field(default=None, ge=0)
