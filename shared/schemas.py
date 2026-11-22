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
    kind: Literal["building", "storage_tank", "pipe_rack", "pipe_segment"]
    min_xyz: Vec3
    max_xyz: Vec3
    model: Optional[Literal[
        "compressor",
        "fractionator",
        "horton_sphere",
        "metering_skid",
        "receiver_skid",
        "cryogenic_vessel",
        "transfer_gantry",
        "compressor_train",
        "cooling_cell",
        "hyperbolic_tower",
        "exhaust_stack",
        "pump_house",
    ]] = None


class FacilityAsset(BaseModel):
    id: str
    asset_type: Literal["valve", "flange", "storage_tank", "pipe_junction"]
    pos: Vec3
    age_years: float


class PipeRun(BaseModel):
    id: str
    from_id: Optional[str] = None
    to_id: Optional[str] = None
    points: list[Vec3]
    radius: float = 0.28
    name: Optional[str] = None
    service: Optional[str] = None
    pressureBar: Optional[float] = None
    color: Optional[int] = None
    pressure_bar: Optional[float] = None
    material: Optional[Literal["gas", "steel", "coolant"]] = None


class LeakSource(BaseModel):
    id: str
    pos: Vec3
    emission_rate: float = 0.02  # kg/s (20 g/s = 72 kg/h industrial CH4 leak)
    buoyancy_vz: Optional[float] = None  # legacy explicit rise override (m/s); None models methane


class GridPointMeta(BaseModel):
    idx: int
    pos: Vec3
    nearest_asset_type: Literal["open_air", "flange", "valve", "storage_tank", "wall"]
    asset_age_years: float
    in_wind_shadow: bool
    upwind_pipe_score: float


class WorldInitPayload(BaseModel):
    type: Literal["world_init"] = "world_init"
    map_id: str = "refinery"
    pipe_runs: list[PipeRun] = Field(default_factory=list)
    bounds_min: Vec3 = (-60.0, -60.0, 0.0)
    bounds_max: Vec3 = (60.0, 60.0, 30.0)
    buildings: list[BuildingBox]
    assets: list[FacilityAsset]
    pipes: list[PipeRun] = []
    grid_coords: list[Vec3]  # N candidate 3D points (~400-600) for TabPFN belief cloud
    layout_seed: Optional[int] = None
    theme: Optional[str] = None


class WindFieldPayload(BaseModel):
    """Coarse 3D mass-consistent wind velocity grid for frontend flow visualization."""

    type: Literal["wind_field"] = "wind_field"
    bounds_min: Vec3
    bounds_max: Vec3
    dims: tuple[int, int, int]  # (nx, ny, nz)
    base_speed: float
    # Flattened C-order (nx * ny * nz * 3) velocity vectors [vx, vy, vz, ...]
    vectors: list[float]


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
    wind_measured: tuple[Optional[float], Optional[float]] | None = None
    battery_pct: float
    samples_taken: int
    dist_to_nearest_leak: float
    status: Literal["searching", "tracking", "locked", "depleted"]


class PuffState(BaseModel):
    pos: Vec3
    radius_xy: float
    radius_z: float
    opacity: float


class MissionReport(BaseModel):
    drone_id: str = "tabpfn"
    elapsed_sim_seconds: float
    elapsed_wall_seconds: float
    estimated_source: Vec3
    model_score: Optional[float] = None
    regional_score_mass: Optional[float] = None
    confidence_note: str = "Uncalibrated model score; not a probability that the location is correct."
    simulator_confirmed: bool = True
    estimated_error_m: float
    measurement_count: int
    ppm_at_detection: float
    estimate_age_seconds: float


class SimTickPayload(BaseModel):
    type: Literal["sim_tick"] = "sim_tick"
    sim_time: float
    wind_vec: Vec2
    leaks: list[LeakSource]  # Hidden on frontend unless "Reveal True Plume" is toggled
    drones: list[DroneTickState]
    puffs: list[PuffState]  # Capped at <= 400 active puffs for 60 FPS rendering
    sensor_dropout_active: bool
    paused: bool = False
    setup_mode: bool = False
    wind_setpoint: Optional[tuple[float, float]] = None  # direction in degrees, speed in m/s
    mission_report: Optional[MissionReport] = None


class TabPFNMapPayload(BaseModel):
    sim_time: float
    scores: list[Optional[float]]  # World grid order. None means the model did not score the cell.
    estimate: Optional[Vec3] = None


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
    hazard_observed_at: Optional[float] = None
    hazard_forecast_time: Optional[float] = None
    source_prediction_backend: Optional[str] = None
    hazard_prediction_backend: Optional[str] = None
    navigation_basis: str = "unspecified"
    prediction_decisions: int = 0
    route: list[Vec3] = []
    inference_seconds: float = 0.0
    measurement_count: int = 0
    exposure_ppm_seconds: float = 0.0
    score_kind: str = "legacy_quantile"
    source_model_score: Optional[float] = None
    source_evidence: Optional[bool] = None
    regional_score_mass: Optional[float] = None
    tabpfn_search_map: Optional[TabPFNMapPayload] = None
    tabpfn_source_map: Optional[TabPFNMapPayload] = None


class BackendStatusPayload(BaseModel):
    type: Literal["backend_status"] = "backend_status"
    sim_time: float
    mode: str
    mission_state: Literal["searching", "paused", "depleted", "no_leak", "found"] = "searching"
    mission_report: Optional[MissionReport] = None
    active_leaks: int = 0
    battery_pct: float = 100.0
    state: Literal["pending", "ready", "error", "mock"]
    error: Optional[str] = None
    inference_seconds: float = 0.0
    measurement_count: int = 0
    exposure_ppm_seconds: float = 0.0
    paused: bool = False
    search_phase: str = "search"
    completed_probes: int = 0
    source_features: int = 0
    hazard_features: int = 0
    navigation_basis: str = "unspecified"
    prediction_decisions: int = 0
    source_prediction_calls: int = 0
    hazard_prediction_calls: int = 0
    adaptive_wind: bool = False
    wind_replans: int = 0
    plan_state: str = 'preparing'
    plan_revision: int = 0
    search_candidate_count: int = 0
    planned_route: list[Vec3] = []


class JudgeCommand(BaseModel):
    type: Literal["judge_cmd"] = "judge_cmd"
    action: Literal[
        "set_wind",
        "place_leak",
        "clear_leaks",
        "toggle_dropout",
        "reset_race",
        "start_mission",
        "select_map",
        "randomize_layout",
        "pause",
        "step",
    ]
    map_id: Optional[str] = None
    wind_angle_deg: Optional[float] = None
    wind_speed: Optional[float] = None
    leak_pos: Optional[Vec3] = None
    replace_existing: bool = False
    keep_leaks: bool = False
    dropout_enabled: Optional[bool] = None
    seed: Optional[int] = Field(default=None, ge=0)
