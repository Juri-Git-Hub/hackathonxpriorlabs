"""Standardized evaluation & demo scenarios shared by headless evals and live CLI control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shared.schemas import JudgeCommand, LeakSource


@dataclass(frozen=True)
class ScheduledEvent:
    """A JudgeCommand fired at a specific sim_time during a scenario."""

    at_time: float
    command: JudgeCommand


@dataclass(frozen=True)
class ScenarioSpec:
    """Declarative scenario configuration runnable in both headless and live modes."""

    name: str
    description: str
    wind_angle_deg: float = 15.0
    wind_speed: float = 3.5
    leaks: list[LeakSource] = field(default_factory=list)
    dropout_enabled: bool = False
    events: list[ScheduledEvent] = field(default_factory=list)
    recommended_duration_s: float = 90.0


SCENARIOS: dict[str, ScenarioSpec] = {
    "open_corridor": ScenarioSpec(
        name="open_corridor",
        description="Single methane leak on aged flange_01 in central pipe corridor with steady wind.",
        wind_angle_deg=15.0,
        wind_speed=3.5,
        leaks=[
            LeakSource(id="leak_01", pos=(-6.0, -7.2, 4.0), emission_rate=1.0, buoyancy_vz=0.25),
        ],
    ),
    "wake_trap": ScenarioSpec(
        name="wake_trap",
        description="Leak hidden on the lee side of tank_north; strong wake cavity traps gas into false maxima.",
        wind_angle_deg=0.0,
        wind_speed=4.2,
        leaks=[
            LeakSource(id="leak_wake", pos=(11.5, 24.0, 6.0), emission_rate=1.1, buoyancy_vz=0.08),
        ],
    ),
    "wind_shift_90": ScenarioSpec(
        name="wind_shift_90",
        description="Wind rotates 90 degrees (+X -> +Y) mid-race at t=20s, invalidating stale plume paths.",
        wind_angle_deg=0.0,
        wind_speed=4.0,
        leaks=[
            LeakSource(id="leak_01", pos=(-6.0, -7.2, 4.0), emission_rate=1.0, buoyancy_vz=0.20),
        ],
        events=[
            ScheduledEvent(
                at_time=20.0,
                command=JudgeCommand(action="set_wind", wind_angle_deg=90.0, wind_speed=4.5),
            ),
        ],
    ),
    "multi_leak": ScenarioSpec(
        name="multi_leak",
        description="Two simultaneous leaks on opposite sides of the refinery (pipe rack + east storage tank).",
        wind_angle_deg=25.0,
        wind_speed=3.8,
        leaks=[
            LeakSource(id="leak_01", pos=(-17.5, -2.0, 3.5), emission_rate=0.9, buoyancy_vz=0.25),
            LeakSource(id="leak_02", pos=(20.5, -7.0, 6.0), emission_rate=1.0, buoyancy_vz=0.20),
        ],
    ),
    "sensor_dropouts": ScenarioSpec(
        name="sensor_dropouts",
        description="30% NaN dropouts on wind_measured and temp_delta throughout the race.",
        wind_angle_deg=15.0,
        wind_speed=3.5,
        leaks=[
            LeakSource(id="leak_01", pos=(-6.0, -7.2, 4.0), emission_rate=1.0, buoyancy_vz=0.25),
        ],
        dropout_enabled=True,
    ),
    "heavy_gas": ScenarioSpec(
        name="heavy_gas",
        description="Dense gas leak (H2S/LPG-like, negative buoyancy) sinking toward ground level.",
        wind_angle_deg=-20.0,
        wind_speed=3.0,
        leaks=[
            LeakSource(id="leak_heavy", pos=(4.0, -7.2, 5.0), emission_rate=1.1, buoyancy_vz=-0.22),
        ],
    ),
    "high_wind_stress": ScenarioSpec(
        name="high_wind_stress",
        description="Strong 7.5 m/s crosswind testing 6-DOF attitude trim, drag, and rapid plume shearing.",
        wind_angle_deg=40.0,
        wind_speed=7.5,
        leaks=[
            LeakSource(id="leak_01", pos=(-10.0, 5.5, 4.0), emission_rate=1.3, buoyancy_vz=0.15),
        ],
    ),
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Available: {valid}")
    return SCENARIOS[name]


def list_scenario_names() -> list[str]:
    return list(SCENARIOS.keys())
