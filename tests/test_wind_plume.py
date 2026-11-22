import math

from agents.wind_plume import WindPlumeAgent, WindPlumeController
from shared.schemas import LeakSource
from sim.engine import SimEngine


def test_wind_plume_baseline_commits_to_tracking_legs_and_confirms_leak():
    agent = WindPlumeAgent(synchronous=True)
    try:
        engine = SimEngine({"tabpfn": agent}, seed=42, layout_seed=42, responsive_sensor=True)
        previous_goal = None
        tracking_legs = 0
        for _ in range(round(180 / engine.DT)):
            tick = engine.step(include_puffs=False)
            if agent.search_phase == "track":
                goal = agent.goal
                if previous_goal is not None and math.dist(previous_goal, goal) >= 3:
                    assert math.dist(tick.drones[0].pos, previous_goal) <= 3.5
                    tracking_legs += 1
                previous_goal = goal
            else:
                previous_goal = None
            if tick.mission_report is not None:
                break
        assert tracking_legs > 0
        assert tick.mission_report is not None
        assert tick.mission_report.model_score is None
        assert tick.drones[0].wind_measured == agent.history[-1].wind_measured
    finally:
        agent.close()


def test_elevated_leak_is_confirmed(monkeypatch):
    # Use the original measurement dwell to reproduce the elevated search.
    monkeypatch.setattr(WindPlumeController, "DWELL_SECONDS", 3.5)
    monkeypatch.setattr(WindPlumeController, "TRANSECT_ARRIVAL_RADIUS", 0.0)
    agent = WindPlumeAgent(synchronous=True)
    try:
        engine = SimEngine({"tabpfn": agent}, seed=153, responsive_sensor=True)
        engine.airflow.set_wind(115.0, 3.0)
        engine.reset_race(
            [
                LeakSource(
                    id="elevated",
                    pos=(15.0, 8.0, 14.0),
                    emission_rate=1.0,
                    buoyancy_vz=0.25,
                )
            ]
        )
        for _ in range(round(180 / engine.DT)):
            tick = engine.step(include_puffs=False)
            if tick.mission_report is not None:
                break
        assert tick.mission_report is not None
    finally:
        agent.close()
