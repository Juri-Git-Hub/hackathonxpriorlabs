import math

from agents.wind_plume import WindPlumeAgent, WindPlumeController
from server.app import build_agents
from sim.engine import SimEngine


def test_default_wind_policy_commits_to_tracking_legs_and_confirms_leak():
    agent = build_agents()["tabpfn"]
    assert isinstance(agent, WindPlumeAgent)
    agent.synchronous = True
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
    from shared.schemas import LeakSource

    # Use the original measurement dwell to reproduce the elevated search.
    monkeypatch.setattr(WindPlumeController, "DWELL_SECONDS", 3.5)
    monkeypatch.setattr(WindPlumeController, "TRANSECT_ARRIVAL_RADIUS", 0.)
    agent = WindPlumeAgent(synchronous=True)
    try:
        engine = SimEngine({"tabpfn": agent}, seed=153, responsive_sensor=True)
        engine.airflow.set_wind(115.0, 3.0)
        engine.reset_race([LeakSource(
            id="elevated",
            pos=(15.0, 8.0, 14.0),
            emission_rate=1.0,
            buoyancy_vz=.25,
        )])
        for _ in range(round(180 / engine.DT)):
            tick = engine.step(include_puffs=False)
            if tick.mission_report is not None:
                break
        assert tick.mission_report is not None
    finally:
        agent.close()


def test_mvp_replays_same_confirmed_flight_after_reset():
    from server.app import create_app
    from shared.schemas import LeakSource

    engine = create_app(mvp=True).state.engine
    agent = engine.agents["tabpfn"]
    try:
        assert agent.synchronous
        from server.app import MVP_LAYOUT_SEEDS
        assert engine.world.layout_seed in MVP_LAYOUT_SEEDS
        flange = next(a for a in engine.world.assets if a.id == "flange_01")
        fixed_leak = [LeakSource(id="leak_01", pos=flange.pos, emission_rate=0.02)]
        reports = []
        for _ in range(2):
            engine.reset_race(fixed_leak)
            assert len(engine.leaks) == 1
            for _ in range(round(150 / engine.DT)):
                engine.step(include_puffs=False)
                if engine.mission_report is not None:
                    break
            report = engine.mission_report
            assert report is not None
            reports.append((report.elapsed_sim_seconds, report.estimated_source))
        assert reports[0] == reports[1]
    finally:
        agent.close()


def test_mvp_randomize_layout_changes_to_another_verified_refinery():
    from fastapi.testclient import TestClient

    from server.app import MVP_LAYOUT_SEEDS, create_app

    with TestClient(create_app(mvp=True, randomized_layout=True)) as client:
        with client.websocket_connect("/ws") as websocket:
            first = websocket.receive_json()
            assert first["map_id"] == "refinery"
            assert first["layout_seed"] in MVP_LAYOUT_SEEDS
            websocket.send_json({"type": "judge_cmd", "action": "randomize_layout"})
            while True:
                message = websocket.receive_json()
                if message["type"] == "world_init":
                    break
            assert message["map_id"] == first["map_id"]
            assert message["layout_seed"] in MVP_LAYOUT_SEEDS
            assert message["layout_seed"] != first["layout_seed"]
