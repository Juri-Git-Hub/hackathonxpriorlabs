"""Schema serialization and FastAPI WebSocket contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from shared.schemas import (
    BeliefUpdatePayload,
    JudgeCommand,
    SimTickPayload,
    WindFieldPayload,
    WorldInitPayload,
)


def test_judge_command_roundtrip() -> None:
    cmd = JudgeCommand(
        action="place_leak",
        leak_pos=(12.0, -5.0, 4.0),
    )
    raw = cmd.model_dump_json()
    parsed = JudgeCommand.model_validate_json(raw)
    assert parsed.type == "judge_cmd"
    assert parsed.action == "place_leak"
    assert parsed.leak_pos == (12.0, -5.0, 4.0)


def test_websocket_emits_world_init_sim_tick_and_belief_and_handles_commands() -> None:
    app = create_app(mock_agents=True, seed=42)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["mock_agents"] is True

        with client.websocket_connect("/ws") as ws:
            # 1. First message must be WorldInitPayload
            first_raw = ws.receive_text()
            world_init = WorldInitPayload.model_validate_json(first_raw)
            assert world_init.type == "world_init"
            assert len(world_init.buildings) >= 4
            assert len(world_init.assets) >= 12
            assert 400 <= len(world_init.grid_coords) <= 1000

            # 2. Send judge commands
            ws.send_text(
                JudgeCommand(
                    action="set_wind", wind_angle_deg=90.0, wind_speed=4.5
                ).model_dump_json()
            )
            ws.send_text(
                JudgeCommand(
                    action="toggle_dropout", dropout_enabled=True
                ).model_dump_json()
            )
            ws.send_text(
                JudgeCommand(
                    action="place_leak", leak_pos=(15.0, 10.0, 4.0)
                ).model_dump_json()
            )
            ws.send_text(JudgeCommand(action="start_mission").model_dump_json())

            # 3. Collect messages until we see WindFieldPayload, SimTickPayload, and BeliefUpdatePayload
            seen_ticks = 0
            seen_belief = False
            seen_wind_field = False
            dropout_applied = False
            leak_added = False

            for _ in range(60):
                raw = ws.receive_text()
                if '"wind_field"' in raw:
                    wf = WindFieldPayload.model_validate_json(raw)
                    nx, ny, nz = wf.dims
                    assert len(wf.vectors) == nx * ny * nz * 3
                    seen_wind_field = True
                elif '"sim_tick"' in raw:
                    tick = SimTickPayload.model_validate_json(raw)
                    seen_ticks += 1
                    assert [d.id for d in tick.drones] == ['tabpfn']
                    assert len(tick.puffs) <= 400
                    if tick.sensor_dropout_active:
                        dropout_applied = True
                    if len(tick.leaks) >= 2:
                        leak_added = True
                elif '"belief_update"' in raw:
                    belief = BeliefUpdatePayload.model_validate_json(raw)
                    assert len(belief.q50) == len(world_init.grid_coords)
                    assert len(belief.uncertainty) == len(world_init.grid_coords)
                    seen_belief = True

                if seen_ticks >= 5 and seen_belief and seen_wind_field and dropout_applied and leak_added:
                    break

            assert seen_ticks >= 5
            assert seen_belief
            assert seen_wind_field
            assert dropout_applied
            assert leak_added

def test_live_status_and_manual_step_contract():
    from shared.schemas import BackendStatusPayload
    app=create_app(mock_agents=True,seed=42)
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            WorldInitPayload.model_validate_json(ws.receive_text())
            status=BackendStatusPayload.model_validate_json(ws.receive_text())
            assert status.state=='mock'
            engine=app.state.engine
            assert engine.get_tick_payload().setup_mode and engine.sim_time == 0
            with pytest.raises(ValueError, match="Start the mission"):
                engine.apply_command(JudgeCommand(action='step'))
            engine.apply_command(JudgeCommand(action='set_wind', wind_angle_deg=90, wind_speed=4.5))
            engine.apply_command(JudgeCommand(action='start_mission'))
            assert engine.get_tick_payload().wind_setpoint == pytest.approx((90, 4.5))
            engine.apply_command(JudgeCommand(action='pause'))
            for cmd in (
                JudgeCommand(action='set_wind', wind_speed=3),
                JudgeCommand(action='select_map', map_id='cooling_works'),
                JudgeCommand(action='randomize_layout'),
                JudgeCommand(action='toggle_dropout'),
                JudgeCommand(action='clear_leaks'),
            ):
                with pytest.raises(ValueError, match="Mission settings are locked"):
                    engine.apply_command(cmd)
            t=engine.sim_time
            engine.apply_command(JudgeCommand(action='step'))
            assert engine.paused and abs(engine.sim_time-t-.5)<1e-6
            engine.apply_command(JudgeCommand(action='reset_race',seed=99))
            assert engine.sim_time==0 and engine._seed==99 and engine.setup_mode
            engine.apply_command(JudgeCommand(action='set_wind', wind_speed=3))
            assert engine.get_tick_payload().wind_setpoint == pytest.approx((90, 3))


def test_extended_belief_payload_preserves_legacy_fields():
    b=BeliefUpdatePayload(sim_time=1,drone_id='tabpfn',q50=[.2,.8],uncertainty=[.1,.1],estimated_source=(1,2,3),next_waypoint=(4,5,6),source_scores=[.2,.8],hazard_probabilities=[.4,.6],route=[(0,0,3),(4,5,6)],inference_seconds=.5,measurement_count=3,score_kind='source_score')
    parsed=BeliefUpdatePayload.model_validate_json(b.model_dump_json())
    assert parsed.route[-1]==(4,5,6) and parsed.hazard_probabilities==[.4,.6]

def test_reset_rejects_negative_seed():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):JudgeCommand(action='reset_race',seed=-1)

def test_demo_has_only_tabpfn_and_exposes_mission_blockers():
    from shared.schemas import BackendStatusPayload
    app=create_app(mock_agents=True)
    assert list(app.state.engine.drones)==['tabpfn']
    app.state.engine.apply_command(JudgeCommand(action='clear_leaks'))
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            WorldInitPayload.model_validate_json(ws.receive_text())
            s=BackendStatusPayload.model_validate_json(ws.receive_text())
            assert s.active_leaks==0 and s.mission_state=='no_leak'

def test_completed_report_is_sent_on_reconnect_and_cleared_by_reset():
    import numpy as np
    import json
    app=create_app(mock_agents=True)
    e=app.state.engine
    with e._lock:
        p=e.leaks[0].pos;e.drones['tabpfn'].pos=np.array(p)
        e._latest_agent_beliefs['tabpfn']=BeliefUpdatePayload(sim_time=1,drone_id='tabpfn',q50=[1],uncertainty=[0],estimated_source=p,next_waypoint=p)
        steps=round(e.LOCK_SUSTAIN_S/e.OBS_INTERVAL)
        for i in range(steps):e.sim_time=1+i*e.OBS_INTERVAL;e._check_completion('tabpfn',30)
        expected_t=1+(steps-1)*e.OBS_INTERVAL
    with TestClient(app) as client:
        for _ in range(2):
            with client.websocket_connect('/ws') as ws:
                ws.receive_text()
                status=json.loads(ws.receive_text())
                assert status['mission_state']=='found'
                assert abs(status['mission_report']['elapsed_sim_seconds']-expected_t)<1e-6
        with client.websocket_connect('/ws') as ws:
            ws.receive_text();ws.receive_text()
            ws.send_text(JudgeCommand(action='reset_race').model_dump_json())
            for _ in range(30):
                msg=json.loads(ws.receive_text())
                if msg['type']=='backend_status' and msg['mission_state']!='found':
                    assert msg['mission_report'] is None;break
            else:raise AssertionError('Reset did not clear completion report')


def test_select_map_broadcasts_new_geometry_before_the_next_mission_tick():
    app = create_app(mock_agents=True, layout_seed=42)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        refinery = WorldInitPayload.model_validate(ws.receive_json())
        for map_id, leak_height in (
            ("pipe_loop", 21), ("tank_terminal", 4), ("compressor_yard", 5),
            ("cooling_works", 13), ("refinery", 4),
        ):
            ws.send_json(JudgeCommand(action="select_map", map_id=map_id).model_dump())
            for _ in range(30):
                message = ws.receive_json()
                if message["type"] == "world_init":
                    break
            else:
                raise AssertionError(f"No geometry broadcast for {map_id}")

            layout = WorldInitPayload.model_validate(message)
            assert layout.map_id == map_id
            if map_id == "refinery":
                assert layout == refinery
            else:
                assert layout.layout_seed is not None
                assert layout.grid_coords != refinery.grid_coords
                assert any(box.kind == "pipe_segment" for box in layout.buildings)
                assert layout.pipe_runs

            for _ in range(30):
                message = ws.receive_json()
                if message["type"] == "sim_tick":
                    tick = SimTickPayload.model_validate(message)
                    assert tick.sim_time < 1
                    assert len(tick.leaks) == 1
                    world = app.state.engine.world
                    leak_pt = np.array(tick.leaks[0].pos, dtype=float)
                    seg_a = world._pipe_seg_a
                    seg_ab = world._pipe_seg_ab
                    t_proj = np.clip(
                        np.sum((leak_pt - seg_a) * seg_ab, axis=1) / world._pipe_seg_len2,
                        0.0,
                        1.0,
                    )
                    closest = seg_a + t_proj[:, None] * seg_ab
                    min_dist = float(np.min(np.linalg.norm(closest - leak_pt, axis=1)))
                    assert min_dist < 1e-6
                    break
            else:
                raise AssertionError(f"No mission tick for {map_id}")


def test_set_comparison_drones_toggles_baselines_live():
    app = create_app(mock_agents=True, layout_seed=42)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_text()  # world_init
        ws.receive_text()  # backend_status

        # Enable all three comparison baselines while in setup mode
        ws.send_json(
            {
                "type": "judge_cmd",
                "action": "set_comparison_drones",
                "comparison_drones": ["wind_plume", "gradient", "lawnmower"],
            }
        )
        for _ in range(30):
            msg = ws.receive_json()
            if msg["type"] == "sim_tick" and len(msg["drones"]) == 4:
                ids = {d["id"] for d in msg["drones"]}
                assert ids == {"tabpfn", "wind_plume", "gradient", "lawnmower"}
                break
        else:
            raise AssertionError("Comparison drones were not activated in sim_tick")

        # Start mission, then toggle down to only gradient + tabpfn live in flight
        ws.send_json({"type": "judge_cmd", "action": "start_mission"})
        ws.send_json(
            {
                "type": "judge_cmd",
                "action": "set_comparison_drones",
                "comparison_drones": ["gradient"],
            }
        )
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "sim_tick" and len(msg["drones"]) == 2:
                ids = {d["id"] for d in msg["drones"]}
                assert ids == {"tabpfn", "gradient"}
                break
        else:
            raise AssertionError("Comparison drones were not updated live during flight")

        # Turn off all baselines
        ws.send_json(
            {
                "type": "judge_cmd",
                "action": "set_comparison_drones",
                "comparison_drones": [],
            }
        )
        for _ in range(40):
            msg = ws.receive_json()
            if msg["type"] == "sim_tick" and len(msg["drones"]) == 1:
                assert msg["drones"][0]["id"] == "tabpfn"
                break
        else:
            raise AssertionError("Comparison drones were not removed")
