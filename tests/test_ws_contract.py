"""Schema serialization and FastAPI WebSocket contract tests."""

from __future__ import annotations

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
            engine.apply_command(JudgeCommand(action='pause'))
            t=engine.sim_time
            engine.apply_command(JudgeCommand(action='step'))
            assert engine.paused and abs(engine.sim_time-t-.5)<1e-6
            engine.apply_command(JudgeCommand(action='reset_race',seed=99))
            assert engine.sim_time==0 and engine._seed==99


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
        for i in range(4):e.sim_time=1+i*.5;e._check_completion('tabpfn',30)
    with TestClient(app) as client:
        for _ in range(2):
            with client.websocket_connect('/ws') as ws:
                ws.receive_text()
                status=json.loads(ws.receive_text())
                assert status['mission_state']=='found'
                assert status['mission_report']['elapsed_sim_seconds']==2.5
        with client.websocket_connect('/ws') as ws:
            ws.receive_text();ws.receive_text()
            ws.send_text(JudgeCommand(action='reset_race').model_dump_json())
            for _ in range(30):
                msg=json.loads(ws.receive_text())
                if msg['type']=='backend_status' and msg['mission_state']!='found':
                    assert msg['mission_report'] is None;break
            else:raise AssertionError('Reset did not clear completion report')
