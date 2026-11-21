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
            assert 400 <= len(world_init.grid_coords) <= 650

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
                    assert len(tick.drones) == 3
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
