import numpy as np
import pytest
from fastapi.testclient import TestClient
from agents.mock_agent import MockTabPFNAgent
from sim.engine import SimEngine
from shared.schemas import JudgeCommand
from server.app import create_app


def test_replacing_leak_starts_clean_single_source_and_restart_keeps_selection():
    e=SimEngine({'tabpfn':MockTabPFNAgent()});p=(-26.,38.,3.5)
    for _ in range(50):e.step()
    e.drones['tabpfn'].battery_pct=5
    e.apply_command(JudgeCommand(action='place_leak',leak_pos=p,replace_existing=True))
    assert len(e.leaks)==1 and e.leaks[0].pos==p
    assert e.sim_time==0 and e.drones['tabpfn'].battery_pct==100
    assert e.drones['tabpfn'].samples_taken==0 and not e._latest_agent_beliefs
    assert e.sensors['tabpfn'].filtered_ppm==0
    e.apply_command(JudgeCommand(action='reset_race',keep_leaks=True,seed=99))
    assert e.leaks[0].pos==p and e._seed==99
    e.apply_command(JudgeCommand(action='reset_race'))
    assert e.leaks[0].pos!=p


@pytest.mark.parametrize('position', [(0,0,3.5),(1000,0,4),(np.nan,0,4)])
def test_invalid_replacement_preserves_current_mission(position):
    e=SimEngine({'tabpfn':MockTabPFNAgent()});previous=e.leaks[0].pos
    with pytest.raises(ValueError):e.apply_command(JudgeCommand(action='place_leak',leak_pos=position,replace_existing=True))
    assert e.leaks[0].pos==previous


def test_websocket_replacement_refreshes_scene_and_errors_are_visible():
    app=create_app(mock_agents=True)
    world=app.state.engine.world
    valid_pos=(-48.0,-45.0,3.5)
    bldg=world.buildings[0]
    inside_pos=[
        0.5*(bldg.min_xyz[0]+bldg.max_xyz[0]),
        0.5*(bldg.min_xyz[1]+bldg.max_xyz[1]),
        0.5*(bldg.min_xyz[2]+bldg.max_xyz[2]),
    ]
    with TestClient(app) as client,client.websocket_connect('/ws') as ws:
        ws.receive_text();ws.receive_text()
        ws.send_json({'type':'judge_cmd','action':'place_leak','leak_pos':list(valid_pos),'replace_existing':True})
        import json
        for _ in range(40):
            message=json.loads(ws.receive_text())
            if message['type']=='world_init':break
        else:raise AssertionError('No scene reset after relocation')
        assert app.state.engine.leaks[0].pos==valid_pos
        assert len(app.state.engine.leaks)==1
        ws.send_json({'type':'judge_cmd','action':'place_leak','leak_pos':inside_pos,'replace_existing':True})
        for _ in range(40):
            message=json.loads(ws.receive_text())
            if message['type']=='error':
                assert 'inside equipment' in message['detail'];break
        else:raise AssertionError('No placement error returned')
