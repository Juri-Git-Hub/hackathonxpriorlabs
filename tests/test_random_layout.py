"""Verify generated infrastructure and the map consumed by live navigation."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.mock_agent import MockTabPFNAgent
from integration.acquisition_control import AcquisitionController
from server.app import create_app
from shared.schemas import JudgeCommand
from sim.engine import SimEngine
from sim.world import World
from sim import layout
from sim.world import _MAPS
from tests.test_search_control import reading


@pytest.mark.parametrize('map_id', list(_MAPS))
def test_randomize_keeps_active_map_and_restores_its_layout(map_id):
    engine = SimEngine({'tabpfn': MockTabPFNAgent()}, layout_seed=42)
    engine.apply_command(JudgeCommand(action='select_map', map_id=map_id))
    original = engine.get_world_init()
    engine.apply_command(JudgeCommand(action='randomize_layout', seed=71))
    randomized = engine.get_world_init()
    assert randomized.map_id == map_id
    assert randomized.layout_seed == 71
    assert randomized.buildings != original.buildings
    other = 'pipe_loop' if map_id == 'refinery' else 'refinery'
    engine.apply_command(JudgeCommand(action='select_map', map_id=other))
    engine.apply_command(JudgeCommand(action='select_map', map_id=map_id))
    assert engine.get_world_init() == randomized


def test_layout_retries_a_failed_placement_deterministically(monkeypatch):
    original = layout._generate_layout_attempt
    calls = []

    def fail_first(*args):
        calls.append(args[2])
        if len(calls) % 2:
            raise layout._PlacementFailed()
        return original(*args)

    monkeypatch.setattr(layout, '_generate_layout_attempt', fail_first)
    first = World(layout_seed=23, map_id='generated').to_world_init()
    second = World(layout_seed=23, map_id='generated').to_world_init()
    assert first == second
    assert calls == [23, 23, 23, 23]


@pytest.mark.parametrize('seed', range(12))
def test_layout_has_clear_launch_pads_and_one_physically_connected_pipe_network(seed):
    world = World(layout_seed=seed)
    nodes = {a.id: a.pos for a in world.assets}
    for b in world.buildings:
        assert np.all(np.array(b.min_xyz) >= world.bounds_min)
        assert np.all(np.array(b.max_xyz) <= world.bounds_max)
        nodes[b.id] = None
    for i, a in enumerate(world.buildings):
        for b in world.buildings[i+1:]:
            assert np.any(np.array(a.max_xyz[:2])+7 < b.min_xyz[:2]) or np.any(np.array(a.min_xyz[:2])-7 > b.max_xyz[:2])
    assert all(not world.is_inside_building((-48., y, 5.), margin=5) for y in (-45., -40., -35.))
    assert all(not world.is_inside_building(a.pos, margin=2) for a in world.assets)

    graph = {key: set() for key in nodes}
    attachments = {}
    assert len(world.pipe_runs) == len(nodes)-1
    for pipe in world.pipe_runs:
        graph[pipe.from_id].add(pipe.to_id); graph[pipe.to_id].add(pipe.from_id)
        assert len(pipe.points) >= 2
        for node_id, point in ((pipe.from_id, pipe.points[0]), (pipe.to_id, pipe.points[-1])):
            if nodes[node_id] is not None:
                assert point == nodes[node_id]
            else:
                b = next(b for b in world.buildings if b.id == node_id)
                assert np.all(np.array(point) >= b.min_xyz) and np.all(np.array(point) <= b.max_xyz)
                assert any(point[d] in (b.min_xyz[d], b.max_xyz[d]) for d in (0, 1))
            assert node_id not in attachments or attachments[node_id] == point
            attachments[node_id] = point
        for a, b in zip(pipe.points, pipe.points[1:]):
            delta = np.subtract(a, b)
            assert np.count_nonzero(np.abs(delta) > 1e-8) == 1
            # Terminal building faces may touch; no segment may enter any solid.
            samples = np.linspace(a, b, max(3, int(np.linalg.norm(delta)*4)))
            assert all(not world.is_inside_building(tuple(p), margin=-1e-5) for p in samples)
    seen, stack = set(), [next(iter(graph))]
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node); stack.extend(graph[node]-seen)
    assert seen == set(nodes)


def test_layout_is_reproducible_and_changes_the_actual_preflight_route():
    a, repeat, b = [World(layout_seed=s).to_world_init() for s in (42, 42, 43)]
    assert a == repeat and a.buildings != b.buildings and a.pipe_runs != b.pipe_runs
    plans = []
    for world in (a, b):
        controller = AcquisitionController(world)
        obs = reading(.5, pos=(-48., -45., 5.), ppm=.5)
        controller.ingest(obs); controller.plan(None, obs, preflight=True)
        route = controller.preview_route(obs.pos, controller.route)
        assert all(controller.planner.visible(x, y) for x, y in zip(route, route[1:]))
        plans.append(route)
    assert plans[0] != plans[1]


def test_layout_reset_rebuilds_physics_and_keeps_wind_but_clears_the_mission():
    engine = SimEngine({'tabpfn': MockTabPFNAgent()}, layout_seed=42)
    engine.apply_command(JudgeCommand(action='set_wind', wind_angle_deg=90, wind_speed=4))
    engine.apply_command(JudgeCommand(action='randomize_layout', seed=71))
    assert engine.world.layout_seed == 71 and engine.sim_time == 0
    assert engine.airflow.world is engine.world and engine.plume.world is engine.world
    assert np.allclose(engine.airflow.w0, (0, 4))
    assert len(engine.leaks) == 1
    assert engine.leaks[0].pos == next(a.pos for a in engine.world.assets if a.id == 'flange_01')
    assert engine.get_world_init().pipe_runs == engine.world.pipe_runs
    old = engine.get_world_init()
    engine.apply_command(JudgeCommand(action='reset_race', keep_leaks=True))
    assert engine.get_world_init() == old
    engine.apply_command(JudgeCommand(action='randomize_layout'))
    assert engine.world.layout_seed == 72


def test_layout_change_broadcasts_the_same_new_geometry_to_browser_and_agent():
    app = create_app(mock_agents=True, layout_seed=42)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        first = json.loads(ws.receive_text()); ws.receive_text()
        assert first['layout_seed'] == 42 and len(first['pipe_runs']) == 19
        ws.send_json({'type': 'judge_cmd', 'action': 'randomize_layout', 'seed': 123})
        for _ in range(40):
            message = json.loads(ws.receive_text())
            if message['type'] == 'world_init':
                assert message['layout_seed'] == 123
                assert message['buildings'] != first['buildings']
                assert message['pipe_runs'] == app.state.engine.get_world_init().model_dump(mode='json')['pipe_runs']
                break
        else:
            pytest.fail('Layout replacement was not broadcast')


def test_generated_world_uses_a_new_seed_without_replacing_refinery():
    engine = SimEngine({"tabpfn": MockTabPFNAgent()}, layout_seed=42)
    refinery = engine.get_world_init()
    command = JudgeCommand(action="randomize_layout", map_id="generated")
    engine.apply_command(command)
    first = engine.get_world_init()
    assert first.map_id == "generated"
    assert first.layout_seed != refinery.layout_seed
    assert first.buildings != refinery.buildings
    assert first.pipe_runs != refinery.pipe_runs

    engine.apply_command(command)
    second = engine.get_world_init()
    assert second.layout_seed != first.layout_seed
    assert second.buildings != first.buildings
    assert engine.airflow.world is engine.plume.world is engine.world

    engine.apply_command(JudgeCommand(action="select_map", map_id="refinery"))
    assert engine.get_world_init() == refinery
