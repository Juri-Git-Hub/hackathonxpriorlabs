import numpy as np
from agents.navigation import Planner3D, RoutedAgent
from shared.schemas import BuildingBox, WorldInitPayload
from sim.drone import Drone
from sim.world import World
from agents.lawnmower import LawnmowerAgent

def test_route_climbs_over_wall_and_descends():
    w=WorldInitPayload(buildings=[BuildingBox(id='wall',kind='building',min_xyz=(-3,-60,0),max_xyz=(3,60,12))],assets=[],grid_coords=[])
    p=Planner3D(w);route=p.route((-20,0,3),(20,0,3))
    assert route and route[0]==(-20,0,3) and route[-1]==(20,0,3)
    assert max(q[2] for q in route)>14.5
    assert all(p.visible(a,b) for a,b in zip(route,route[1:]))
    assert not p.visible((-20,0,3),(20,0,3))

def test_vertical_drone_motion_up_and_down():
    world=World();d=Drone('test',(-48,-45,4));d.set_target((-48,-45,23))
    for _ in range(200):d.step(.05,np.zeros(3),world)
    assert d.pos[2]>20
    d.set_target((-48,-45,3))
    for _ in range(220):d.step(.05,np.zeros(3),world)
    assert d.pos[2]<5

def test_volumetric_baseline_changes_height_before_crossing_facility():
    a=LawnmowerAgent();a.reset(World().to_world_init())
    first=a._waypoints[:6]
    assert len({p[2] for p in first})>=4
    assert len({p[:2] for p in first})==1

def test_planner_rejects_targets_outside_flight_volume():
    p=Planner3D(World().to_world_init())
    assert not p.route((-48,-45,5),(-48,-45,40))
    assert not p.route((-48,-45,5),(-48,-45,0))

def test_unobstructed_route_moves_in_all_three_axes():
    p=Planner3D(WorldInitPayload(buildings=[],assets=[],grid_coords=[]))
    assert p.route((-20,-20,3),(20,20,23))==[(-20,-20,3),(20,20,23)]


def test_planners_route_around_pipeline_without_proxy_buildings():
    from agents.tabpfn_nav import FastPlanner3D
    from shared.schemas import PipeRun

    pipe = PipeRun(id="elevated", points=[(-12, 0, 12), (12, 0, 12)], radius=1)
    start, goal = (0, -10, 12), (0, 10, 12)
    for field in ("pipe_runs", "pipes"):
        world = WorldInitPayload(buildings=[], assets=[], grid_coords=[], **{field: [pipe]})
        for planner_type in (Planner3D, FastPlanner3D):
            planner = planner_type(world)
            assert not planner.free((0, 0, 12))
            assert not planner.visible(start, goal)
            assert not planner.visible((0, -10, 5), (0, 10, 5))  # Midpoint support
            assert planner.visible((6, -10, 5), (6, 10, 5))
            route = planner.route(start, goal)
            assert len(route) >= 3
            assert all(planner.visible(a, b) for a, b in zip(route, route[1:]))


def test_drone_detours_around_pipe_without_intersecting_it():
    from agents.mock_agent import MockTabPFNAgent
    from shared.flight_geometry import sweep_motion
    from shared.schemas import JudgeCommand
    from sim.engine import SimEngine

    engine = SimEngine({"tabpfn": MockTabPFNAgent()})
    engine.apply_command(JudgeCommand(action="select_map", map_id="pipe_loop"))
    world = engine.world
    start, goal = (0, -17, 21), (0, 4, 21)
    assert sweep_motion(world.flight_bounds, start, goal)[1] == 1
    drone = Drone("test", start_pos=start)
    drone.set_target(goal)
    waypoint = drone._navigation_target(world, start)
    assert waypoint not in (start, goal)
    assert sweep_motion(world.flight_bounds, start, waypoint)[1] == -1

    for _ in range(360):
        previous = drone.pos_tuple
        drone.step(.05, np.zeros(3), world)
        assert sweep_motion(world.flight_bounds, previous, drone.pos_tuple)[1] == -1
        assert not any(
            all(a <= x <= b for a, x, b in zip(lo, drone.pos, hi))
            for lo, hi in world.flight_bounds
        )
    assert np.linalg.norm(drone.pos - goal) < 2.5


def test_tabpfn_keeps_a_corner_until_the_next_segment_is_visible():
    from agents.reliable_tabpfn import ReliableTabPFNAgent
    from shared.schemas import BuildingBox, DroneObservation

    wall = BuildingBox(id="wall", kind="building", min_xyz=(-2, -2, 0), max_xyz=(2, 2, 8))
    world = WorldInitPayload(buildings=[wall], assets=[], grid_coords=[])
    agent = ReliableTabPFNAgent()
    try:
        agent.reset(world)
        corner, goal = (4.8, 4.8, 5.0), (-10.0, 4.8, 5.0)
        obs = DroneObservation(sim_time=1, drone_id="tabpfn", pos=(4.8, 3.0, 5.0), vel=(0, 0, 0),
                               wind_measured=(3, 0), ppm_measured=0, temp_delta=0,
                               nearest_asset_type="wall", asset_age_years=0, in_wind_shadow=False,
                               upwind_pipe_score=0, battery_pct=100)
        agent.route = [corner, goal]
        assert agent._follow_route(obs) == corner
        assert agent._follow_route(obs.model_copy(update={"pos": corner})) == goal
    finally:
        agent.close()


def test_collision_backstop_stops_momentum_at_building_and_pipeline():
    world = World(map_id="pipe_loop")
    pipe_face = -6 - 1.05 * 1.42 - Drone.BUILDING_CLEARANCE
    building_face = -13 - Drone.BUILDING_CLEARANCE
    for x, z, face in ((5, 21, pipe_face), (-32, 4, building_face)):
        drone = Drone("test", (x, face - .03, z))
        drone.vel[1] = 8
        drone.step(.05, np.zeros(3), world)
        assert face - .01 < drone.pos[1] < face
        assert drone.vel[1] < 1


def test_drone_brakes_before_pipe_clearance():
    world = World(map_id="pipe_loop")
    face = -6 - 1.05 * 1.42 - Drone.BUILDING_CLEARANCE
    drone = Drone("test", (5, face - 9, 21))
    drone.vel[1] = 8
    drone.set_target((5, 4, 21))
    for _ in range(100):
        drone.step(.05, np.zeros(3), world)

    assert drone.clearance_contacts == 0
    assert np.linalg.norm(drone.pos - (5, 4, 21)) < 1.0
