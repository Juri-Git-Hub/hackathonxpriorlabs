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
