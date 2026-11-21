import numpy as np
import pytest

from integration.navigation import Planner3D
from integration.prediction_control import PredictionController
from integration.wind_planning import WindAwarePlanner
from shared.schemas import BuildingBox,WorldInitPayload
from sim.world import World
from tests.test_search_control import reading


def test_headwind_and_crosswind_cost_more_without_tailwind_speed_bonus():
    p=WindAwarePlanner(World().to_world_init())
    p.observe_wind([],reading(wind=(4,0)))
    a=np.array([-40.,-40.,5.])
    tail=p.cost(a,a+[6,0,0]);head=p.cost(a,a+[-6,0,0]);cross=p.cost(a,a+[0,6,0])
    assert head>cross>tail==pytest.approx(.75)


def test_no_wind_matches_original_cost_and_invalid_or_stale_wind_falls_back():
    p=WindAwarePlanner(World().to_world_init());a=np.array([-40.,-40.,5.]);b=np.array([-35.,-34.,8.])
    for wind in [(0,0),(None,None),(np.nan,0)]:
        p.observe_wind([],reading(wind=wind))
        assert p.cost(a,b)==Planner3D.cost(a,b)
        assert p.measurement_cost(a,b)==np.linalg.norm(b-a)
    p.observe_wind([reading(.5,wind=(4,0))],reading(6,wind=(None,None)))
    assert not p.wind_available and p.cost(a,b)==Planner3D.cost(a,b)


def test_wind_estimate_uses_recent_measurements_and_bounds_outliers():
    p=WindAwarePlanner(World().to_world_init())
    p.observe_wind([reading(.5,wind=(3,1)),reading(1,wind=(3,1))],reading(1.5,wind=(1000,1000)))
    np.testing.assert_allclose(p.measured_wind,(3,1))
    p.observe_wind([],reading(2,wind=(1000,1000)))
    assert np.linalg.norm(p.measured_wind)<=8.000001
    a=np.array([-40.,-40.,5.]);b=a+[-6,0,0]
    assert Planner3D.cost(a,b)<=p.cost(a,b)<=1.35*Planner3D.cost(a,b)


def prepared(wind):
    c=PredictionController(World().to_world_init(),wind_aware=True)
    for t in [.5,1,1.5,2,2.5,3]:c.ingest(reading(t,wind=wind))
    return c


def test_equal_source_scores_choose_different_targets_when_wind_reverses():
    points=[(-46,-40,5),(-34,-40,5)]
    east=prepared((4,0));west=prepared((-4,0))
    east.set_source_prediction(points,[.9,.9],3)
    west.set_source_prediction(points,[.9,.9],3)
    assert east.goal[0]>-40 and west.goal[0]<-40


def test_strong_source_prediction_overrides_headwind_preference_and_goal_stays_committed():
    c=prepared((4,0));points=[(-46,-40,5),(-34,-40,5)]
    c.set_source_prediction(points,[.99,.01],3)
    assert c.goal[0]<-40
    goal=c.goal.copy()
    for t in [3.5,4,4.5,5]:c.ingest(reading(t,wind=(-4,0)))
    c.set_source_prediction(points,[.01,.99],5)
    np.testing.assert_array_equal(c.goal,goal)


def test_wind_aware_route_still_climbs_and_checks_obstacles():
    w=WorldInitPayload(buildings=[BuildingBox(id='wall',kind='building',min_xyz=(-3,-60,0),max_xyz=(3,60,12))],assets=[],grid_coords=[])
    p=WindAwarePlanner(w);p.observe_wind([],reading(wind=(3,1)))
    route=p.route((-20,0,3),(20,0,3))
    assert route and max(q[2] for q in route)>14.5
    assert all(p.visible(a,b) for a,b in zip(route,route[1:]))
