import numpy as np

from integration.wind_search import WindReactiveController
from sim.world import World
from tests.test_search_control import reading


def cold_reading(t,wind=(3,0)):
    return reading(t,pos=(-48,-45,5),ppm=0,wind=wind)


def test_sustained_wind_reversal_refreshes_cold_search_target():
    c=WindReactiveController(World().to_world_init())
    for t in np.arange(.5,4.1,.5):c.ingest(cold_reading(float(t)))
    # Force an in-flight target selected using established eastward wind.
    c._survey(c.history[-1]);old_goal=c.goal.copy()
    for t in np.arange(4.5,12.6,.5):c.ingest(cold_reading(float(t),(-3,0)))
    assert c.wind_replans>=1
    assert np.linalg.norm(c.goal-old_goal)>1
    assert c.goal_at>=12


def test_calm_missing_and_brief_gusts_do_not_repeatedly_cancel_search():
    c=WindReactiveController(World().to_world_init())
    for t in np.arange(.5,4.1,.5):c.ingest(cold_reading(float(t)))
    c._survey(c.history[-1]);goal=c.goal.copy()
    for i,t in enumerate(np.arange(4.5,12.6,.5)):
        wind=(-3,0) if i==6 else (None,None) if i%3==0 else (3,0)
        c.ingest(cold_reading(float(t),wind))
    assert c.wind_replans==0
    np.testing.assert_array_equal(c.goal,goal)


def test_more_samples_of_unchanged_wind_do_not_cancel_the_initial_target():
    c=WindReactiveController(World().to_world_init())
    c.ingest(cold_reading(.5));goal=c.goal.copy()
    for t in np.arange(1,12.6,.5):c.ingest(cold_reading(float(t)))
    assert c.wind_replans==0
    np.testing.assert_array_equal(c.goal,goal)


def test_wind_reversal_preserves_committed_tabpfn_goal():
    c=WindReactiveController(World().to_world_init())
    for t in [.5,1,1.5,2,2.5,3]:c.ingest(reading(t,wind=(3,0)))
    c.set_source_prediction([(-34,-40,5)],[.99],3);goal=c.goal.copy()
    for t in np.arange(3.5,12.6,.5):c.ingest(reading(float(t),wind=(-3,0)))
    assert c.wind_replans==0 and c.phase=='track'
    np.testing.assert_array_equal(c.goal,goal)
