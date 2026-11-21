"""Regressions for measured features, source resolution and local 3D sensing."""
import numpy as np
import pandas as pd
import pytest

from integration.features import rich_features
from integration.reduced_features import COMPACT, RELATIVE_COLUMNS, lean_features, relative_features
from integration.search_control import SearchController, safe_goal, source_queries
from shared.schemas import DroneObservation, WorldInitPayload
from sim.world import World


def reading(t=0.5, pos=(-40, -40, 5), ppm=20, wind=(2, 1), vel=(0, 0, 0)):
    return DroneObservation(
        sim_time=t, drone_id='tabpfn', pos=pos, vel=vel,
        wind_measured=wind, ppm_measured=ppm, temp_delta=None,
        nearest_asset_type='open_air', asset_age_years=0,
        in_wind_shadow=False, upwind_pipe_score=0, battery_pct=100,
    )


@pytest.mark.parametrize('wind', [(2, 1), (None, None), (0, None)])
def test_fast_feature_builder_matches_verified_training_features(wind):
    history=[reading(0.5, ppm=20, wind=wind), reading(4, (-36, -39, 8), 20),
             reading(7, (-32, -35, 5), 4)]
    grid=[(-40, -40, 5), (-32, -35, 8), (15, 0, 23)]
    compact=lean_features(history, grid)
    pd.testing.assert_frame_equal(compact, rich_features(history, grid)[COMPACT])
    assert np.isfinite(compact.to_numpy()).all()


def test_source_features_are_translation_and_rotation_invariant():
    history=[reading(), reading(2, (-35, -35, 7), 8)]
    grid=np.array([[-34, -33, 4], [-30, -36, 10]])
    initial=relative_features(lean_features(history, grid))
    shift=np.array([10, -5, 3])
    moved=[o.model_copy(update={'pos': tuple(np.array(o.pos)+shift)}) for o in history]
    translated=relative_features(lean_features(moved, grid+shift))
    pd.testing.assert_frame_equal(initial, translated)

    def rotate(p):
        return (-p[1], p[0], p[2])
    turned=[o.model_copy(update={'pos': rotate(o.pos),
        'wind_measured': (-o.wind_measured[1], o.wind_measured[0])}) for o in history]
    rotated=relative_features(lean_features(turned, [rotate(p) for p in grid]))
    np.testing.assert_allclose(initial, rotated, atol=1e-12)
    assert list(initial)==RELATIVE_COLUMNS and len(initial.columns)==12


def test_missing_peak_wind_uses_current_observed_wind():
    history=[reading(ppm=100, wind=(None, None)), reading(2, ppm=5, wind=(0, 2))]
    f=relative_features(lean_features(history, [(-40, -36, 6)]))
    assert f.along_wind.iloc[0]==4 and f.wind_speed.iloc[0]==2
    assert np.isfinite(f.to_numpy()).all()


def test_source_hypotheses_are_not_restricted_to_navigable_grid():
    world=World().to_world_init()
    # A known piece of equipment near the default pipe rack: the source may
    # be there even though navigation must keep drone clearance around it.
    equipment=np.array([-6., -7.2, 4.])
    coarse=np.array(world.grid_coords)
    assert np.linalg.norm(coarse-equipment, axis=1).min()>5
    fine=source_queries(world, [reading()], coarse)
    np.testing.assert_array_equal(fine[:len(coarse)], coarse)
    assert np.linalg.norm(fine-equipment, axis=1).min()<0.01
    planner=SearchController(world).planner
    assert not planner.free(equipment)
    assert planner.free(np.array(safe_goal(planner, equipment)))


def test_fine_source_queries_cover_non_asset_xyz_neighborhood():
    world=World().to_world_init();coarse=np.array(world.grid_coords)
    contact=(35, 35, 19)
    fine=source_queries(world, [reading(pos=contact, ppm=100)], coarse)
    assert np.linalg.norm(fine-np.array(contact), axis=1).min()<1e-6
    assert len(np.unique(fine[:,2]))>len(np.unique(coarse[:,2]))
    assert len(np.unique(fine, axis=0))==len(fine)


def test_contact_waits_for_sensor_then_tests_model_hint_in_three_dimensions():
    c=SearchController(World().to_world_init());first=reading()
    c.ingest(first);c.set_source_hint((-36, -40, 9), first.sim_time)
    np.testing.assert_allclose(c.goal, first.pos)
    for t in [1, 1.5, 2, 2.5]:c.ingest(reading(t))
    assert c.completed_probes==0
    c.ingest(reading(3))
    assert c.completed_probes==1 and c.phase=='track'
    assert c.goal[0]>first.pos[0] and c.goal[2]>first.pos[2]
    assert c.probe_kind=='probe'
    assert all(c.planner.visible(a, b) for a,b in zip([first.pos,*c.route],c.route))


def test_sensor_settling_survives_small_gusts_and_uses_measured_position():
    c=SearchController(World().to_world_init());c.ingest(reading())
    for i in range(1,7):
        c.ingest(reading(.5+i*.5, (-39, -40, 5), vel=(1.5, 0, 0)))
    assert c.completed_probes>=1
    np.testing.assert_allclose(c.anchor, (-39, -40, 5))


def test_empty_probe_cycle_shrinks_and_allows_return_to_contact():
    c=SearchController(World().to_world_init());o=reading();c.ingest(o)
    c.probes=[];c.cycle_active=True;c._next_probe(o)
    assert c.step==pytest.approx(3.6) and c.probe_kind=='center'
    np.testing.assert_allclose(c.goal, o.pos)


def test_source_queries_and_controller_need_no_simulator_or_hidden_leak():
    # This payload deliberately contains only public map geometry and assets.
    world=WorldInitPayload(buildings=[],assets=[],grid_coords=[(-40,-40,5),(0,0,15)])
    c=SearchController(world);c.ingest(reading(ppm=0))
    q=source_queries(world, c.history, world.grid_coords)
    assert q.shape==(2,3) and c.goal is not None
    assert 'leaks' not in world.model_dump()


def test_cached_hazard_keeps_its_original_forecast_timestamp():
    from agents.plumesense_agent import PlumeSenseAgent
    class Model:
        def __init__(self):self.calls=0
        def predict(self, frame):
            self.calls+=1
            return np.full(len(frame),.1)
    source=Model();hazard=Model()
    agent=PlumeSenseAgent(models=[source,hazard]);agent.reset(World().to_world_init())
    try:
        for t,expected in [(.5,4.5),(3.5,4.5),(6.5,4.5),(12.5,16.5)]:
            obs=reading(t);agent._record_observation(obs)
            _,belief=agent._run_inference(obs)
            assert belief.hazard_forecast_time==expected
            assert belief.hazard_observed_at==expected-4
        assert source.calls==4 and hazard.calls==2
        assert agent.source_prediction_calls==4 and agent.hazard_prediction_calls==2
        assert belief.source_prediction_backend=='injected_test_model'
        agent.reset(World().to_world_init())
        assert agent.source_prediction_calls==agent.hazard_prediction_calls==0
    finally:agent.close()
