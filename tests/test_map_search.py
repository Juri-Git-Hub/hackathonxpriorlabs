import json
import numpy as np
import pandas as pd
import pytest
from shared.schemas import BuildingBox,WorldInitPayload
from sim.world import World
from integration.map_features import MapPlanner3D,candidates,feature_table,VERSION,TARGET,CONTACT_PPM,HORIZON_SECONDS
from integration.acquisition_control import AcquisitionController
from integration.map_scoring import load_search_model
from tests.test_search_control import reading


def empty_world():
    return WorldInitPayload(bounds_min=(-20,-20,0),bounds_max=(20,20,20),buildings=[],assets=[],grid_coords=[])


def test_navigation_and_candidates_follow_shifted_map_bounds():
    world=WorldInitPayload(bounds_min=(100,200,0),bounds_max=(160,260,32),buildings=[],assets=[],grid_coords=[])
    planner=MapPlanner3D(world);q=candidates(world,planner)
    assert np.all(q[:,0]>=102) and np.all(q[:,1]>=202)
    assert np.all(q[:,0]<=158) and len(np.unique(q[:,2]))>=3
    assert planner.route((104,204,5),(155,255,25))
    assert not planner.route((0,0,5),(155,255,25))


def test_obstacle_changes_candidates_and_route_climbs_over_wall():
    world=empty_world();open_planner=MapPlanner3D(world);open_q=candidates(world,open_planner)
    blocked=world.model_copy(update={'buildings':[BuildingBox(id='wall',kind='building',min_xyz=(-8,-20,0),max_xyz=(8,20,10))]})
    planner=MapPlanner3D(blocked);q=candidates(blocked,planner)
    assert len(q)<len(open_q)
    route=planner.route((-12,0,3),(12,0,3))
    assert route and max(p[2] for p in route)>12
    assert all(planner.free(np.array(p)) for p in route)
    assert all(planner.visible(a,b) for a,b in zip(route,route[1:]))


def test_boundary_overshoot_has_a_real_recovery_segment_without_crossing_obstacles():
    world=empty_world();planner=MapPlanner3D(world)
    route=planner.route((-18.5,-18.3,5),(0,0,5))
    assert route[0]==(-18.5,-18.3,5) and route[1]==(-18.,-18.,5)
    assert all(planner.visible(a,b) for a,b in zip(route,route[1:]))
    assert not planner.route((-21,0,5),(0,0,5))
    blocked=world.model_copy(update={'buildings':[BuildingBox(id='wall',kind='building',min_xyz=(-20,-20,0),max_xyz=(-15,-15,10))]})
    assert not MapPlanner3D(blocked).route((-18.5,-18.3,5),(0,0,5))
    q=candidates(world,planner)
    assert np.all(q[:,:2]>planner.bounds_lo[:2]) and np.all(q[:,:2]<planner.bounds_hi[:2])


def test_map_table_masks_unobserved_gas_and_uses_only_past_measurements():
    w=empty_world();q=np.array([[0,0,3],[12,12,11]])
    initial=feature_table(w,q,(0,0,5),wind=(None,None))
    assert initial.sensor_missing.eq(1).all() and initial.wind_missing.eq(1).all()
    assert initial.measurement_count.eq(0).all()
    assert not any('target' in c or 'source' in c or 'leak' in c for c in initial)
    observed=feature_table(w,q,(0,0,5),[reading(1,pos=(0,0,5),ppm=9)],as_of=1)
    assert observed.sensor_missing.eq(0).all()
    assert np.isfinite(initial.to_numpy()).all() and np.isfinite(observed.to_numpy()).all()
    assert observed.sensor_now_log.eq(np.log1p(9)).all()
    with pytest.raises(ValueError,match='Future'):feature_table(w,q,(0,0,5),[reading(2)],as_of=1)


def test_map_features_distinguish_upwind_obstacle_from_clear_space():
    w=empty_world().model_copy(update={'buildings':[BuildingBox(id='wall',kind='building',min_xyz=(-2,-10,0),max_xyz=(2,10,10))]})
    f=feature_table(w,[[10,0,3],[-10,0,3]],(0,-15,5),wind=(2,0))
    assert f.upwind_obstacle_count.tolist()==[1,0]
    reverse=feature_table(w,[[10,0,3],[-10,0,3]],(0,-15,5),wind=(-2,0))
    assert reverse.upwind_obstacle_count.tolist()==[0,1]


def test_preflight_scores_change_path_on_same_map_and_no_sensor_data_is_fabricated():
    class Score:
        backend='fixture'
        def __init__(self,sign):self.sign=sign
        def predict(self,f):
            assert f.sensor_missing.eq(1).all()
            return np.where(self.sign*f.query_x>5,.99,.001)
    w=empty_world();obs=reading(.5,pos=(0,0,5),ppm=.5)
    a=AcquisitionController(w);b=AcquisitionController(w)
    a.ingest(obs);b.ingest(obs)
    assert a.goal is None and not a.preflight_complete
    a.plan(Score(-1),obs,preflight=True);b.plan(Score(1),obs,preflight=True)
    assert a.goal[0]<0 and b.goal[0]>0
    assert len(a.plan_targets)>1 and a.plan_state=='injected_test_model'
    preview=a.preview_route(obs.pos,a.route)
    assert all(a.planner.visible(x,y) for x,y in zip(preview,preview[1:]))


def test_fallback_has_candidates_behind_compressor_and_no_equipment_route_dependency():
    w=World().to_world_init();c=AcquisitionController(w)
    q=c.search_grid
    assert np.any((q[:,0]<-37)&(q[:,1]>16)&(q[:,1]<32))
    assert np.any((q[:,1]>34)&(q[:,0]>-35)&(q[:,0]<-18))
    # Removing equipment metadata cannot change a geometric fallback route.
    no_assets=w.model_copy(update={'assets':[]});d=AcquisitionController(no_assets)
    obs=reading(pos=(-48,-45,5),ppm=.5)
    c.ingest(obs);d.ingest(obs);c.plan(None,obs,preflight=True);d.plan(None,obs,preflight=True)
    assert c.plan_state=='coverage_fallback'
    assert c.plan_targets==d.plan_targets


def test_checkpoint_replanning_and_gas_switch_use_current_measurements():
    c=AcquisitionController(empty_world());o=reading(.5,pos=(0,0,5),ppm=.5)
    c.ingest(o);c.plan(None,o,preflight=True);goal=tuple(c.goal)
    c.ingest(reading(1,pos=goal,ppm=.5));assert c.needs_plan
    c.plan(None,c.history[-1]);assert c.plan_revision==2 and not c.needs_plan
    for t in [1.5,2,2.5]:c.ingest(reading(t,pos=goal,ppm=30))
    assert c.phase=='track' and not c.tail_route
    for t in [3,3.5,4,4.5,5]:c.ingest(reading(t,pos=goal,ppm=30))
    c.set_source_prediction([np.array(goal)+[0,0,6]],[.99],5)
    assert c.prediction_decisions==1 and c.goal[2]>goal[2]


def test_missing_or_incompatible_dataset_is_not_misrepresented_as_tabpfn_map(tmp_path,monkeypatch):
    monkeypatch.delenv('PLUMESENSE_SEARCH_DATASET',raising=False)
    with pytest.raises(ValueError,match='does not exist'):load_search_model(empty_world(),tmp_path/'missing')
    (tmp_path/'manifest.json').write_text(json.dumps({'feature_version':'old-hazard'}))
    with pytest.raises(ValueError,match='map-search-v1'):load_search_model(empty_world(),tmp_path)


def test_loader_rejects_held_out_rows_before_any_api_call(tmp_path):
    w=empty_world();frame=feature_table(w,[[0,0,3],[5,5,3]],(0,0,5))
    columns=list(frame);frame[TARGET]=[0,1];frame['split']=['train','test']
    frame.to_parquet(tmp_path/'train.parquet')
    (tmp_path/'manifest.json').write_text(json.dumps({'feature_version':VERSION,'feature_columns':columns,'target':TARGET,'contact_threshold_ppm':CONTACT_PPM,'horizon_seconds':HORIZON_SECONDS}))
    with pytest.raises(ValueError,match='non-training'):load_search_model(w,tmp_path)


def test_preflight_is_nonblocking_and_reset_discards_an_old_model_plan():
    import threading
    import time
    from agents.plumesense_agent import PlumeSenseAgent
    entered=threading.Event();release=threading.Event()
    class BlockedMap:
        def predict(self,f):
            entered.set();release.wait(5);return np.full(len(f),.1)
    class Legacy:
        def predict(self,f):return np.full(len(f),.1)
    agent=PlumeSenseAgent(models=[Legacy(),Legacy()],search_model=BlockedMap())
    world=empty_world();agent.reset(world);obs=reading(pos=(0,0,5),ppm=.5)
    try:
        t=time.perf_counter();target=agent.update(obs)
        assert time.perf_counter()-t<.1 and target==obs.pos
        assert entered.wait(2) and agent.planned_route==[]
        agent.reset(world);release.set();agent._executor.shutdown(wait=True)
        assert agent.plan_state=='preparing' and agent.planned_route==[]
        assert agent.pop_belief_update() is None and agent.history==[]
    finally:
        release.set();agent.close()
