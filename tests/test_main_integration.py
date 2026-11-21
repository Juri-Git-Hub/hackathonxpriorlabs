import threading
import time
import numpy as np
from integration.features import features
from agents.plumesense_agent import PlumeSenseAgent
from sim.world import World
from shared.schemas import DroneObservation

def obs(t=.5):
    return DroneObservation(sim_time=t,drone_id='tabpfn',pos=(-40,-40,5),vel=(0,0,0),wind_measured=(None,None),ppm_measured=2,temp_delta=None,nearest_asset_type='valve',asset_age_years=10,in_wind_shadow=False,upwind_pipe_score=0,battery_pct=100)

def test_features_use_xyz_and_mobile_observations_only():
    f=features([obs()],[(0,0,3),(0,0,18)])
    assert f.query_z.tolist()==[3,18]
    assert f.wind_x_missing.eq(1).all()
    assert np.isfinite(f.to_numpy()).all()
    assert f.recent_0_log_ppm.eq(np.log1p(2)).all()
    assert not any(c.startswith('target') or 'source' in c or 'leak' in c for c in f)

def test_remote_agent_is_async_and_reset_discards_old_results():
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    class Model:
        def predict(self,f):
            entered.set();release.wait(timeout=5);finished.set();return np.full(len(f),.1)
    agent=PlumeSenseAgent(models=[Model(),Model()]);world=World().to_world_init();agent.reset(world)
    start=time.perf_counter();agent.update(obs());assert time.perf_counter()-start<.1
    assert entered.wait(timeout=2)
    agent.reset(world);release.set();assert finished.wait(timeout=2)
    agent._executor.shutdown(wait=True)
    assert agent.pop_belief_update() is None
    assert agent.hazard is None and agent.source_scores is None
    assert agent.history==[]
