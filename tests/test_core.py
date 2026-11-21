import numpy as np
from plumesense import *

def test_wall_flux():
    s=Simulation(1);s.walls[:]=False;s.walls[:,12]=True;s.source=np.array([3,3]);s.wind=np.array([.7,0])
    for _ in range(60):s.step()
    assert np.all(s.gas[:,12:]==0)
    assert np.all(s.gas>=0)

def test_features_do_not_read_ground_truth():
    s=Simulation(0);s.step();d=np.array([0,0]);obs=s.observe(d)
    a=features(s.walls,s.free,d,s.wind,obs,obs);s.gas[:]=999
    assert a.equals(features(s.walls,s.free,d,s.wind,obs,obs))

def test_astar_avoids_risk_and_walls():
    w=np.zeros((5,5),bool);r=np.zeros((5,5));r[2,1:4]=1
    route=astar(w,r,(0,2),(4,2));assert route[0]==(0,2) and route[-1]==(4,2)
    assert all(r[y,x]==0 for x,y in route)
    w[:,2]=True;assert astar(w,r,(0,2),(4,2))==[]

def test_generation_episode_isolation():
    d=generate(4,steps=6,queries=8)
    assert not set(d[d.episode_id<3].episode_id)&set(d[d.episode_id>=3].episode_id)
    assert set(d[TARGET])<={0,1}
    assert not any(c.startswith('target') for c in FEATURES)

def test_api_probabilities():
    import pytest
    assert np.allclose(api_probabilities({'probabilities':[[.2,.8]],'classes':[0,1]},1),[.8])
    assert np.allclose(api_probabilities({'predict_proba':[[.2,.8]],'classes':[1,0]},1),[.2])
    with pytest.raises(ValueError):api_probabilities({'predictions':[1]},1)
    with pytest.raises(ValueError):api_probabilities({'probabilities':[float('nan')]},1)


def test_api_request(monkeypatch):
    import json
    import plumesense
    from contextlib import contextmanager
    s=Simulation(0);obs=s.observe(np.array([0,0]));f=features(s.walls,s.free[:2],np.array([0,0]),s.wind,obs,obs)
    train=f.copy();train[TARGET]=[0,1]
    @contextmanager
    def fake_urlopen(request,timeout):
        from io import BytesIO
        body=json.loads(request.data)
        assert set(body)=={'X_train','y_train','X_test'}
        assert len(body['X_train'][0])==len(FEATURES)
        assert timeout==300
        yield BytesIO(json.dumps({'probabilities':[[.8,.2],[.3,.7]]}).encode())
    monkeypatch.setenv('PLUMESENSE_AUTO_TUNNEL','0')
    monkeypatch.setattr(plumesense,'urlopen',fake_urlopen)
    assert np.allclose(Predictor('tabpfn-api').fit(train).predict(f),[.2,.7])

def test_tunnel_reuses_existing_listener(monkeypatch):
    import tunnel
    monkeypatch.setattr(tunnel,'_listening',lambda host,port:True)
    monkeypatch.setattr(tunnel.subprocess,'Popen',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must reuse listener')))
    tunnel.ensure_tunnel()

def test_tunnel_skips_external_endpoint(monkeypatch):
    import tunnel
    monkeypatch.setattr(tunnel.subprocess,'Popen',lambda *a,**k: (_ for _ in ()).throw(AssertionError('no SSH for external URL')))
    tunnel.ensure_tunnel('http://example.org:8000/predict')

def test_predictor_requires_fit_and_valid_backend():
    import pytest
    with pytest.raises(ValueError,match='Unknown predictor backend'):Predictor('invalid')
    with pytest.raises(RuntimeError,match='not fitted'):Predictor('tabpfn').predict(pd.DataFrame())

def test_controller_commits_despite_changing_heatmap():
    s=Simulation(100);controller=NavigationController();visits=np.zeros(s.walls.shape);risk=np.zeros(s.walls.shape)
    drone=np.array([0,0]);goal=np.array([23,23])
    path=controller.plan(s,drone,goal,risk,visits)
    expected=path.copy()
    rng=np.random.default_rng(2)
    for i in range(1,10):
        drone=np.array(path[1])
        risk=rng.uniform(0,.7,s.walls.shape)
        path=controller.plan(s,drone,goal,risk,visits)
        assert path==expected[i:]

def test_active_target_persists_when_scores_change():
    s=Simulation(100);controller=NavigationController();v=np.zeros(s.walls.shape)
    risk=np.zeros(s.walls.shape);risk[10,10]=.5
    p=controller.plan(s,np.array([0,0]),[23,23],risk,v,True)
    target=controller.target
    assert len(p)>2
    risk[:]=.4
    q=controller.plan(s,np.array(p[1]),[23,23],risk,v,True)
    assert controller.target==target and q==p[1:]

def test_source_search_uses_observations_only():
    from source_search_physics import SourceSearch
    sim=Simulation(100);sim.step();drone=np.array([0,0]);obs=sim.observe(drone)
    a=SourceSearch(sim.walls,sim.wind);b=SourceSearch(sim.walls,sim.wind)
    a.update(1,obs,drone)
    sim.source[:]=[0,0];sim.gas[:]=999;sim.emission=999
    b.update(1,obs,drone)
    assert np.allclose(a.posterior,b.posterior)
    assert np.isclose(a.posterior.sum(),1)
    before=a.log_weights.copy();a.update(1,obs,drone)
    assert np.array_equal(before,a.log_weights)

def test_search_localizes_held_out_episode():
    from source_search_physics import SourceSearch
    sim=Simulation(100);drone=np.array([0,0]);search=SourceSearch(sim.walls,sim.wind)
    visits=np.zeros(sim.walls.shape);risk=np.zeros(sim.walls.shape)
    for _ in range(100):
        sim.step();search.update(sim.t,sim.observe(drone),drone)
        if search.found:break
        path=search.plan(drone,risk,visits)
        if len(path)>1:drone=np.array(path[1])
        visits[drone[1],drone[0]]+=1
    assert search.found
    assert np.linalg.norm(search.estimate-sim.source)<=2

def test_sensor_is_only_on_drone():
    s=Simulation(100);drone=np.array([7,4]);s.gas[:]=0;s.gas[4,7]=1
    positions,values=s.observe(drone)
    assert positions.shape==(1,2) and values.shape==(1,)
    assert np.array_equal(positions[0],drone)
    assert abs(values[0]-1)<.1

def test_source_search_rejects_remote_sensors():
    import pytest
    from source_search_physics import SourceSearch
    s=Simulation(0);search=SourceSearch(s.walls,s.wind)
    with pytest.raises(ValueError,match='only the sensor'):
        search.update(1,(np.array([[0,0],[2,2]]),np.array([0.,0.])),[0,0])

def test_source_emits_continuously():
    s=Simulation(100)
    decay=.997
    added=s.emission*.25*sum(decay**i for i in range(4))
    for _ in range(100):
        mass=s.gas.sum()
        s.step()
        assert np.isclose(s.gas.sum(),mass*decay**4+added)
