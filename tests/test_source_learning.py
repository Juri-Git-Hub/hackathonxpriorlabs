import numpy as np
from source_search import SourceSearch
from source_learning import History
from plumesense import load_map

class StubModel:
    def __init__(self):self.inputs=[]
    def predict(self,frame):
        self.inputs.append(frame.copy())
        return np.exp(-((frame.query_x-5)**2+(frame.query_y-3)**2))

def test_source_prediction_is_supplied_by_model():
    walls,_=load_map();model=StubModel();search=SourceSearch(walls,[.2,.1],model)
    obs=(np.array([[0,0]]),np.array([.02]))
    search.update(1,obs,[0,0])
    assert len(model.inputs)==1
    assert tuple(search.estimate)==(5,3)
    assert np.isclose(search.posterior.sum(),1)
    search.update(1,obs,[0,0]);assert len(model.inputs)==1
    assert not hasattr(search,'fields')

def test_full_history_contributes_after_recent_window():
    walls,_=load_map();a=History();b=History()
    for t in range(1,12):
        a.add(t,(np.array([[0,0]]),np.array([.01])),[0,0])
        b.add(t,(np.array([[0,0]]),np.array([.02 if t==1 else .01])),[0,0])
    fa=a.features([[0,0]],[0,0],walls);fb=b.features([[0,0]],[0,0],walls)
    assert fa['bin_0_0_mean'][0]!=fb['bin_0_0_mean'][0]
    assert not any(c.startswith('target') for c in fa)
    assert fa['count'][0]==11

def test_uninformative_model_explores_interior():
    walls,_=load_map()
    class Uniform:
        def predict(self,frame):return np.full(len(frame),.1)
    search=SourceSearch(walls,[.2,0],Uniform())
    search.update(1,(np.array([[0,0]]),np.array([0.])),[0,0])
    visits=np.zeros(walls.shape);risk=np.zeros(walls.shape)
    path=search.plan([0,0],risk,visits)
    assert len(path)>2
    assert search.target[0]>=2 and search.target[1]>=2
    target=search.target
    search.update(2,(np.array([path[1]]),np.array([0.])),path[1])
    assert search.plan(path[1],risk,visits)==path[1:]
    assert search.target==target

def test_query_features_use_actual_mobile_samples():
    walls,_=load_map();h=History()
    h.add(1,(np.array([[3,4]]),np.array([.7])),[3,4])
    f=h.features([[3,4],[20,20]],[.2,.1],walls)
    assert f.nearest_0_distance.iloc[0]==0
    assert f.local_count.tolist()==[1,0]
    assert f.local_peak.tolist()==[.7,0]
    assert f.nearest_0_value.tolist()==[.7,.7]
