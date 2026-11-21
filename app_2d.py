import time
import hashlib
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plumesense as core
# Streamlit reruns this file, but imported modules can survive those reruns.
core_version=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()
if getattr(core,'_loaded_source_version',None)!=core_version:
    importlib.reload(core)
    core._loaded_source_version=core_version
from plumesense import *
import source_learning
importlib.reload(source_learning)
import source_search
importlib.reload(source_search)
from source_search import SourceSearch,fit_source_model

st.set_page_config(page_title='PlumeSense',layout='wide')
st.title('PlumeSense · Autonomous Gas Leak Detection')
st.caption('Ein Gassensor auf der Drohne · Simulation · Prognosehorizont 4 Schritte · Ground Truth ausschließlich zur Visualisierung und Auswertung')
backend=st.sidebar.selectbox('Prognosemodell',['tabpfn-api','tabpfn'])
seed=st.sidebar.number_input('Episode / Seed',min_value=100,value=100,step=1)
st.sidebar.caption('Mission: Gasleck finden · Active Sensing aktiv')
show_truth=st.sidebar.checkbox('Simulator-Ground-Truth anzeigen',value=True)
@st.cache_resource
def predictor(backend, implementation_version):
    d=pd.read_csv(ROOT/'data/synthetic_drone_only.csv');train=d[d.episode_id<int(d.episode_id.nunique()*.75)]
    return Predictor(backend).fit(train)
implementation_version=hashlib.sha256((ROOT/'plumesense.py').read_bytes()+(ROOT/'tunnel.py').read_bytes()).hexdigest()
@st.cache_resource
def source_predictor(backend,version):
    return fit_source_model(backend)
source_version=hashlib.sha256(Path(source_search.__file__).read_bytes()+(ROOT/'source_learning.py').read_bytes()+(ROOT/'data/source_histories.csv').stat().st_mtime_ns.to_bytes(8,'little')).hexdigest()
try:
    model=predictor(backend, implementation_version)
    source_model=source_predictor(backend,source_version)
except Exception as e:
    st.error(f'Modell nicht verfügbar: {e}');st.info('Zuerst python run.py generate ausführen. TabPFN benötigt installierte Abhängigkeiten und Modellzugriff.');st.stop()
if st.sidebar.button('Reset') or 'sim' not in st.session_state or st.session_state.get('key')!=(seed,backend,core_version,source_version):
    sim=Simulation(seed);sim.step();drone=np.array([0,0])
    st.session_state.update(sim=sim,drone=drone,prev=sim.observe(drone),visits=np.zeros(sim.walls.shape),trail=[drone.copy()],exposure=0.,key=(seed,backend,core_version,source_version),now=sim.observe(drone),controller=NavigationController(),search=SourceSearch(sim.walls,sim.wind,source_model))
s=st.session_state
if 'search' not in s or s.get('search_version')!=source_version:
    s.search=SourceSearch(s.sim.walls,s.sim.wind,source_model)
    s.search_version=source_version
if 'controller' not in s or s.get('navigation_version')!=core_version:
    s.controller=NavigationController()
    s.navigation_version=core_version
run=st.sidebar.toggle('Live laufen')
step=st.sidebar.button('Ein Schritt')
now=s.now
try:
    risk,latency=risk_map(s.sim,s.drone,now,s.prev,model)
except Exception as e:
    st.error(str(e));st.stop()
try:
    s.search.update(s.sim.t,now,s.drone)
except Exception as e:
    st.error(f'TabPFN-Quellenschätzung nicht verfügbar: {e}');st.stop()
path=s.search.plan(s.drone,risk,s.visits)
if (step or run) and not s.search.found:
    if len(path)>1:s.drone=np.array(path[1])
    s.exposure+=float(s.sim.gas[s.drone[1],s.drone[0]])
    s.visits[s.drone[1],s.drone[0]]+=1;s.trail.append(s.drone.copy());s.prev=now;s.sim.step();s.now=s.sim.observe(s.drone)
    now=s.now
    risk,latency=risk_map(s.sim,s.drone,now,s.prev,model)
    try:
        s.search.update(s.sim.t,now,s.drone)
    except Exception as e:
        st.error(f'TabPFN-Quellenschätzung nicht verfügbar: {e}');st.stop()
    path=s.search.plan(s.drone,risk,s.visits)
fig=make_subplots(rows=1,cols=3,subplot_titles=['Gaswolke (Simulator)','Gefahrenwahrscheinlichkeit t+4','TabPFN: vermutete Quelle'])
fig.add_trace(go.Heatmap(z=s.sim.gas if show_truth else np.zeros_like(risk),colorscale='YlOrRd',showscale=False),1,1)
fig.add_trace(go.Heatmap(z=risk,zmin=0,zmax=1,colorscale='YlOrRd'),1,2)
fig.add_trace(go.Heatmap(z=s.search.map(),colorscale='Blues',showscale=False),1,3)
for col in [1,2,3]:
    wall=np.where(s.sim.walls,1,np.nan)
    fig.add_trace(go.Heatmap(z=wall,colorscale=[[0,'#333'],[1,'#333']],showscale=False),1,col)
    trail=np.array(s.trail)
    fig.add_trace(go.Scatter(x=trail[:,0],y=trail[:,1],mode='lines',line_color='cyan',showlegend=False),1,col)
    fig.add_trace(go.Scatter(x=[s.drone[0]],y=[s.drone[1]],mode='markers',marker=dict(size=13,color='blue'),name='Drohne',showlegend=col==1),1,col)
    if path:
        route=np.array(path);fig.add_trace(go.Scatter(x=route[:,0],y=route[:,1],mode='lines',line=dict(color='green',dash='dot'),showlegend=False),1,col)
if show_truth:fig.add_trace(go.Scatter(x=[s.sim.source[0]],y=[s.sim.source[1]],mode='markers',marker_symbol='x',name='Quelle (Ground Truth)'),1,1)
fig.update_layout(height=540);fig.update_yaxes(autorange='reversed',dtick=4);fig.update_xaxes(dtick=4)
st.plotly_chart(fig,width='stretch')
c=st.columns(4);c[0].metric('Schritt',s.sim.t);c[1].metric('Exposition (Σ Konzentration)',f'{s.exposure:.3f}');c[2].metric('Inferenz',f'{latency*1000:.0f} ms');c[3].metric('Erwartete Gefahrenfläche',f'{risk.sum():.1f} Zellen')
estimate=tuple(map(int,s.search.estimate))
st.write(f'Wind → ({s.sim.wind[0]:+.2f}, {s.sim.wind[1]:+.2f}) · Drohne {tuple(map(int,s.drone))} · Messziel {s.search.target}')
st.write(f'Suchmodus: {s.search.mode}')
st.caption(f'TabPFN-Quelleninferenz: {s.search.latency*1000:.0f} ms · Historie: {len(s.search.history.rows)} Messungen')
st.write(f'Geschätztes Leck: {estimate} · Anteil normierter Scores im Umkreis von 2 Zellen: {s.search.confidence:.1%}')
if s.search.found:st.success('Leck vermutet: stabile TabPFN-Schätzung und drei positive Bestätigungsmessungen vor Ort.')
else:st.info('Lecksuche läuft: Quelle aus Messhistorie eingrenzen und gezielt nachmessen.')
if show_truth:
    error=float(np.linalg.norm(s.search.estimate-s.sim.source))
    st.caption(f'Nur Simulator-Auswertung: Ortungsfehler {error:.2f} Zellen')
st.caption('Quellenschätzung und Gefahrenprognose: zwei TabPFN-Modelle. Alle bisherigen mobilen Messungen fließen in Historienmerkmale ein. Karte zeigt normierte Quellenscores, keine kalibrierten Wahrscheinlichkeiten. Kein physikalisches Quellenmodell in der Suche.')
if run and not s.search.found:time.sleep(.3);st.rerun()
