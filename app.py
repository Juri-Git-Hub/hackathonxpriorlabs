"""Main's 3D physics and race, visualized with our existing Streamlit frontend."""
import hashlib
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agents.plumesense_agent import load_models
from integration.generate import DATA
from integration.runtime import Runtime
from shared.schemas import JudgeCommand

st.set_page_config(page_title='PlumeSense · 3D',layout='wide')
st.title('PlumeSense · 3D Lecksuche')
st.caption('Simulation aus main · 20 Hz Physik · ein mobiler Gassensor pro Drohne · asynchrone TabPFN-Inferenz auf RTX 4090')
seed=st.sidebar.number_input('Seed',min_value=0,value=42,step=1)
reveal=st.sidebar.checkbox('Simulator-Ground-Truth anzeigen',value=False)
@st.cache_resource
def models(version):return load_models()
if not DATA.exists():
    st.info('Zuerst .venv/bin/python -m integration.generate ausführen; 2D-Modelle werden nicht für 3D verwendet.');st.stop()
try:trained=models(hashlib.sha256(DATA.read_bytes()).hexdigest())
except Exception as exc:st.error(f'3D-Modellkontext nicht verfügbar: {exc}');st.stop()
reset=st.sidebar.button('Reset')
if 'runtime' not in st.session_state or st.session_state.get('seed')!=seed or reset:
    if 'runtime' in st.session_state:st.session_state.runtime.close()
    st.session_state.runtime=Runtime(seed,trained);st.session_state.seed=seed
    st.session_state.trails={k:[] for k in ['tabpfn','gradient','lawnmower']}
r=st.session_state.runtime
run=st.sidebar.toggle('Live laufen',value=False)
with r.engine._lock:r.engine.paused=not run
if st.sidebar.button('Ein Schritt'):
    with r.engine._lock:
        r.engine.paused=False
        for _ in range(10):r.engine.step()
        r.engine.paused=not run
if st.sidebar.button('Wind drehen'):
    r.engine.apply_command(JudgeCommand(action='set_wind',wind_angle_deg=90,wind_speed=2))
dropout=st.sidebar.checkbox('Sensorausfälle (Wind/Temperatur)',value=False)
r.engine.apply_command(JudgeCommand(action='toggle_dropout',dropout_enabled=dropout))
tick,beliefs=r.snapshot();world=r.engine.get_world_init();grid=np.asarray(world.grid_coords)
fig=make_subplots(rows=1,cols=3,specs=[[{'type':'scene'}]*3],subplot_titles=['3D-Simulation','TabPFN-Quellenscores','Gefahr in 4 Sekunden'])
# Main uses Z-up coordinates; Plotly also displays Z as altitude.
for col in [1,2,3]:
    for b in world.buildings:
        lo,hi=b.min_xyz,b.max_xyz
        vertices=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        fig.add_trace(go.Mesh3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],i=faces[:,0],j=faces[:,1],k=faces[:,2],color='#777',opacity=.65,showscale=False,name=b.id,showlegend=False),1,col)
    for drone in tick.drones:
        color={'tabpfn':'#00bfff','gradient':'#ff9d00','lawnmower':'#33cc66'}[drone.id]
        if col==1:
            trail=st.session_state.trails[drone.id]
            if not trail or trail[-1]!=drone.pos:trail.append(drone.pos)
        trail=np.array(st.session_state.trails[drone.id][-1500:])
        if len(trail):fig.add_trace(go.Scatter3d(x=trail[:,0],y=trail[:,1],z=trail[:,2],mode='lines',line=dict(color=color,width=3),showlegend=False),1,col)
        fig.add_trace(go.Scatter3d(x=[drone.pos[0]],y=[drone.pos[1]],z=[drone.pos[2]],mode='markers',marker=dict(color=color,size=5),name=drone.id,showlegend=col==1),1,col)
    if reveal:
        for leak in tick.leaks:fig.add_trace(go.Scatter3d(x=[leak.pos[0]],y=[leak.pos[1]],z=[leak.pos[2]],mode='markers',marker=dict(color='red',symbol='x',size=7),showlegend=False),1,col)
if reveal and tick.puffs:
    p=np.array([v.pos for v in tick.puffs]);fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(color='orange',size=5,opacity=.15),showlegend=False),1,1)
for col,values,scale in [(2,r.agent.source_scores,'Blues'),(3,r.agent.hazard,'YlOrRd')]:
    if values is not None:fig.add_trace(go.Scatter3d(x=grid[:,0],y=grid[:,1],z=grid[:,2],mode='markers',marker=dict(size=3,color=values,colorscale=scale,opacity=.5,showscale=False)),1,col)
for i in range(1,4):
    key='scene' if i==1 else f'scene{i}'
    fig.update_layout(**{key:dict(xaxis=dict(range=[-60,60]),yaxis=dict(range=[-60,60]),zaxis=dict(range=[0,30]),aspectmode='data')})
fig.update_layout(height=570,uirevision='plumesense3d',margin=dict(l=0,r=0,t=35,b=0))
st.plotly_chart(fig,width='stretch')
st.write(f'Simulationszeit: {tick.sim_time:.1f} s · Wind: {tick.wind_vec} · Letzte kombinierte Inferenz: {r.agent.latency:.2f} s')
b=beliefs.get('tabpfn')
if b:st.write(f'TabPFN-Quellenschätzung: {tuple(round(x,1) for x in b.estimated_source)} · Messziel: {tuple(round(x,1) for x in b.next_waypoint)}')
st.dataframe([dict(Drohne=d.id,Gas_ppm=round(d.ppm,2),Messungen=d.samples_taken,Status=d.status,Batterie=round(d.battery_pct,1)) for d in tick.drones],hide_index=True)
if r.agent.error:st.error(f'TabPFN-API: {r.agent.error}')
if r.error:st.error(f'Simulationsfehler: {r.error}')
st.caption('Quellenscores sind normierte Klassifikationsscores, keine kalibrierten Wahrscheinlichkeiten. Gefahrenschwelle: 10 ppm, Prognosehorizont: 4 Sekunden. Status „locked“ wird von main mit Ground Truth ausschließlich zur Auswertung vergeben; der Agent erhält diese Daten nicht.')
if run:time.sleep(.2);st.rerun()
