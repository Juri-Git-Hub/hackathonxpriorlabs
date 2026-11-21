"""Observation-only plume acquisition and local 3D measurement planning.

No leak coordinates, concentration-field queries, or source dynamics are used.
The learned source estimate proposes a probe; measured concentrations decide
whether local search advances. Sensor settling and revisits are intentional.
"""
from __future__ import annotations
import numpy as np
from integration.navigation import Planner3D


def safe_goal(planner,point,current=None):
    p=np.clip(np.asarray(point,float),planner.bounds_lo,planner.bounds_hi)
    if planner.free(p):return tuple(map(float,p))
    options=[]
    for lo,hi in zip(planner.lo,planner.hi):
        if np.all(p>=lo) and np.all(p<=hi):
            for axis in range(3):
                for value in [lo[axis]-.25,hi[axis]+.25]:
                    q=p.copy();q[axis]=value
                    if planner.free(q):options.append(q)
    if not options:options=[planner.points[np.linalg.norm(planner.points-p,axis=1).argmin()]]
    # Prefer a nearby horizontal escape over an unnecessary rooftop flight.
    costs=[np.linalg.norm((q-p)*[1,1,1.8]) for q in options]
    return tuple(map(float,options[int(np.argmin(costs))]))


def source_queries(world,history,coarse):
    """Fine source hypotheses are distinct from collision-free drone waypoints."""
    parts=[np.asarray(coarse,float),np.array([a.pos for a in world.assets],float).reshape(-1,3)]
    positive=[o for o in history if o.ppm_measured>3 and history[-1].sim_time-o.sim_time<=60]
    if positive:
        anchor=max(positive,key=lambda o:o.ppm_measured)
        wind=np.array([0 if x is None else x for x in anchor.wind_measured]);norm=np.linalg.norm(wind)
        u=wind/max(norm,.1);cross=np.array([-u[1],u[0]])
        if norm<.2:u=np.array([1.,0.]);cross=np.array([0.,1.])
        along,across,z=np.meshgrid(np.arange(-30,7,3),np.arange(-12,13,3),np.arange(-8,5,2),indexing='ij')
        q=np.tile(anchor.pos,(along.size,1)).astype(float)
        q[:,:2]+=along.ravel()[:,None]*u+across.ravel()[:,None]*cross;q[:,2]+=z.ravel()
        parts.append(q)
        # Highest-signal neighborhood gets 2m XYZ resolution.
        x,y,z=np.meshgrid(np.arange(-6,7,2),np.arange(-6,7,2),np.arange(-6,5,2),indexing='ij')
        parts.append(np.array(anchor.pos)+np.c_[x.ravel(),y.ravel(),z.ravel()])
    allq=np.concatenate(parts)
    lo=np.array(world.bounds_min)+[0,0,1.5];hi=np.array(world.bounds_max)-[0,0,1.5]
    inside=np.all((allq>=lo)&(allq<=hi),axis=1);allq=allq[inside]
    # Preserve coarse-grid prefix for backwards-compatible display arrays.
    seen=set();rows=[]
    for q in allq:
        key=tuple(np.round(q,4))
        if key not in seen:seen.add(key);rows.append(q)
    return np.array(rows)


class SearchController:
    DWELL_SECONDS=2.5
    def __init__(self,world):
        self.world=world;self.planner=Planner3D(world)
        self.history=[];self.goal=None;self.route=[];self.phase='search';self.anchor=None;self.center_ppm=0.
        self.step=6.;self.probes=[];self.cycle_active=False;self.probe_kind='center';self.arrived=None;self.goal_at=0.
        self.source_hint=None;self.source_hint_time=-100.;self.visited=[];self.last_contact=-100.
        self.latest_samples=[];self.completed_probes=0;self.transitions=[]
    def set_source_hint(self,point,sim_time):
        self.source_hint=np.array(point,float);self.source_hint_time=sim_time
    def set_goal(self,point,obs,kind):
        goal=safe_goal(self.planner,point,obs.pos)
        route=self.planner.route(obs.pos,goal)
        if not route:return False
        self.goal=np.array(goal);self.route=route[1:];self.probe_kind=kind;self.arrived=None;self.goal_at=obs.sim_time
        return True
    def waypoint(self,obs):
        while len(self.route)>1 and np.linalg.norm(np.array(self.route[0])-obs.pos)<1.2:self.route.pop(0)
        return self.route[0] if self.route else obs.pos
    def _wind(self,obs):
        valid=[o for o in self.history[-8:] if None not in o.wind_measured]
        wind=np.mean([o.wind_measured for o in valid],axis=0) if valid else np.array([1.,0.])
        u=wind/max(np.linalg.norm(wind),.1)
        return np.r_[u,0],np.r_[-u[1],u[0],0]
    def _survey(self,obs):
        u,cross=self._wind(obs);candidates=[]
        for asset in self.world.assets:
            for distance in [10,25]:
                q=np.array(asset.pos,float)+u*distance;q[2]=np.clip(q[2]+2,3,26)
                candidates.append(safe_goal(self.planner,q,obs.pos))
        # If equipment-focused acquisition finds no plume, expand into the full
        # XYZ volume, including higher and non-asset locations.
        if obs.sim_time>60 and obs.sim_time-self.last_contact>30:
            candidates.extend(self.world.grid_coords)
        if not candidates:candidates=list(self.world.grid_coords)
        grid=np.unique(np.array(candidates),axis=0)
        positions=np.array([o.pos for o in self.history])
        nearest=np.linalg.norm(grid[:,None,:]-positions[None,:,:],axis=2).min(axis=1)
        distance=np.linalg.norm(grid-np.array(obs.pos),axis=1)
        # Sample downwind of known equipment, not an arbitrary source-score maximum.
        utility=(1-np.exp(-nearest/8))/(1+.02*distance)
        for i in np.argsort(utility)[::-1]:
            if self.set_goal(grid[i],obs,'survey'):return
    def _next_probe(self,obs):
        if not self.probes and self.cycle_active:
            self.step=max(1.5,self.step*.6);self.cycle_active=False
            self.set_goal(self.anchor,obs,'center');return
        if not self.probes:
            self.cycle_active=True
            u,cross=self._wind(obs);s=self.step
            deltas=[]
            if self.source_hint is not None and obs.sim_time-self.source_hint_time<8:
                d=self.source_hint-self.anchor;norm=np.linalg.norm(d)
                if norm>1:deltas.append(d*min(1,s/norm))
            deltas += [-u*s,cross*s,-cross*s,np.array([0,0,-s*.6]),np.array([0,0,s*.6]),u*s]
            self.probes=[self.anchor+d for d in deltas]
        while self.probes:
            p=self.probes.pop(0)
            if np.linalg.norm(np.array(safe_goal(self.planner,p))-self.anchor)<1:continue
            if self.set_goal(p,obs,'probe'):return
        self.step=max(1.5,self.step*.6)
        self.set_goal(self.anchor,obs,'center')
    def ingest(self,obs):
        if self.history and obs.sim_time<=self.history[-1].sim_time:return
        self.history.append(obs)
        recent=self.history[-3:];signal=np.median([o.ppm_measured for o in recent])
        if signal>3:self.last_contact=obs.sim_time
        if self.phase=='search' and signal>3:
            self.phase='track';self.anchor=np.array(obs.pos);self.step=6.;self.probes=[]
            self.transitions.append(dict(time=obs.sim_time,phase='track',ppm=float(signal)))
            self.set_goal(obs.pos,obs,'center')
        if self.goal is None:self._survey(obs)
        if obs.sim_time-self.goal_at>15 and np.linalg.norm(np.array(obs.pos)-self.goal)>3:
            # Recover from unreachable/blocked waypoint without relying on leak truth.
            if self.phase=='search':self._survey(obs)
            else:self.set_goal(self.anchor,obs,'center')
        distance=np.linalg.norm(np.array(obs.pos)-self.goal)
        near=distance<3 and np.linalg.norm(obs.vel)<3
        if not near:
            if distance>5:self.arrived=None
            return
        if self.arrived is None:self.arrived=obs.sim_time
        if obs.sim_time-self.arrived<self.DWELL_SECONDS:return
        samples=[o for o in self.history if o.sim_time>=self.arrived+1 and np.linalg.norm(np.array(o.pos)-self.goal)<3]
        if len(samples)<3:return
        samples=samples[-4:];ppm=float(np.median([o.ppm_measured for o in samples]));position=np.median([o.pos for o in samples],axis=0)
        self.visited.append((obs.sim_time,self.goal.copy(),ppm));self.completed_probes+=1
        if self.phase=='search':self._survey(obs);return
        if self.probe_kind=='center':
            self.center_ppm=ppm;self.anchor=position.copy()
        elif ppm>self.center_ppm*1.10+.3:
            self.anchor=position.copy();self.center_ppm=ppm;self.probes=[];self.cycle_active=False
        # Only leave the neighborhood after sustained loss, not after one low sample.
        if obs.sim_time-self.last_contact>30 and ppm<3:
            self.phase='search';self.probes=[];self._survey(obs);return
        self._next_probe(obs)
