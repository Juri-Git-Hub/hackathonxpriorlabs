"""Versioned observation-only map tables for a future TabPFN search model."""
import itertools
import numpy as np
import pandas as pd
from integration.navigation import Planner3D

VERSION='map-search-v1'
TARGET='target_gas_contact_t4'
CONTACT_PPM=3.0
HORIZON_SECONDS=4.0


def axis_points(low,high,spacing):
    if high<low:raise ValueError('Map is smaller than the flight clearance')
    return np.unique(np.r_[np.arange(low,high,spacing),high])


class MapPlanner3D(Planner3D):
    """A* graph derived from this map's bounds and obstacle heights."""
    def __init__(self,world,spacing=12.):
        self.world=world
        self.bounds_lo=np.asarray(world.bounds_min,float)+[2,2,1.5]
        self.bounds_hi=np.asarray(world.bounds_max,float)-[2,2,1.5]
        self.lo=np.array([b.min_xyz for b in world.buildings],float).reshape(-1,3)-self.CLEARANCE+1e-6
        self.hi=np.array([b.max_xyz for b in world.buildings],float).reshape(-1,3)+self.CLEARANCE-1e-6
        z=axis_points(self.bounds_lo[2],self.bounds_hi[2],8)
        tops=[b.max_xyz[2]+self.CLEARANCE+.25 for b in world.buildings]
        z=np.unique(np.r_[z,[v for v in tops if self.bounds_lo[2]<=v<=self.bounds_hi[2]]])
        self.axes=[axis_points(self.bounds_lo[i],self.bounds_hi[i],spacing) for i in range(2)]+[z]
        self.nodes={}
        for index in itertools.product(*[range(len(a)) for a in self.axes]):
            point=np.array([self.axes[i][index[i]] for i in range(3)])
            if self.free(point):self.nodes[index]=point
        if not self.nodes:raise ValueError('Map has no navigable cells')
        self.indices=list(self.nodes);self.points=np.array(list(self.nodes.values()))
        self.adjacency={}
        self.offsets=[o for o in itertools.product([-1,0,1],repeat=3) if o!=(0,0,0)]

    def route(self,start,goal):
        start=np.asarray(start,float)
        if self.free(start):return super().route(start,goal)
        # The physical boundary backstop permits 0.5 m more than the planning
        # margin. Recover from momentum overshoot without teleporting the drone
        # or accepting a start outside the actual facility / inside an obstacle.
        if np.any(start<self.world.bounds_min) or np.any(start>self.world.bounds_max):return []
        entry=np.clip(start,self.bounds_lo,self.bounds_hi)
        if not self.free(entry) or not self.visible(start,entry):return []
        path=super().route(entry,goal)
        return [tuple(start),*path] if path else []


def candidates(world,planner,spacing_xy=12.,spacing_z=8.):
    lo=planner.bounds_lo.copy();hi=planner.bounds_hi.copy()
    # Measurement targets need braking room; graph nodes may still use the
    # outer corridor to route around structures.
    inset=np.minimum(4.,(hi[:2]-lo[:2])/8)
    lo[:2]+=inset;hi[:2]-=inset
    lo[2]=min(hi[2],lo[2]+1.5)
    axes=[axis_points(lo[0],hi[0],spacing_xy),axis_points(lo[1],hi[1],spacing_xy),axis_points(lo[2],hi[2],spacing_z)]
    points=np.array([p for p in itertools.product(*axes) if planner.free(np.array(p))],float)
    if not len(points):raise ValueError('Map has no search candidates')
    return points


def feature_table(world,points,drone_pos,history=(),wind=None,as_of=None):
    """One row/cell; no leak coordinates, simulator field or future readings.

    Empty history means genuinely unobserved, with explicit missing indicators.
    Target contract for the next dataset step: >=3ppm at query XYZ after 4s.
    """
    q=np.asarray(points,float);pos=np.asarray(drone_pos,float);n=len(q)
    times=[o.sim_time for o in history]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Measurement history must be strictly chronological')
    if times and as_of is not None and times[-1]>as_of:raise ValueError('Future measurements are forbidden')
    lo=np.asarray(world.bounds_min,float);span=np.asarray(world.bounds_max,float)-lo
    if np.any(span<=0):raise ValueError('Map bounds must have positive volume')
    f={}
    for i,a in enumerate('xyz'):
        f[f'query_{a}']=q[:,i];f[f'query_{a}_norm']=(q[:,i]-lo[i])/span[i]
        f[f'drone_{a}']=np.full(n,pos[i]);f[f'query_d{a}']=q[:,i]-pos[i]
        f[f'map_span_{a}']=np.full(n,span[i])
    f['building_count']=np.full(n,len(world.buildings))
    f['travel_distance']=np.linalg.norm(q-pos,axis=1)
    f['travel_seconds_estimate']=np.maximum(np.linalg.norm(q[:,:2]-pos[:2],axis=1)/8,np.abs(q[:,2]-pos[2])/3)
    distances=[]
    for b in world.buildings:
        low=np.asarray(b.min_xyz);high=np.asarray(b.max_xyz)
        distances.append(np.linalg.norm(np.maximum(np.maximum(low-q,q-high),0),axis=1))
    f['wall_distance']=np.min(distances,axis=0) if distances else np.full(n,np.linalg.norm(span))
    f['boundary_distance']=np.minimum(q-lo,lo+span-q).min(axis=1)
    f['ceiling_clearance']=lo[2]+span[2]-q[:,2]
    if world.assets:
        d=np.linalg.norm(q[:,None,:]-np.array([a.pos for a in world.assets])[None,:,:],axis=2)
        index=d.argmin(axis=1);f['asset_distance']=d[np.arange(n),index]
        f['asset_age']=np.array([world.assets[i].age_years for i in index])
        for kind in ['valve','flange','storage_tank','pipe_junction']:
            f[f'asset_{kind}']=np.array([int(world.assets[i].asset_type==kind) for i in index])
    else:
        f['asset_distance']=np.full(n,np.linalg.norm(span));f['asset_age']=np.zeros(n)
        for kind in ['valve','flange','storage_tank','pipe_junction']:f[f'asset_{kind}']=np.zeros(n)
    if wind is None and history:wind=history[-1].wind_measured
    known=wind is not None and all(v is not None and np.isfinite(v) for v in wind)
    w=np.asarray(wind,float) if known else np.zeros(2);speed=np.linalg.norm(w)
    f['wind_x']=np.full(n,w[0]);f['wind_y']=np.full(n,w[1]);f['wind_speed']=np.full(n,speed);f['wind_missing']=np.full(n,int(not known))
    direction=w/max(speed,.1);rel=q[:,:2]-pos[:2]
    f['along_wind']=rel@direction;f['cross_wind']=rel@np.array([-direction[1],direction[0]])
    blocked=np.zeros(n)
    if known and speed>=.1:
        start=q-np.r_[direction*20,0];delta=q-start
        for b in world.buildings:
            enter=np.zeros(n);leave=np.ones(n)
            for axis in range(3):
                if abs(delta[0,axis])<1e-9:
                    outside=(start[:,axis]<b.min_xyz[axis])|(start[:,axis]>b.max_xyz[axis])
                    enter=np.where(outside,np.inf,enter)
                else:
                    a=(b.min_xyz[axis]-start[:,axis])/delta[:,axis];z=(b.max_xyz[axis]-start[:,axis])/delta[:,axis]
                    enter=np.maximum(enter,np.minimum(a,z));leave=np.minimum(leave,np.maximum(a,z))
            blocked+=(enter<=leave)
    f['upwind_obstacle_count']=blocked
    f['measurement_count']=np.full(n,len(history));f['sensor_missing']=np.full(n,int(not history))
    f['sensor_now_log']=np.full(n,np.log1p(history[-1].ppm_measured) if history else 0.)
    f['sensor_prev_log']=np.full(n,np.log1p(history[-2 if len(history)>1 else -1].ppm_measured) if history else 0.)
    if history:
        h=np.array([[*o.pos,np.log1p(o.ppm_measured)] for o in history]);d=np.linalg.norm(q[:,None,:]-h[None,:,:3],axis=2)
        weights=1/np.maximum(d,1)**2
        f['history_idw_log']=weights@h[:,3]/weights.sum(axis=1);f['nearest_measurement_distance']=d.min(axis=1)
        f['local_measurement_count']=(d<=6).sum(axis=1)
    else:
        f['history_idw_log']=np.zeros(n);f['nearest_measurement_distance']=np.full(n,np.linalg.norm(span));f['local_measurement_count']=np.zeros(n)
    return pd.DataFrame(f,dtype=float)
