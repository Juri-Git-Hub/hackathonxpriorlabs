"""Bounded planning effort from the drone's own recent wind measurements.

The position controller rejects wind drag and commands ground speed, so adding
wind speed directly to the 8m/s limit would overstate tailwind benefits. Instead
use a small, capped head/crosswind penalty. This is a planning heuristic, not a
calibrated energy or arrival-time prediction, and never estimates the gas source.
"""
import numpy as np

from integration.navigation import Planner3D


class WindAwarePlanner(Planner3D):
    SPEED_SCALE=8.0
    HEADWIND_WEIGHT=.25
    CROSSWIND_WEIGHT=.10

    def __init__(self,world):
        super().__init__(world)
        self.measured_wind=np.zeros(2)
        self.wind_available=False

    def observe_wind(self,history,obs):
        readings=[*history[-7:],obs]
        # Ignore old values during sustained sensor dropouts.
        valid=[o.wind_measured for o in readings
               if obs.sim_time-o.sim_time<=4
               and all(v is not None and np.isfinite(v) for v in o.wind_measured)]
        self.wind_available=bool(valid)
        if not valid:
            self.measured_wind=np.zeros(2)
            return
        wind=np.median(valid,axis=0)
        self.measured_wind=wind*min(1,self.SPEED_SCALE/max(float(np.linalg.norm(wind)),1e-9))

    def wind_penalty(self,a,b):
        delta=(np.asarray(b,float)-a)[:2]
        wx,wy=self.measured_wind
        head=max(0.,-float(np.dot(delta,self.measured_wind)))
        cross=abs(float(delta[0]*wy-delta[1]*wx))
        return (self.HEADWIND_WEIGHT*head+self.CROSSWIND_WEIGHT*cross)/self.SPEED_SCALE**2

    def cost(self,a,b):
        return super().cost(a,b)+self.wind_penalty(a,b)

    def measurement_cost(self,a,b):
        # Same distance scale as the existing target utility; calm/missing wind
        # reproduces its Euclidean cost exactly. Model probability stays primary.
        return float(np.linalg.norm(np.asarray(b,float)-a))+self.SPEED_SCALE*self.wind_penalty(a,b)
