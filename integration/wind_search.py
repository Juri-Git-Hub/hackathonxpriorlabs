"""Refresh pre-contact search targets when measured wind changes substantially.

Source predictions and post-contact targets still come from TabPFN. Wind changes
may refresh an outdated coverage target, but never interrupt a committed model
measurement flight. Only local sensor history is available to this controller.
"""
import numpy as np

from integration.prediction_control import PredictionController

DEFAULT_ADAPTIVE_WIND=False


class WindReactiveController(PredictionController):
    WIND_REPLAN_SECONDS=8.0
    DIRECTION_THRESHOLD_DEG=60.0

    def __init__(self,world):
        super().__init__(world)
        self.wind_at_goal=None
        self.wind_replans=0

    def measured_wind(self,obs):
        valid=[o.wind_measured for o in self.history[-8:]
               if obs.sim_time-o.sim_time<=4
               and all(v is not None and np.isfinite(v) for v in o.wind_measured)]
        if len(valid)<3:return None
        wind=np.median(valid,axis=0)
        # Direction in calm or contradictory gust readings is not dependable.
        speeds=np.linalg.norm(valid,axis=1)
        if np.linalg.norm(wind)<.5 or np.mean(speeds>=.5)<.75:return None
        unit=np.asarray(valid)/np.maximum(speeds[:,None],.5)
        if np.linalg.norm(unit.mean(axis=0))<.7:return None
        return wind

    def _survey(self,obs):
        # Record the bearing actually used by the existing coverage planner,
        # including the first valid measurement. Gaining more samples of the
        # same wind must not itself cancel a committed search target.
        u,_=self._wind(obs)
        valid=[o.wind_measured for o in self.history[-8:] if None not in o.wind_measured]
        known=bool(valid) and np.isfinite(valid).all() and np.linalg.norm(np.mean(valid,axis=0))>=.5
        super()._survey(obs)
        self.wind_at_goal=u[:2].copy() if known else None

    def ingest(self,obs):
        if self.history and obs.sim_time<=self.history[-1].sim_time:return
        super().ingest(obs)
        if self.phase!='search' or self.goal is None:return
        if obs.sim_time-self.goal_at<self.WIND_REPLAN_SECONDS:return
        wind=self.measured_wind(obs)
        if wind is None:return
        if self.wind_at_goal is None:
            changed=True
        else:
            old_speed=np.linalg.norm(self.wind_at_goal);speed=np.linalg.norm(wind)
            angle=np.degrees(np.arccos(np.clip(np.dot(wind,self.wind_at_goal)/(speed*old_speed),-1,1)))
            changed=angle>=self.DIRECTION_THRESHOLD_DEG
        if changed:
            self._survey(obs)
            self.wind_replans+=1
            self.decision_basis='wind_adapted_acquisition'
