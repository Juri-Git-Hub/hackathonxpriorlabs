"""TabPFN chooses the source hypotheses and post-contact measurement goals.

Before any gas contact there is little source information: retain map coverage.
After contact, no gas-gradient rule or fixed directional probe picks a goal.
Fresh model scores choose local measurements; A* only makes the flight feasible.
"""
import numpy as np

from integration.search_control import SearchController, safe_goal


class PredictionController(SearchController):
    MAX_STEP=6.0
    MAX_PREDICTION_AGE=5.0

    def __init__(self, world, wind_aware=False):
        super().__init__(world)
        self.wind_aware=wind_aware
        if wind_aware:
            from integration.wind_planning import WindAwarePlanner
            self.planner=WindAwarePlanner(world)
        self.awaiting_prediction=False
        self.prediction_time=-100.0
        self.prediction_decisions=0
        self.selected_source_score=None
        self.failed_goals=[]
        self.decision_basis='coverage_until_gas_contact'

    def set_source_prediction(self, points, scores, sim_time):
        points=np.asarray(points,float);scores=np.asarray(scores,float)
        if points.ndim!=2 or points.shape[1]!=3 or scores.shape!=(len(points),) or not len(points):
            raise ValueError('Source predictions require one probability for each XYZ hypothesis')
        if not np.isfinite(points).all() or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
            raise ValueError('Source predictions must contain finite probabilities in [0,1]')
        self.prediction_time=sim_time
        self.set_source_hint(points[int(scores.argmax())],sim_time)
        if self.phase!='track' or not self.awaiting_prediction:return
        obs=self.history[-1]
        if obs.sim_time-sim_time>self.MAX_PREDICTION_AGE:return
        current=np.array(obs.pos)
        # Probability is the main utility. Measurement novelty breaks nearby
        # high-score plateaus; travel cost prevents impractical global jumps.
        proposals=[];seen=set()
        recent=np.array([o.pos for o in self.history if obs.sim_time-o.sim_time<=30])
        for i in np.argsort(scores)[::-1][:128]:
            delta=points[i]-current;distance=np.linalg.norm(delta)
            point=current+delta*min(1,self.MAX_STEP/max(distance,1e-9))
            goal=np.array(safe_goal(self.planner,point))
            key=tuple(np.round(goal,2))
            if key in seen:continue
            seen.add(key)
            if any(obs.sim_time-t<30 and np.linalg.norm(goal-p)<2 for t,p in self.failed_goals):continue
            novelty=1-np.exp(-np.linalg.norm(recent-goal,axis=1).min()/3)
            travel=self.planner.measurement_cost(current,goal) if self.wind_aware else np.linalg.norm(goal-current)
            utility=float(scores[i])*(.55+.45*novelty)/(1+.025*travel)
            proposals.append((utility,int(i),goal))
        for _,index,goal in sorted(proposals,key=lambda row:row[0],reverse=True):
            if self.set_goal(goal,obs,'tabpfn'):
                self.awaiting_prediction=False
                self.selected_source_score=float(scores[index])
                self.prediction_decisions+=1
                self.decision_basis='tabpfn_source_prediction'
                return

    def ingest(self, obs):
        if self.history and obs.sim_time<=self.history[-1].sim_time:return
        if self.wind_aware:self.planner.observe_wind(self.history,obs)
        self.history.append(obs)
        signal=float(np.median([o.ppm_measured for o in self.history[-3:]]))
        if signal>3:self.last_contact=obs.sim_time
        if self.phase=='search' and signal>3:
            self.phase='track';self.awaiting_prediction=False
            self.transitions.append(dict(time=obs.sim_time,phase='track',ppm=signal))
            self.decision_basis='sensor_settling'
            self.set_goal(obs.pos,obs,'settle')
        if self.phase=='track' and obs.sim_time-self.last_contact>30:
            self.phase='search';self.awaiting_prediction=False
            self.decision_basis='coverage_until_gas_contact';self._survey(obs)
        if self.goal is None:self._survey(obs)
        distance=np.linalg.norm(np.array(obs.pos)-self.goal)
        if obs.sim_time-self.goal_at>15 and distance>3:
            if self.phase=='search':self._survey(obs)
            else:
                self.failed_goals.append((obs.sim_time,self.goal.copy()))
                self.set_goal(obs.pos,obs,'settle');self.awaiting_prediction=True
                self.decision_basis='waiting_for_tabpfn'
            return
        if self.awaiting_prediction:return
        near=distance<3 and np.linalg.norm(obs.vel)<3
        if not near:
            if distance>5:self.arrived=None
            return
        if self.arrived is None:self.arrived=obs.sim_time
        if obs.sim_time-self.arrived<self.DWELL_SECONDS:return
        samples=[o for o in self.history if o.sim_time>=self.arrived+1 and np.linalg.norm(np.array(o.pos)-self.goal)<3]
        if len(samples)<3:return
        ppm=float(np.median([o.ppm_measured for o in samples[-4:]]))
        self.visited.append((obs.sim_time,self.goal.copy(),ppm));self.completed_probes+=1
        if self.phase=='search':self._survey(obs)
        else:
            self.awaiting_prediction=True
            self.decision_basis='waiting_for_tabpfn'
