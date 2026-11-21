"""20 Hz physics independent of Streamlit reruns and remote inference."""
import atexit
import threading
import time
from agents.plumesense_agent import PlumeSenseAgent
from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from sim.engine import SimEngine

class Runtime:
    def __init__(self,seed=42,models=None):
        self.agent=PlumeSenseAgent(models=models)
        self.engine=SimEngine({'tabpfn':self.agent,'gradient':GradientAgent(),'lawnmower':LawnmowerAgent()},seed=seed)
        self.engine.paused=True;self.beliefs={};self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();atexit.register(self.close)
    def _run(self):
        try:
            while not self.stop.is_set():
                start=time.monotonic()
                self.engine.step()
                with self.engine._lock:
                    for b in self.engine.pop_belief_updates():self.beliefs[b.drone_id]=b
                self.stop.wait(max(0,.05-(time.monotonic()-start)))
        except Exception as exc:self.error=str(exc)
    def snapshot(self):
        with self.engine._lock:return self.engine.get_tick_payload(),dict(self.beliefs)
    def close(self):
        if self.stop.is_set():return
        self.stop.set();self.thread.join(timeout=2)
        for agent in self.engine.agents.values():
            close=getattr(agent,'close',None)
            if close:close()
