"""Abstract BaseAgent and non-blocking AsyncInferenceAgent for ML controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Optional

from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    Vec3,
    WorldInitPayload,
)


class BaseAgent(ABC):
    """Synchronous non-blocking contract required by SimEngine."""

    def __init__(self, drone_id: str) -> None:
        self.drone_id = drone_id

    @abstractmethod
    def reset(self, world_init: WorldInitPayload) -> None:
        """Reset internal state for a new race."""
        ...

    @abstractmethod
    def update(self, obs: DroneObservation) -> Vec3:
        """Return current target_pos immediately (must be non-blocking, < 2ms)."""
        ...

    def pop_belief_update(self) -> Optional[BeliefUpdatePayload]:
        """Optional: return newest belief grid if updated since last call."""
        return None


class AsyncInferenceAgent(BaseAgent):
    """Non-blocking wrapper that runs heavy inference (e.g. TabPFN) in a worker thread.

    Subclasses implement `_on_reset(world_init)` and `_run_inference(obs)` which
    returns `(target_pos, Optional[BeliefUpdatePayload])`. `update(obs)` never blocks
    the 20 Hz physics loop and atomically swaps in new targets/beliefs when ready.
    """

    def __init__(self, drone_id: str, default_target: Vec3 = (0.0, 0.0, 6.0)) -> None:
        super().__init__(drone_id)
        self._default_target: Vec3 = default_target
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"agent-{drone_id}"
        )
        self._generation: int = 0
        self._in_flight: bool = False
        self._current_target: Vec3 = default_target
        self._latest_belief: Optional[BeliefUpdatePayload] = None

    def reset(self, world_init: WorldInitPayload) -> None:
        with self._lock:
            self._generation += 1
            self._in_flight = False
            self._current_target = self._default_target
            self._latest_belief = None
        self._on_reset(world_init)

    @abstractmethod
    def _on_reset(self, world_init: WorldInitPayload) -> None:
        """Reset subclass-specific state on race reset."""
        ...

    @abstractmethod
    def _record_observation(self, obs: DroneObservation) -> Optional[Vec3]:
        """Fast (< 1ms) ingestion of obs on the sim thread.

        Optionally return an immediate waypoint (e.g. during warm-start) to skip
        launching background inference on this step.
        """
        ...

    @abstractmethod
    def _run_inference(self, obs: DroneObservation) -> tuple[Vec3, Optional[BeliefUpdatePayload]]:
        """Heavy inference executed on the single background worker thread."""
        ...

    def update(self, obs: DroneObservation) -> Vec3:
        immediate_target = self._record_observation(obs)
        if immediate_target is not None:
            with self._lock:
                self._current_target = immediate_target
                return self._current_target

        with self._lock:
            if not self._in_flight:
                self._in_flight = True
                gen = self._generation
                self._executor.submit(self._worker, obs, gen)
            return self._current_target

    def _worker(self, obs: DroneObservation, gen: int) -> None:
        try:
            target, belief = self._run_inference(obs)
        except Exception:
            # Never crash the worker thread; keep previous target
            with self._lock:
                if self._generation == gen:
                    self._in_flight = False
            return

        with self._lock:
            if self._generation == gen:
                self._current_target = target
                if belief is not None:
                    self._latest_belief = belief
                self._in_flight = False

    def pop_belief_update(self) -> Optional[BeliefUpdatePayload]:
        with self._lock:
            belief = self._latest_belief
            self._latest_belief = None
            return belief

    def close(self) -> None:
        self._executor.shutdown(wait=False)
