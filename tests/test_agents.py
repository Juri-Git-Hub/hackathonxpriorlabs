"""Agent contract compliance, async non-blocking guarantee, and 3-drone headless race tests."""

from __future__ import annotations

import time
from typing import Optional

from agents.base import AsyncInferenceAgent
from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from agents.mock_agent import MockTabPFNAgent
from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    Vec3,
    WorldInitPayload,
)
from sim.engine import SimEngine
from sim.world import World


class _SlowDummyAsyncAgent(AsyncInferenceAgent):
    """Simulates a slow (60ms) ML model to verify update() never blocks the sim loop."""

    def __init__(self) -> None:
        super().__init__(drone_id="tabpfn", default_target=(0.0, 0.0, 5.0))
        self._grid_len = 0
        self.inferences_completed = 0

    def _on_reset(self, world_init: WorldInitPayload) -> None:
        self._grid_len = len(world_init.grid_coords)
        self.inferences_completed = 0

    def _record_observation(self, obs: DroneObservation) -> Optional[Vec3]:
        return None

    def _run_inference(
        self, obs: DroneObservation
    ) -> tuple[Vec3, Optional[BeliefUpdatePayload]]:
        time.sleep(0.06)
        self.inferences_completed += 1
        target = (10.0, 10.0, 6.0)
        belief = BeliefUpdatePayload(
            sim_time=obs.sim_time,
            drone_id=self.drone_id,
            q50=[0.5] * self._grid_len,
            uncertainty=[0.2] * self._grid_len,
            estimated_source=(5.0, 5.0, 4.0),
            next_waypoint=target,
        )
        return target, belief


def _make_obs(drone_id: str, t: float, ppm: float = 2.0, dropout: bool = False) -> DroneObservation:
    return DroneObservation(
        sim_time=t,
        drone_id=drone_id,
        pos=(0.0, 0.0, 5.0),
        vel=(1.0, 0.0, 0.0),
        wind_measured=(None, None) if dropout else (3.0, 0.5),
        ppm_measured=ppm,
        temp_delta=None if dropout else 0.05,
        nearest_asset_type="valve",
        asset_age_years=15.0,
        in_wind_shadow=False,
        upwind_pipe_score=0.4,
        battery_pct=95.0,
    )


def test_baseline_and_mock_agents_handle_normal_and_dropout_obs() -> None:
    world_init = World().to_world_init()
    agents = [
        LawnmowerAgent("lawnmower"),
        GradientAgent("gradient"),
        MockTabPFNAgent("tabpfn"),
    ]
    for agent in agents:
        agent.reset(world_init)
        for step in range(10):
            obs = _make_obs(
                agent.drone_id,
                t=step * 0.5,
                ppm=12.0 if step % 2 == 0 else 1.5,
                dropout=(step % 3 == 0),
            )
            t0 = time.perf_counter()
            target = agent.update(obs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert elapsed_ms < 10.0
            assert len(target) == 3

        if isinstance(agent, MockTabPFNAgent):
            belief = agent.pop_belief_update()
            assert belief is not None
            assert len(belief.q50) == len(world_init.grid_coords)
            assert len(belief.uncertainty) == len(world_init.grid_coords)


def test_async_inference_agent_never_blocks_and_swaps_atomically() -> None:
    world_init = World().to_world_init()
    agent = _SlowDummyAsyncAgent()
    agent.reset(world_init)
    try:
        t0 = time.perf_counter()
        target = agent.update(_make_obs("tabpfn", t=0.5))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 5.0
        assert target == (0.0, 0.0, 5.0)

        # Wait for background worker to finish
        time.sleep(0.12)
        new_target = agent.update(_make_obs("tabpfn", t=1.0))
        belief = agent.pop_belief_update()
        assert new_target == (10.0, 10.0, 6.0)
        assert belief is not None
        assert len(belief.q50) == len(world_init.grid_coords)
        assert agent.inferences_completed >= 1
    finally:
        agent.close()


def test_headless_three_drone_race() -> None:
    agents = {
        "tabpfn": MockTabPFNAgent("tabpfn"),
        "gradient": GradientAgent("gradient"),
        "lawnmower": LawnmowerAgent("lawnmower"),
    }
    engine = SimEngine(agents=agents, seed=42)
    world_init = engine.get_world_init()

    beliefs_received = 0
    last_tick = None
    for _ in range(200):  # 10 seconds of sim time
        last_tick = engine.step()
        for b in engine.pop_belief_updates():
            assert len(b.q50) == len(world_init.grid_coords)
            assert len(b.uncertainty) == len(world_init.grid_coords)
            beliefs_received += 1

    assert last_tick is not None
    assert len(last_tick.drones) == 3
    assert len(last_tick.puffs) <= 400
    assert beliefs_received >= 5
    for d in last_tick.drones:
        assert d.samples_taken >= 15
        assert d.battery_pct < 100.0


