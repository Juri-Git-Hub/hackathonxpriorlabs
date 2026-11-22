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


def test_contact_requires_repeated_weak_readings():
    from types import SimpleNamespace
    from agents.reliable_tabpfn import ReliableTabPFNAgent

    agent = ReliableTabPFNAgent()
    def reading(t, raw, instant=4.):
        return SimpleNamespace(sim_time=t, ppm_raw=raw, ppm_inst=instant, is_clean=False)
    try:
        agent._obs_hist.samples = [reading(1., 6.2), reading(1.1, 2.1)]
        assert not agent._has_recent_contact()
        agent._obs_hist.samples.append(reading(1.2, 6.2))
        assert agent._has_recent_contact()
        agent._obs_hist.samples = [reading(1., 15.)]
        assert agent._has_recent_contact()
    finally:
        agent.close()


def test_feature_history_recognizes_current_clean_air_baseline():
    from agents.tabpfn_features import ObservationHistory

    history = ObservationHistory()
    reading = _make_obs('tabpfn', 1.0).model_copy(update={'ppm_measured': 1.9})
    processed = history.append(reading)
    assert processed.is_clean
    assert processed.ppm_inst == 0.0


def test_tabpfn_backend_adapter_supports_sdk_caching_and_quantiles(monkeypatch):
    import numpy as np
    from agents.tabpfn_remote import RemoteTabPFNRegressor, resolve_backend

    fit_calls = []

    class _FakeRegressor:
        def fit(self, X, y):
            fit_calls.append(len(y))
            return self

        def predict(self, X, output_type="mean", quantiles=None):
            n = len(X)
            if output_type == "quantiles":
                assert quantiles is not None
                return [np.full(n, float(q)) for q in quantiles]
            return np.full(n, 0.42)

    monkeypatch.delenv("TABPFN_API_URL", raising=False)
    monkeypatch.delenv("TABPFN_BACKEND", raising=False)
    assert resolve_backend(api_url="http://127.0.0.1:8001/regress") == "http"

    reg = RemoteTabPFNRegressor(backend="local")
    monkeypatch.setattr(reg, "_create_sdk_model", lambda: _FakeRegressor())

    mean, qs = reg.predict_mean_and_quantiles(
        columns=["a", "b"],
        x_train_rows=[[1.0, 2.0], [3.0, 4.0]],
        y_train=[0.1, 0.9],
        x_test_rows=[[2.0, 3.0], [4.0, 5.0], [5.0, 6.0]],
        quantiles=(0.1, 0.5, 0.9),
        cache_key="ctx",
    )
    assert not reg.last_fit_reused
    assert len(fit_calls) == 1
    assert np.allclose(mean, 0.42)
    assert len(qs) == 3 and all(q.shape == (3,) for q in qs)

    # Second call with same cache_key & training data reuses fitted model
    qs2 = reg.predict_quantiles(
        columns=["a", "b"],
        x_train_rows=[[1.0, 2.0], [3.0, 4.0]],
        y_train=[0.1, 0.9],
        x_test_rows=[[2.0, 3.0]],
        quantiles=(0.25, 0.75),
        cache_key="ctx",
    )
    assert reg.last_fit_reused
    assert len(fit_calls) == 1
    assert len(qs2) == 2 and all(q.shape == (1,) for q in qs2)
