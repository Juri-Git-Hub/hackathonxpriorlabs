"""Master 20 Hz simulation loop and multi-drone race coordinator."""

from __future__ import annotations

import math
import threading
from typing import Optional, Protocol

import numpy as np

from shared.schemas import (
    BeliefUpdatePayload,
    DroneObservation,
    DroneTickState,
    JudgeCommand,
    LeakSource,
    SimTickPayload,
    Vec3,
    WindFieldPayload,
    WorldInitPayload,
)
from sim.airflow import Airflow
from sim.drone import Drone
from sim.plume import Plume
from sim.sensor import GasSensor
from sim.world import World


class AgentProtocol(Protocol):
    drone_id: str

    def reset(self, world_init: WorldInitPayload) -> None: ...

    def update(self, obs: DroneObservation) -> Vec3: ...

    def pop_belief_update(self) -> Optional[BeliefUpdatePayload]: ...


DEFAULT_START_POSITIONS: dict[str, Vec3] = {
    "tabpfn": (-48.0, -45.0, 5.0),
    "gradient": (-48.0, -40.0, 5.0),
    "lawnmower": (-48.0, -35.0, 5.0),
}


class SimEngine:
    """Headless 20 Hz physics engine & multi-drone race coordinator.

    Has zero dependency on FastAPI, WebSockets, Three.js, or serial hardware.
    """

    DT: float = 0.05  # 20 Hz fixed physics step
    OBS_INTERVAL: float = 0.5  # 2 Hz agent observation callback
    LOCK_DISTANCE_M: float = 5.0
    LOCK_SUSTAIN_S: float = 2.0

    def __init__(
        self,
        agents: dict[str, AgentProtocol],
        seed: int = 42,
    ) -> None:
        self._lock = threading.RLock()
        self._seed = seed

        self.world = World()
        self.airflow = Airflow(self.world, seed=seed)
        self.plume = Plume(self.world, self.airflow, seed=seed + 100)

        self.agents: dict[str, AgentProtocol] = dict(agents)
        self.drones: dict[str, Drone] = {}
        self.sensors: dict[str, GasSensor] = {}
        self._lock_timers: dict[str, float] = {}
        self._best_estimate: dict[str, tuple[float, Vec3]] = {}  # (best_ppm, pos)

        self.leaks: list[LeakSource] = []
        self.sim_time: float = 0.0
        self.paused: bool = False
        self.sensor_dropout_active: bool = False

        self._obs_accum: float = 0.0
        self._pending_beliefs: list[BeliefUpdatePayload] = []

        self._init_drones()
        self.reset_race()

    def _init_drones(self) -> None:
        for i, drone_id in enumerate(self.agents):
            start = DEFAULT_START_POSITIONS.get(drone_id, (-48.0, -45.0 + 5.0 * i, 5.0))
            self.drones[drone_id] = Drone(drone_id, start)
            self.sensors[drone_id] = GasSensor(seed=self._seed + 200 + i)

    def get_world_init(self) -> WorldInitPayload:
        return self.world.to_world_init()

    def get_wind_field(self) -> WindFieldPayload:
        with self._lock:
            return self.airflow.get_wind_field_payload()

    def _default_leaks(self) -> list[LeakSource]:
        return [
            LeakSource(
                id="leak_01",
                pos=(-6.0, -7.2, 4.0),  # Near aged flange_01 on central pipe rack
                emission_rate=1.0,
                buoyancy_vz=0.25,
            )
        ]

    def reset_race(self, leaks: list[LeakSource] | None = None) -> None:
        """Reset simulation time, plume, drones, sensors, and all agents."""
        with self._lock:
            self.sim_time = 0.0
            self._obs_accum = 0.0
            self.paused = False
            self._pending_beliefs.clear()

            self.airflow.reset(seed=self._seed)
            self.plume.reset(seed=self._seed + 100)
            self.leaks = list(leaks) if leaks is not None else self._default_leaks()

            # Pre-warm plume for 8 seconds so gas is already dispersed at race start
            for _ in range(int(8.0 / self.DT)):
                self.airflow.step(self.DT)
                self.plume.step(self.DT, self.leaks)

            world_init = self.world.to_world_init()
            for i, (drone_id, drone) in enumerate(self.drones.items()):
                drone.reset()
                self.sensors[drone_id].reset(seed=self._seed + 200 + i)
                self._lock_timers[drone_id] = 0.0
                self._best_estimate[drone_id] = (0.0, drone.pos_tuple)
                self.agents[drone_id].reset(world_init)

    def apply_command(self, cmd: JudgeCommand) -> None:
        """Apply a JudgeCommand thread-safely."""
        with self._lock:
            if cmd.action == "set_wind":
                if cmd.wind_angle_deg is not None and cmd.wind_speed is not None:
                    self.airflow.set_wind(cmd.wind_angle_deg, cmd.wind_speed)
                elif cmd.wind_angle_deg is not None:
                    cur_spd = float(np.linalg.norm(self.airflow.w0))
                    self.airflow.set_wind(cmd.wind_angle_deg, cur_spd)
                elif cmd.wind_speed is not None:
                    cur_ang = math.degrees(math.atan2(self.airflow.w0[1], self.airflow.w0[0]))
                    self.airflow.set_wind(cur_ang, cmd.wind_speed)
            elif cmd.action == "place_leak":
                if cmd.leak_pos is not None:
                    lx, ly, lz = cmd.leak_pos
                    new_id = f"leak_{len(self.leaks) + 1:02d}"
                    self.leaks.append(
                        LeakSource(
                            id=new_id,
                            pos=(float(lx), float(ly), max(1.0, float(lz))),
                        )
                    )
            elif cmd.action == "clear_leaks":
                self.leaks.clear()
            elif cmd.action == "toggle_dropout":
                if cmd.dropout_enabled is not None:
                    self.sensor_dropout_active = bool(cmd.dropout_enabled)
                else:
                    self.sensor_dropout_active = not self.sensor_dropout_active
            elif cmd.action == "reset_race":
                self.reset_race()
            elif cmd.action == "pause":
                self.paused = not self.paused

    def _dist_to_nearest_leak(self, pos: Vec3) -> float:
        if not self.leaks:
            return 999.0
        px, py, pz = pos
        return min(
            math.sqrt((px - lk.pos[0]) ** 2 + (py - lk.pos[1]) ** 2 + (pz - lk.pos[2]) ** 2)
            for lk in self.leaks
        )

    def step(self, dt: float | None = None) -> SimTickPayload:
        """Advance simulation by one fixed physics step (default 0.05s = 20 Hz)."""
        step_dt = self.DT if dt is None else dt

        with self._lock:
            if not self.paused:
                self.sim_time += step_dt
                self.airflow.step(step_dt)
                self.plume.step(step_dt, self.leaks)

                wind_amb = self.airflow.current_wind_vec()
                self._obs_accum += step_dt
                trigger_obs = self._obs_accum >= self.OBS_INTERVAL - 1e-9
                if trigger_obs:
                    self._obs_accum -= self.OBS_INTERVAL

                # Batch local wind query for all drones in a single vectorized call
                drone_items = list(self.drones.items())
                drone_positions = np.array([d.pos for _, d in drone_items], dtype=np.float64)
                drone_winds = self.airflow.sample_wind(drone_positions)

                # Step each drone and its chemical sensor
                for idx, (drone_id, drone) in enumerate(drone_items):
                    wind_local = drone_winds[idx]
                    drone.step(step_dt, wind_local, self.world)

                    true_ppm = self.plume.sample_concentration(drone.pos_tuple)
                    self.sensors[drone_id].step(true_ppm, step_dt)

                    if trigger_obs and drone.status != "depleted":
                        ppm_meas, wind_meas, temp_delta = self.sensors[drone_id].read(
                            wind_local,
                            dropout_active=self.sensor_dropout_active,
                            drone_tilt_rad=drone.tilt_rad,
                        )
                        drone.last_ppm = ppm_meas
                        drone.samples_taken += 1

                        ctx = self.world.query_spatial_context(drone.pos_tuple, wind_amb)
                        obs = DroneObservation(
                            sim_time=self.sim_time,
                            drone_id=drone_id,
                            pos=drone.pos_tuple,
                            vel=drone.vel_tuple,
                            wind_measured=wind_meas,
                            ppm_measured=ppm_meas,
                            temp_delta=temp_delta,
                            nearest_asset_type=ctx["nearest_asset_type"],
                            asset_age_years=ctx["asset_age_years"],
                            in_wind_shadow=ctx["in_wind_shadow"],
                            upwind_pipe_score=ctx["upwind_pipe_score"],
                            battery_pct=drone.battery_pct,
                        )

                        agent = self.agents[drone_id]
                        target = agent.update(obs)
                        drone.set_target(target)

                        # Track peak reading position as fallback source estimate
                        best_ppm, _ = self._best_estimate[drone_id]
                        if ppm_meas > best_ppm:
                            self._best_estimate[drone_id] = (ppm_meas, drone.pos_tuple)

                        belief = agent.pop_belief_update()
                        est_pos = self._best_estimate[drone_id][1]
                        if belief is not None:
                            self._pending_beliefs.append(belief)
                            est_pos = belief.estimated_source

                        # Update lock/tracking status
                        if drone.status != "locked":
                            est_dist = self._dist_to_nearest_leak(est_pos)
                            drone_dist = self._dist_to_nearest_leak(drone.pos_tuple)
                            if (
                                est_dist <= self.LOCK_DISTANCE_M
                                and (ppm_meas > 15.0 or drone_dist <= self.LOCK_DISTANCE_M)
                            ):
                                self._lock_timers[drone_id] += self.OBS_INTERVAL
                                if self._lock_timers[drone_id] >= self.LOCK_SUSTAIN_S:
                                    drone.status = "locked"
                                else:
                                    drone.status = "tracking"
                            else:
                                self._lock_timers[drone_id] = max(
                                    0.0, self._lock_timers[drone_id] - self.OBS_INTERVAL
                                )
                                drone.status = "tracking" if ppm_meas > 5.0 else "searching"

            return self.get_tick_payload()

    def get_tick_payload(self) -> SimTickPayload:
        """Build SimTickPayload snapshot for broadcasting."""
        with self._lock:
            drone_states = [
                DroneTickState(
                    id=d.id,
                    pos=d.pos_tuple,
                    rot_rpy=d.rot_rpy_tuple,
                    target_pos=d.target_pos_tuple,
                    ppm=d.last_ppm,
                    battery_pct=d.battery_pct,
                    samples_taken=d.samples_taken,
                    dist_to_nearest_leak=self._dist_to_nearest_leak(d.pos_tuple),
                    status=d.status,
                )
                for d in self.drones.values()
            ]
            return SimTickPayload(
                sim_time=self.sim_time,
                wind_vec=self.airflow.current_wind_vec(),
                leaks=list(self.leaks),
                drones=drone_states,
                puffs=self.plume.get_puff_states(),
                sensor_dropout_active=self.sensor_dropout_active,
            )

    def pop_belief_updates(self) -> list[BeliefUpdatePayload]:
        """Drain any BeliefUpdatePayloads produced since the last call."""
        with self._lock:
            updates = list(self._pending_beliefs)
            self._pending_beliefs.clear()
            return updates
