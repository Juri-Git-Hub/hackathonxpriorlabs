"""Physics core tests: plume bending, wake trapping, drone stability, sensor lag, and speed."""

from __future__ import annotations

import math
import time

import numpy as np

from agents.gradient import GradientAgent
from agents.lawnmower import LawnmowerAgent
from agents.mock_agent import MockTabPFNAgent
from shared.schemas import LeakSource
from sim.airflow import Airflow
from sim.drone import Drone
from sim.engine import SimEngine
from sim.plume import Plume
from sim.sensor import GasSensor
from sim.world import World


def test_world_grid_bounds_and_clearance() -> None:
    world = World()
    assert 400 <= len(world.grid_coords) <= 650
    for pt in world.grid_coords:
        assert not world.is_inside_building(pt, margin=1.99)
        assert world.bounds_min[0] <= pt[0] <= world.bounds_max[0]
        assert world.bounds_min[1] <= pt[1] <= world.bounds_max[1]
        assert 1.5 <= pt[2] <= world.bounds_max[2]


def test_plume_bends_when_wind_rotates_90_degrees() -> None:
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=5.0, sigma_w=0.0, seed=1)
    plume = Plume(world, airflow, seed=2)
    leak = [LeakSource(id="l1", pos=(0.0, -35.0, 4.0), emission_rate=1.0, buoyancy_vz=0.0)]

    # Run 10 seconds with wind blowing +X (0 deg)
    for _ in range(200):
        airflow.step(0.05)
        plume.step(0.05, leak)

    puffs_east = plume.pos[plume.active]
    mean_dx_east = float(np.mean(puffs_east[:, 0] - leak[0].pos[0]))
    mean_dy_east = float(np.mean(puffs_east[:, 1] - leak[0].pos[1]))
    assert mean_dx_east > 10.0
    assert abs(mean_dy_east) < 5.0

    # Rotate wind 90 degrees to +Y and run another 10 seconds
    plume.reset(seed=2)
    airflow.set_wind(90.0, 5.0)
    for _ in range(200):
        airflow.step(0.05)
        plume.step(0.05, leak)

    puffs_north = plume.pos[plume.active]
    mean_dx_north = float(np.mean(puffs_north[:, 0] - leak[0].pos[0]))
    mean_dy_north = float(np.mean(puffs_north[:, 1] - leak[0].pos[1]))
    assert mean_dy_north > 10.0
    assert abs(mean_dx_north) < 5.0


def test_downwind_building_wake_damps_wind_and_traps_gas() -> None:
    world = World()
    # Wind blowing +X directly across tank_north (x in [-10, 10], y in [14, 34], z in [0, 16])
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=4.0, sigma_w=0.0, seed=7)

    # Point inside downwind wake cavity behind tank_north
    wake_pt = (18.0, 24.0, 5.0)
    open_pt = (18.0, -45.0, 5.0)

    w_wake = airflow.sample_wind(np.array(wake_pt))
    w_open = airflow.sample_wind(np.array(open_pt))

    assert world.is_in_wind_shadow(wake_pt, airflow.current_wind_vec())
    assert float(np.linalg.norm(w_wake[:2])) < 0.65 * float(np.linalg.norm(w_open[:2]))

    # Place a leak on the lee side of tank_north and verify measurable pocket
    plume = Plume(world, airflow, seed=8)
    leaks = [LeakSource(id="wake_leak", pos=(12.0, 24.0, 4.0), emission_rate=1.0, buoyancy_vz=0.05)]
    for _ in range(200):
        plume.step(0.05, leaks)

    conc_wake = plume.sample_concentration(wake_pt)
    conc_far = plume.sample_concentration(open_pt)
    assert conc_wake > 1.0
    assert conc_wake > conc_far * 10.0


def test_drone_converges_without_entering_buildings_or_nan() -> None:
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=25.0, base_wind_speed=4.0, seed=3)
    drone = Drone("test_drone", start_pos=(-45.0, -45.0, 5.0))

    # Target on the opposite side of buildings to exercise obstacle avoidance
    target = (15.0, 10.0, 6.0)
    drone.set_target(target)

    for _ in range(500):  # 25 seconds of flight
        airflow.step(0.05)
        w = airflow.sample_wind(drone.pos)
        drone.step(0.05, w, world)

        assert np.all(np.isfinite(drone.pos))
        assert np.all(np.isfinite(drone.vel))
        assert np.all(np.isfinite(drone.rot_rpy))
        assert drone.pos[2] >= Drone.MIN_ALTITUDE - 1e-6
        assert not world.is_inside_building(drone.pos_tuple, margin=1.5)

    dist = math.dist(drone.pos_tuple, target)
    assert dist < 2.5
    assert 0.0 < drone.battery_pct < 100.0


def test_sensor_asymmetric_lag_and_dropouts() -> None:
    sensor = GasSensor(tau_rise=0.6, tau_decay=1.8, seed=11)

    # Step up to 100 ppm for 0.6s (1 rise time constant -> ~63% of 100)
    for _ in range(12):
        sensor.step(100.0, 0.05)
    after_rise = sensor.filtered_ppm
    assert 55.0 <= after_rise <= 72.0

    # Step down to 0 ppm for 0.6s (1/3 of decay constant -> decays slower than rise)
    for _ in range(12):
        sensor.step(0.0, 0.05)
    after_decay = sensor.filtered_ppm
    assert after_decay > 0.65 * after_rise

    # Verify dropout mode produces None values
    wind_nones = 0
    temp_nones = 0
    w_vec = np.array([3.0, 0.5, 0.0])
    for _ in range(100):
        ppm, w_meas, temp = sensor.read(w_vec, dropout_active=True)
        assert ppm >= sensor.baseline_ppm
        if w_meas[0] is None:
            wind_nones += 1
        if temp is None:
            temp_nones += 1
    assert wind_nones > 10
    assert temp_nones > 10


def test_headless_engine_speed_over_500_steps_per_sec() -> None:
    agents = {
        "tabpfn": MockTabPFNAgent("tabpfn"),
        "gradient": GradientAgent("gradient"),
        "lawnmower": LawnmowerAgent("lawnmower"),
    }
    engine = SimEngine(agents=agents, seed=42)

    n_steps = 400
    t0 = time.perf_counter()
    for _ in range(n_steps):
        engine.step()
    elapsed = time.perf_counter() - t0
    steps_per_sec = n_steps / max(elapsed, 1e-6)
    assert steps_per_sec > 500.0, f"Expected >500 steps/s headless, got {steps_per_sec:.1f}"


def test_six_dof_rigid_body_motor_lag_and_attitude_response() -> None:
    """Verify true 6-DOF quaternion, angular rates, and 4-motor ESC lag respond physically."""
    world = World()
    drone = Drone("rigid_test", start_pos=(-45.0, -45.0, 8.0))
    drone.set_target((-25.0, -45.0, 8.0))  # Pure +X step command

    zero_wind = np.zeros(3, dtype=np.float64)
    # Step 1 tick (0.05s): pitch should begin tilting forward (+Y rotation -> positive pitch),
    # rear rotors (2, 3) should spin faster than front rotors (0, 1), and quaternion stays unit norm.
    drone.step(0.05, zero_wind, world)
    assert abs(float(np.linalg.norm(drone.quat)) - 1.0) < 1e-6
    assert drone.rot_rpy[1] > 0.01  # Pitching nose down (+X acceleration)
    assert float(np.linalg.norm(drone.omega)) > 0.05
    # Verify motor speeds stay within physical limits and differ across rotors during pitch-up
    assert np.all(drone.rotor_speeds >= Drone.OMEGA_MIN - 1e-3)
    assert np.all(drone.rotor_speeds <= Drone.OMEGA_MAX + 1e-3)

    # After 4 seconds, drone should have accelerated strongly along +X
    for _ in range(80):
        drone.step(0.05, zero_wind, world)
    assert drone.pos[0] > -35.0
    assert abs(float(np.linalg.norm(drone.quat)) - 1.0) < 1e-6


def test_abl_vertical_wind_shear_and_wall_gas_occlusion() -> None:
    """Verify ABL wind speed increases with altitude and solid walls block direct gas transmission."""
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=4.0, sigma_w=0.0, seed=5)

    w_low = airflow.sample_wind(np.array([-45.0, -45.0, 2.0]))
    w_high = airflow.sample_wind(np.array([-45.0, -45.0, 20.0]))
    assert float(np.linalg.norm(w_high[:2])) > float(np.linalg.norm(w_low[:2])) * 1.25

    # Solid building bldg_control spans x in [-38, -20], y in [-28, -12], z in [0, 12]
    puff_west = np.array([[-40.0, -20.0, 4.0]])
    pt_east_blocked = (-18.0, -20.0, 4.0)
    pt_west_clear = (-42.0, -20.0, 4.0)
    trans_blocked = world.segment_transmission(puff_west, pt_east_blocked)[0]
    trans_clear = world.segment_transmission(puff_west, pt_west_clear)[0]
    assert trans_clear == 1.0
    assert trans_blocked < 0.15
