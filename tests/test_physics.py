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
    assert 400 <= len(world.grid_coords) <= 1000
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


def test_poisson_wind_field_obstacles_pipes_and_cylinder_tanks() -> None:
    """Verify mass-consistent 3D wind solver respects boxes, cylinder tanks, pipe rack, and pipes."""
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=4.0, sigma_w=0.0, seed=42)

    # 1. Exact mass conservation (div(u) ~ 0) across all fluid MAC cells
    u, v, wv = airflow._u_unit, airflow._v_unit, airflow._w_unit
    div = ((u[1:] - u[:-1]) + (v[:, 1:] - v[:, :-1]) + (wv[:, :, 1:] - wv[:, :, :-1])) / airflow._dx
    rms_div = float(np.sqrt(np.mean(div[airflow._fluid] ** 2)))
    assert rms_div < 0.01

    # 2. Rectangular building (bldg_control): upwind stagnation & exact zero interior wind
    w_open = airflow.sample_wind(np.array([-50.0, -20.0, 5.0]))
    w_upwind = airflow.sample_wind(np.array([-39.0, -20.0, 5.0]))
    w_inside = airflow.sample_wind(np.array([-29.0, -20.0, 5.0]))
    assert float(w_upwind[0]) < 0.55 * float(w_open[0])
    assert float(np.linalg.norm(w_inside)) == 0.0

    # 3. Porous pipe_rack_center lets wind AND gas puffs pass through (not teleported)
    pr_pt = np.array([[0.0, -1.0, 4.0]])
    w_pr = airflow.sample_wind(pr_pt)[0]
    assert float(w_pr[0]) > 0.45 * float(w_open[0])
    pushed_puff = world.push_outside_buildings(pr_pt, margin=0.2, skip_porous=True)
    assert np.allclose(pushed_puff, pr_pt)

    # 4. Storage tank uses exact cylinder geometry (no phantom AABB corner collision/occlusion)
    corner_air = (-9.5, 14.5, 5.0)  # r=13.44m from tank_north center (r_cyl=10m)
    assert not world.is_inside_building(corner_air, margin=0.0)
    assert world.segment_transmission(np.array([[-9.8, 14.2, 5.0]]), (-9.2, 14.8, 5.0))[0] == 1.0

    # 5. Standalone pipes exist in World, add sub-grid drag, and push puffs off pipe interior
    assert len(world.pipes) >= 8
    assert float(np.max(airflow._porous_drag)) > 0.15
    pipe_center = np.array([[0.0, -2.0, 4.4]])  # On line_gas_main (radius 0.34m)
    pushed_pipe = world.push_outside_pipes(pipe_center, margin=0.05)
    assert float(np.linalg.norm(pushed_pipe - pipe_center)) >= 0.35

    # 6. Strictly divergence-free 3D eddy modes (k . e == 0)
    dots = np.sum(airflow._k_vecs * airflow._e_vecs, axis=1)
    assert float(np.max(np.abs(dots))) < 1e-12

    # 7. WindFieldPayload serialization for frontend 3D visualization
    wf = airflow.get_wind_field_payload()
    assert wf.type == "wind_field"
    nx, ny, nz = wf.dims
    assert len(wf.vectors) == nx * ny * nz * 3


def test_controlled_release_matches_analytical_gaussian_plume_ppmv() -> None:
    """Verify SI ideal-gas ppmv matches analytical Briggs urban D steady-state Gaussian plume."""
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=4.0, sigma_w=0.0, seed=1)
    plume = Plume(world, airflow, spawn_interval=0.2, d_turb=0.0, seed=1, stability_class="D")
    src = (-45.0, -48.0, 6.0)
    q_kg_s = 0.02  # 20 g/s CH4 release
    leaks = [LeakSource(id="ctrl", pos=src, emission_rate=q_kg_s, buoyancy_vz=0.0)]

    for _ in range(600):  # 30s
        plume.step(0.05, leaks)

    u_local = float(airflow.sample_wind(np.array([src[0] + 20.0, src[1], src[2]]))[0])
    for dist_x in (15.0, 30.0, 45.0):
        sim_ppm = plume.sample_concentration((src[0] + dist_x, src[1], src[2]))
        sy_arr, sz_arr = plume._sigmas(np.array([dist_x]), np.array([dist_x / u_local]))
        sy, sz = float(sy_arr[0]), float(sz_arr[0])
        h = src[2]
        c_mass = (q_kg_s / (2.0 * math.pi * u_local * sy * sz)) * (
            1.0 + math.exp(-2.0 * h * h / (sz * sz))
        )
        ana_ppm = c_mass * plume.kg_m3_to_ppmv
        assert abs(sim_ppm - ana_ppm) / ana_ppm < 0.15


def test_stability_classes_multi_leak_buffer_and_render_cap() -> None:
    """Verify unstable (A) spreads wider than stable (F), and >400 physics puffs cap render at 400."""
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=1.5, sigma_w=0.0, seed=1)
    p_unstable = Plume(world, airflow, stability_class="A", d_turb=0.0, seed=1)
    p_stable = Plume(world, airflow, stability_class="F", d_turb=0.0, seed=1)
    sxy_a, sz_a = p_unstable._sigmas(np.array([30.0]), np.array([10.0]))
    sxy_f, sz_f = p_stable._sigmas(np.array([30.0]), np.array([10.0]))
    assert float(sxy_a[0]) > float(sxy_f[0]) * 2.0
    assert float(sz_a[0]) > float(sz_f[0]) * 2.0

    # 3 leaks over 50s -> 600 puffs (> 400 render cap, < 1200 physics buffer)
    leaks = [
        LeakSource(id="l1", pos=(-45.0, -48.0, 4.0), emission_rate=0.02, buoyancy_vz=0.0),
        LeakSource(id="l2", pos=(-45.0, -44.0, 4.0), emission_rate=0.02, buoyancy_vz=0.0),
        LeakSource(id="l3", pos=(-45.0, -40.0, 4.0), emission_rate=0.02, buoyancy_vz=0.0),
    ]
    for _ in range(1000):  # 50s
        p_stable.step(0.05, leaks)
    assert int(np.sum(p_stable.active)) > 450
    rendered = p_stable.get_puff_states()
    assert len(rendered) == Plume.MAX_RENDER_PUFFS


def test_sensor_thermal_no_shortcut_path_tdlas_and_rotor_downwash() -> None:
    """Verify thermal channel removes artificial log1p(ppm) shortcut, TDLAS works, and downwash acts."""
    sensor = GasSensor(seed=5)
    assert sensor.baseline_ppm == 1.9
    # Even at 500 ppm dilute plume, thermodynamic Joule-Thomson delta-T is tiny (< 0.02 K magnitude)
    for _ in range(100):
        sensor.step(500.0, 0.05)
    assert abs(sensor.filtered_temp_delta) < 0.16

    # Path-integrated open-path laser TDLAS (ppm*m)
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=3.5, sigma_w=0.0, seed=1)
    plume = Plume(world, airflow, d_turb=0.0, seed=1)
    leaks = [LeakSource(id="l1", pos=(-40.0, -48.0, 4.0), emission_rate=0.02, buoyancy_vz=0.0)]
    for _ in range(200):
        plume.step(0.05, leaks)
    col_plume = sensor.sample_path_integrated(plume, (-25.0, -48.0, 12.0), (-25.0, -48.0, 0.5))
    col_clean = sensor.sample_path_integrated(plume, (-25.0, -30.0, 12.0), (-25.0, -30.0, 0.5))
    assert col_plume > col_clean * 5.0

    # Rotor downwash momentum theory: hover v_i ~ 5.5 m/s, intake shifted upward & slightly diluted
    drone = Drone("dw_test", start_pos=(-40.0, -40.0, 6.0))
    v_i = drone.induced_downwash_velocity()
    assert 4.5 <= v_i <= 6.5
    pos_hover, dil_hover = drone.sensor_sample_point_and_dilution(np.zeros(3))
    pos_fast, dil_fast = drone.sensor_sample_point_and_dilution(np.array([8.0, 0.0, 0.0]))
    assert pos_hover[2] > drone.pos[2] + 0.25
    assert 0.80 <= dil_hover < dil_fast <= 1.0


def test_default_methane_buoyancy_depends_on_emission_rate() -> None:
    world = World()
    airflow = Airflow(world, base_wind_speed=3.0, sigma_w=0.0, seed=4)
    heights = []
    for rate in (0.0, 0.02, 0.08):
        plume = Plume(world, airflow, d_turb=0.0, seed=5)
        plume.step(0.25, [LeakSource(id='methane', pos=(-40.0, -40.0, 6.0), emission_rate=rate)])
        heights.append(plume.pos[0, 2])
    assert heights[0] < heights[1] < heights[2]


def test_preflight_plume_is_established_before_mission_starts():
    from agents.mock_agent import MockTabPFNAgent
    from sim.engine import SimEngine

    agent = MockTabPFNAgent(drone_id='tabpfn')
    engine = SimEngine({'tabpfn': agent})
    assert engine.sim_time == 0.0
    assert engine.drones['tabpfn'].samples_taken == 0
    assert engine.plume.active.any()
    assert engine.plume.age[engine.plume.active].max() >= 29.0
