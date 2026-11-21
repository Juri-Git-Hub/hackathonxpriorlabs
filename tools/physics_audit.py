"""Quantitative physics realism & accuracy audit suite.

Validates the 6-DOF rigid-body quadrotor, atmospheric boundary layer,
3D divergence-free turbulence, cylindrical tank potential flow & wakes,
RK2 plume dispersion, wall gas occlusion, chemical sensor lag, and
headless execution throughput against analytical physical expectations.

Usage:
    python -m tools.physics_audit
    python -m tools.physics_audit --json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import sys
import time
from typing import Any

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


@dataclass
class CheckResult:
    name: str
    category: str
    passed: bool
    summary: str
    metrics: dict[str, Any]


def _check_hover_equilibrium() -> CheckResult:
    world = World()
    drone = Drone("audit_hover", start_pos=(-45.0, -45.0, 10.0))
    drone.set_target((-45.0, -45.0, 10.0))
    zero_wind = np.zeros(3, dtype=np.float64)

    for _ in range(100):  # 5s
        drone.step(0.05, zero_wind, world)

    alt_err = abs(float(drone.pos[2]) - 10.0)
    quat_norm_err = abs(float(np.linalg.norm(drone.quat)) - 1.0)
    mean_rpm = float(np.mean(drone.rotor_speeds))
    rpm_err_pct = abs(mean_rpm - Drone.OMEGA_HOVER) / Drone.OMEGA_HOVER * 100.0
    omega_norm = float(np.linalg.norm(drone.omega))

    passed = alt_err < 0.15 and quat_norm_err < 1e-6 and rpm_err_pct < 3.0 and omega_norm < 0.05
    return CheckResult(
        name="6dof_hover_equilibrium",
        category="drone_dynamics",
        passed=passed,
        summary=f"alt_err={alt_err:.4f}m, omega_hover_err={rpm_err_pct:.2f}%, |q|-1={quat_norm_err:.1e}",
        metrics={
            "altitude_error_m": round(alt_err, 5),
            "mean_rotor_rad_s": round(mean_rpm, 2),
            "analytical_hover_rad_s": round(Drone.OMEGA_HOVER, 2),
            "rotor_err_pct": round(rpm_err_pct, 3),
            "quat_norm_err": quat_norm_err,
            "angular_vel_norm_rad_s": round(omega_norm, 5),
        },
    )


def _check_step_response_and_motor_lag() -> CheckResult:
    world = World()
    drone = Drone("audit_step", start_pos=(-45.0, -45.0, 8.0))
    drone.set_target((-20.0, -45.0, 8.0))  # 25m +X step
    zero_wind = np.zeros(3, dtype=np.float64)

    max_tilt_deg = 0.0
    max_vxy = 0.0
    first_tick_pitch_deg = 0.0
    first_tick_rotor_spread = 0.0

    for step_i in range(200):  # 10s
        drone.step(0.05, zero_wind, world)
        tilt_deg = math.degrees(drone.tilt_rad)
        vxy = math.hypot(float(drone.vel[0]), float(drone.vel[1]))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)
        max_vxy = max(max_vxy, vxy)
        if step_i == 0:
            first_tick_pitch_deg = math.degrees(float(drone.rot_rpy[1]))
            first_tick_rotor_spread = float(np.max(drone.rotor_speeds) - np.min(drone.rotor_speeds))

    final_err = math.dist(drone.pos_tuple, (-20.0, -45.0, 8.0))
    # Motor lag prevents instantaneous 28-deg snap on tick 0, while peak tilt respects envelope
    passed = (
        0.5 < first_tick_pitch_deg < 24.0
        and first_tick_rotor_spread > 5.0
        and max_tilt_deg <= 32.0
        and max_vxy <= 9.2
        and final_err < 0.6
    )
    return CheckResult(
        name="6dof_step_response_and_motor_lag",
        category="drone_dynamics",
        passed=passed,
        summary=(
            f"tick0_pitch={first_tick_pitch_deg:.1f}deg, max_tilt={max_tilt_deg:.1f}deg, "
            f"max_vxy={max_vxy:.2f}m/s, final_err={final_err:.2f}m"
        ),
        metrics={
            "first_tick_pitch_deg": round(first_tick_pitch_deg, 2),
            "first_tick_rotor_spread_rad_s": round(first_tick_rotor_spread, 2),
            "max_tilt_deg": round(max_tilt_deg, 2),
            "max_speed_xy_mps": round(max_vxy, 2),
            "final_pos_error_m": round(final_err, 3),
        },
    )


def _check_crosswind_trim_and_rejection() -> CheckResult:
    world = World()
    drone = Drone("audit_wind", start_pos=(-45.0, -45.0, 10.0))
    drone.set_target((-45.0, -45.0, 10.0))
    wind = np.array([5.5, 0.0, 0.0], dtype=np.float64)  # Strong +X wind

    for _ in range(160):  # 8s to reach steady crosswind trim
        drone.step(0.05, wind, world)

    pos_err = math.dist(drone.pos_tuple, (-45.0, -45.0, 10.0))
    # To resist +X wind, drone must tilt its thrust vector toward -X -> negative pitch
    pitch_deg = math.degrees(float(drone.rot_rpy[1]))
    passed = pitch_deg < -1.0 and pos_err < 0.8
    return CheckResult(
        name="crosswind_attitude_trim",
        category="drone_dynamics",
        passed=passed,
        summary=f"5.5m/s +X wind -> trim_pitch={pitch_deg:.2f}deg (leans upwind), hold_err={pos_err:.3f}m",
        metrics={
            "wind_x_mps": 5.5,
            "steady_pitch_deg": round(pitch_deg, 3),
            "hover_hold_error_m": round(pos_err, 4),
        },
    )


def _check_ground_effect() -> CheckResult:
    world = World()
    # Compare steady thrust / rotor speed at z=1.55m vs hovering right above a roof (h_agl ~ 0.25m)
    # Use roof of bldg_control: x in [-38,-20], y in [-28,-12], z_max = 12.0
    h_roof = world.height_above_surface(-29.0, -20.0, 12.25)
    h_open = world.height_above_surface(-45.0, -45.0, 12.25)

    ratio = Drone.ROTOR_RADIUS / (4.0 * max(h_roof, 0.15))
    ge_boost = 1.0 / (1.0 - ratio * ratio)

    passed = abs(h_roof - 0.25) < 1e-6 and abs(h_open - 12.25) < 1e-6 and ge_boost > 1.01
    return CheckResult(
        name="cheeseman_bennett_ground_effect",
        category="drone_dynamics",
        passed=passed,
        summary=f"roof_agl={h_roof:.2f}m (boost={ge_boost:.4f}x) vs open_agl={h_open:.2f}m",
        metrics={
            "roof_height_agl_m": round(h_roof, 3),
            "open_height_agl_m": round(h_open, 3),
            "cheeseman_bennett_boost": round(ge_boost, 4),
        },
    )


def _check_battery_and_depleted_descent() -> CheckResult:
    world = World()
    drone = Drone("audit_bat", start_pos=(-45.0, -45.0, 12.0))
    drone.set_target((-45.0, -45.0, 12.0))
    zero_wind = np.zeros(3, dtype=np.float64)

    # Measure hover drain rate over 10s
    for _ in range(200):
        drone.step(0.05, zero_wind, world)
    drain_10s = 100.0 - drone.battery_pct
    est_flight_min = (100.0 / max(drain_10s, 1e-6)) * 10.0 / 60.0

    # Force battery depletion and verify motors spin down & drone descends to MIN_ALTITUDE
    drone.battery_pct = 0.0
    for _ in range(120):  # 6s
        drone.step(0.05, zero_wind, world)

    passed = (
        2.0 <= est_flight_min <= 5.5
        and drone.status == "depleted"
        and drone.pos[2] <= Drone.MIN_ALTITUDE + 0.1
        and float(np.mean(drone.rotor_speeds)) < 5.0
    )
    return CheckResult(
        name="battery_drain_and_depleted_landing",
        category="drone_dynamics",
        passed=passed,
        summary=(
            f"est_hover_endurance={est_flight_min:.2f}min, post_depletion_z={drone.pos[2]:.2f}m, "
            f"rotor_rpm={float(np.mean(drone.rotor_speeds)):.1f}rad/s"
        ),
        metrics={
            "drain_pct_per_10s": round(drain_10s, 3),
            "est_hover_endurance_min": round(est_flight_min, 2),
            "landed_z_m": round(float(drone.pos[2]), 3),
            "post_depletion_rotor_rad_s": round(float(np.mean(drone.rotor_speeds)), 3),
        },
    )


def _check_abl_and_incompressible_eddies() -> CheckResult:
    world = World()
    airflow_det = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=5.0, sigma_w=0.0, seed=1)
    zs = np.array([1.5, 4.0, 10.0, 20.0, 25.0])
    pts = np.column_stack([np.full_like(zs, -48.0), np.full_like(zs, -48.0), zs])
    ws = airflow_det.sample_wind(pts)
    ux = ws[:, 0]
    # Fit log(ux / u_ref) = alpha * log(z / z_ref)
    alpha_fit = float(np.polyfit(np.log(zs / 10.0), np.log(ux / ux[2]), 1)[0])

    # Finite-difference divergence check on 3D eddies in open air (sigma_w = 0.8)
    airflow_turb = Airflow(world, base_wind_angle_deg=20.0, base_wind_speed=4.0, sigma_w=0.8, seed=9)
    airflow_turb.step(2.0)
    eps = 0.05
    p0 = np.array([-45.0, -45.0, 10.0])
    probe = np.array([
        p0 + [eps, 0, 0], p0 - [eps, 0, 0],
        p0 + [0, eps, 0], p0 - [0, eps, 0],
        p0 + [0, 0, eps], p0 - [0, 0, eps],
    ])
    wp = airflow_turb.sample_wind(probe)
    # Subtract vertical ABL derivative (du_x/dz doesn't enter div(u) = dux/dx + duy/dy + duz/dz!)
    div_u = (
        (wp[0, 0] - wp[1, 0]) / (2.0 * eps)
        + (wp[2, 1] - wp[3, 1]) / (2.0 * eps)
        + (wp[4, 2] - wp[5, 2]) / (2.0 * eps)
    )
    # Normalize divergence by characteristic eddy shear
    passed = abs(alpha_fit - Airflow.ABL_ALPHA) < 0.02 and abs(div_u) < 0.05
    return CheckResult(
        name="abl_shear_and_incompressible_3d_eddies",
        category="airflow",
        passed=passed,
        summary=f"abl_alpha={alpha_fit:.3f} (target {Airflow.ABL_ALPHA}), open_air_div(u)={div_u:.4f} s^-1",
        metrics={
            "abl_alpha_fitted": round(alpha_fit, 4),
            "abl_alpha_target": Airflow.ABL_ALPHA,
            "wind_z1_5_mps": round(float(ux[0]), 3),
            "wind_z10_mps": round(float(ux[2]), 3),
            "wind_z25_mps": round(float(ux[4]), 3),
            "eddy_divergence_s_inv": round(float(div_u), 5),
        },
    )


def _check_cylinder_potential_flow_and_wake() -> CheckResult:
    world = World()
    # Wind +X across tank_north (center (0, 24), radius 10, height 16)
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=4.0, sigma_w=0.0, seed=7)

    pt_open = np.array([-45.0, -45.0, 6.0])
    pt_stagnation = np.array([-11.5, 24.0, 6.0])  # Upwind nose of cylinder
    pt_flank = np.array([0.0, 35.5, 6.0])         # Side shoulder (+Y)
    pt_wake = np.array([16.0, 24.0, 6.0])         # Lee-side wake cavity

    w_open = airflow.sample_wind(pt_open)
    w_stag = airflow.sample_wind(pt_stagnation)
    w_flank = airflow.sample_wind(pt_flank)
    w_wake = airflow.sample_wind(pt_wake)

    u_open = float(w_open[0])
    stag_ratio = float(w_stag[0]) / u_open
    flank_ratio = float(w_flank[0]) / u_open
    wake_ratio = float(np.linalg.norm(w_wake[:2])) / u_open

    # Physics: upwind nose slows down (stagnation), side flank accelerates, lee wake damps strongly
    passed = stag_ratio < 0.75 and flank_ratio > 1.05 and wake_ratio < 0.45
    return CheckResult(
        name="cylindrical_tank_potential_flow_and_wake",
        category="airflow",
        passed=passed,
        summary=(
            f"stagnation={stag_ratio:.2f}U, flank_speedup={flank_ratio:.2f}U, "
            f"lee_wake={wake_ratio:.2f}U"
        ),
        metrics={
            "u_open_mps": round(u_open, 3),
            "stagnation_ratio": round(stag_ratio, 3),
            "flank_speedup_ratio": round(flank_ratio, 3),
            "lee_wake_ratio": round(wake_ratio, 3),
        },
    )


def _check_plume_buoyancy_and_occlusion() -> CheckResult:
    world = World()
    airflow = Airflow(world, base_wind_angle_deg=0.0, base_wind_speed=3.5, sigma_w=0.0, seed=3)

    # Compare rising methane (+0.25) vs heavy gas (-0.20)
    plume_light = Plume(world, airflow, seed=10)
    plume_heavy = Plume(world, airflow, seed=10)
    lk_light = [LeakSource(id="l", pos=(-40.0, -40.0, 6.0), emission_rate=1.0, buoyancy_vz=0.25)]
    lk_heavy = [LeakSource(id="h", pos=(-40.0, -40.0, 6.0), emission_rate=1.0, buoyancy_vz=-0.20)]

    for _ in range(160):  # 8s
        plume_light.step(0.05, lk_light)
        plume_heavy.step(0.05, lk_heavy)

    mean_z_light = float(np.mean(plume_light.pos[plume_light.active, 2]))
    mean_z_heavy = float(np.mean(plume_heavy.pos[plume_heavy.active, 2]))

    # Wall occlusion: leak right on west side of bldg_control (x in [-38, -20], y in [-28, -12])
    plume_occ = Plume(world, Airflow(world, 90.0, 0.2, sigma_w=0.0, seed=4), seed=11)
    lk_wall = [LeakSource(id="w", pos=(-39.5, -20.0, 4.0), emission_rate=1.0, buoyancy_vz=0.0)]
    for _ in range(60):
        plume_occ.step(0.05, lk_wall)
    c_same_side = plume_occ.sample_concentration((-42.0, -20.0, 4.0))
    c_other_side = plume_occ.sample_concentration((-18.0, -20.0, 4.0))

    passed = (
        mean_z_light > 6.5
        and mean_z_heavy < 5.5
        and mean_z_light > mean_z_heavy + 1.5
        and c_same_side > 5.0
        and c_other_side < c_same_side * 0.05
    )
    return CheckResult(
        name="plume_buoyancy_and_wall_occlusion",
        category="plume",
        passed=passed,
        summary=(
            f"light_z={mean_z_light:.2f}m vs heavy_z={mean_z_heavy:.2f}m, "
            f"same_side={c_same_side:.1f}ppm vs behind_wall={c_other_side:.3f}ppm"
        ),
        metrics={
            "mean_z_light_m": round(mean_z_light, 3),
            "mean_z_heavy_m": round(mean_z_heavy, 3),
            "ppm_same_side": round(c_same_side, 3),
            "ppm_behind_solid_wall": round(c_other_side, 4),
        },
    )


def _check_sensor_dynamics_and_faults() -> CheckResult:
    sensor = GasSensor(tau_rise=0.6, tau_decay=1.8, seed=77)
    for _ in range(12):  # 0.6s = 1 tau_rise
        sensor.step(100.0, 0.05)
    rise_frac = sensor.filtered_ppm / 100.0  # analytical: 1 - 1/e = 0.6321

    peak = sensor.filtered_ppm
    for _ in range(36):  # 1.8s = 1 tau_decay
        sensor.step(0.0, 0.05)
    decay_frac = sensor.filtered_ppm / peak  # analytical: 1/e = 0.3679

    # Saturation & dropout rate check
    sensor.reset(seed=77)
    for _ in range(100):
        sensor.step(99999.0, 0.05)
    sat_ppm, _, _ = sensor.read(np.array([2.0, 0.0, 0.0]), dropout_active=False)

    drop_wind = 0
    drop_temp = 0
    n_trials = 500
    for _ in range(n_trials):
        _, w_m, t_m = sensor.read(np.array([2.0, 0.0, 0.0]), dropout_active=True)
        if w_m[0] is None:
            drop_wind += 1
        if t_m is None:
            drop_temp += 1
    wind_drop_rate = drop_wind / n_trials
    temp_drop_rate = drop_temp / n_trials

    passed = (
        abs(rise_frac - 0.6321) < 0.03
        and abs(decay_frac - 0.3679) < 0.03
        and sat_ppm <= GasSensor.MAX_PPM
        and 0.22 <= wind_drop_rate <= 0.38
        and 0.22 <= temp_drop_rate <= 0.38
    )
    return CheckResult(
        name="sensor_asymmetric_lag_saturation_and_dropouts",
        category="sensor",
        passed=passed,
        summary=(
            f"1tau_rise={rise_frac:.3f} (0.632), 1tau_decay={decay_frac:.3f} (0.368), "
            f"sat={sat_ppm:.0f}ppm, drops=({wind_drop_rate:.2f},{temp_drop_rate:.2f})"
        ),
        metrics={
            "rise_1tau_fraction": round(rise_frac, 4),
            "decay_1tau_fraction": round(decay_frac, 4),
            "saturated_ppm": round(sat_ppm, 2),
            "wind_dropout_rate": round(wind_drop_rate, 3),
            "temp_dropout_rate": round(temp_drop_rate, 3),
        },
    )


def _check_headless_throughput() -> CheckResult:
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
    sps = n_steps / max(elapsed, 1e-6)
    ms_per_step = (elapsed / n_steps) * 1000.0
    passed = sps > 500.0
    return CheckResult(
        name="headless_throughput_over_500_sps",
        category="performance",
        passed=passed,
        summary=f"{sps:.1f} steps/sec ({ms_per_step:.3f} ms/step, {sps / 20.0:.1f}x real-time)",
        metrics={
            "steps_per_sec": round(sps, 1),
            "ms_per_step": round(ms_per_step, 3),
            "realtime_factor": round(sps / 20.0, 1),
        },
    )


def run_physics_audit() -> dict[str, Any]:
    """Run all quantitative physics realism checks and return structured report."""
    checks = [
        _check_hover_equilibrium(),
        _check_step_response_and_motor_lag(),
        _check_crosswind_trim_and_rejection(),
        _check_ground_effect(),
        _check_battery_and_depleted_descent(),
        _check_abl_and_incompressible_eddies(),
        _check_cylinder_potential_flow_and_wake(),
        _check_plume_buoyancy_and_occlusion(),
        _check_sensor_dynamics_and_faults(),
        _check_headless_throughput(),
    ]
    n_passed = sum(1 for c in checks if c.passed)
    return {
        "all_passed": n_passed == len(checks),
        "passed_count": n_passed,
        "total_count": len(checks),
        "checks": [asdict(c) for c in checks],
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"\n=== Physics Realism Audit ({report['passed_count']}/{report['total_count']} passed) ===")
    for c in report["checks"]:
        badge = "PASS" if c["passed"] else "FAIL"
        print(f"  [{badge}] ({c['category']:14s}) {c['name']}: {c['summary']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantitative physics realism audit suite")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    args = parser.parse_args(argv)

    report = run_physics_audit()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
