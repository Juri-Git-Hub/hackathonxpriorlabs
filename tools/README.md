# Simulation Evaluation, Physics Audit & CLI Control (`tools/`)

This toolkit gives **humans and AI agents** a scriptable, JSON-first interface to:
1. **Verify physics realism** quantitatively against analytical benchmarks (`physics_audit.py`).
2. **Benchmark and debug agents headlessly** at `>500 steps/sec` with JSONL trace exports (`eval_runner.py`).
3. **Control and inspect both headless AND live (non-headless / 3D frontend) simulations** via a single CLI (`sim_ctl.py`).

All commands return exit code `0` on success (`1` on failed physics checks) and support machine-readable JSON output so AI agents can iterate autonomously without parsing visual output.

---

## 1. Physics Realism Audit (`tools/physics_audit.py`)

Runs 10 quantitative checks validating:
- **6-DOF Hover Equilibrium:** altitude hold error, unit quaternion norm (`|q| - 1 < 1e-6`), and analytical hover rotor speed (`sqrt(mg / 4 k_F)`).
- **Step Response & ESC Motor Lag:** finite pitch rise time from 35 ms motor lag, rotor speed spread, tilt/speed envelope compliance, and convergence.
- **Crosswind Attitude Trim:** drone physically leans upwind under a 5.5 m/s crosswind while holding position.
- **Cheeseman-Bennett Ground Effect:** roof vs. open-air height-above-surface and thrust boost factor.
- **Battery Endurance & Depleted Landing:** realistic 3–4 min hover endurance + motor spin-down & descent to `MIN_ALTITUDE` on depletion.
- **ABL Wind Shear & Incompressible 3D Eddies:** empirical power-law exponent (`alpha ≈ 0.18`) and finite-difference `div(u) ≈ 0` in open air.
- **Cylindrical Tank Potential Flow & Wake Cavity:** upwind stagnation (`< 0.75 U`), lateral flank speedup (`> 1.05 U`), and lee-side wake damping (`< 0.45 U`).
- **Plume Buoyancy & Solid-Wall Gas Occlusion:** rising methane vs. sinking heavy gas, and solid building wall blocking direct gas transmission (`< 5%`).
- **Asymmetric Chemical Sensor Lag & Faults:** empirical `1 - 1/e` rise (`tau = 0.6s`) and `1/e` decay (`tau = 1.8s`), saturation (`<= 5000 ppm`), and `~30%` NaN dropout rate.
- **Headless Throughput:** `>500 steps/sec` (`>25x` real-time).

```bash
# Human-readable scorecard
python -m tools.physics_audit

# Machine-readable JSON (for AI agents / CI)
python -m tools.physics_audit --json
# or via unified CLI:
python -m tools.sim_ctl audit --json
```

---

## 2. Headless Multi-Scenario Evaluation (`tools/eval_runner.py`)

Standardized scenarios defined in `tools/scenarios.py`:
- `open_corridor` — single methane leak on aged `flange_01` in steady wind
- `wake_trap` — leak hidden on lee side of `tank_north` (creates false local maxima)
- `wind_shift_90` — wind rotates 90° mid-race at `t = 20s`
- `multi_leak` — two simultaneous leaks on opposite sides of the refinery
- `sensor_dropouts` — 30% `NaN` dropouts on `wind_measured` and `temp_delta`
- `heavy_gas` — dense gas (`buoyancy_vz < 0`) sinking toward ground level
- `high_wind_stress` — strong 7.5 m/s crosswind

```bash
# List all scenarios as JSON
python -m tools.sim_ctl scenarios

# Benchmark mock TabPFN vs Gradient vs Lawnmower across all scenarios & 3 seeds
python -m tools.eval_runner --scenarios all --seeds 42,43,44 --duration 90

# Benchmark REAL TabPFN agent (requires `tabpfn` and `pandas` installed)
python -m tools.eval_runner --scenarios open_corridor,wake_trap,wind_shift_90 --real-tabpfn --json

# Export a JSONL debug trace (2 Hz) to inspect why an agent missed or got stuck
python -m tools.eval_runner --scenarios wake_trap --duration 60 --trace-out /tmp/wake_trace.jsonl
```

**Output metrics per agent:**
`locked`, `time_to_lock_s`, `samples_to_lock`, `min_dist_to_leak_m`, `final_dist_to_leak_m`, `peak_ppm`, `final_battery_pct`, `mean_speed_mps`, `max_tilt_deg`, `final_status`.

---

## 3. Unified CLI Controller (`tools/sim_ctl.py`)

### A. Headless Mode (No Server Required)
```bash
# Run a custom headless race with scheduled mid-run wind shift & dropout injection
python -m tools.sim_ctl headless run \
  --scenario wake_trap \
  --duration 45 \
  --seed 42 \
  --wind-at 15:90:5.0 \
  --dropout-at 25:on \
  --trace-out /tmp/debug_trace.jsonl

# Spatial physics debugger: query exact 3D wind, wake turbulence, roof AGL, spatial context & PPM
python -m tools.sim_ctl headless probe \
  --pos 18,24,5 \
  --wind-angle 0 \
  --wind-speed 4.0 \
  --leak 12,24,4 \
  --warmup 10
```

### B. Live / Non-Headless Mode (Controlling the Running Server & 3D Frontend)
Start the server (`python -m server.app --port 8000`) and optionally open the 3D UI (`http://localhost:5173`). Any AI agent or script can now inspect and steer the live simulation:

```bash
# Check server health & sim_time
python -m tools.sim_ctl live health

# Grab the latest SimTick snapshot (all drones, wind, leaks, puff count) as JSON
python -m tools.sim_ctl live status

# Apply a full scenario preset live (resets race, sets wind, leaks, dropouts)
python -m tools.sim_ctl live scenario wake_trap

# Send individual JudgeCommands live
python -m tools.sim_ctl live cmd set-wind --angle 90 --speed 5.5
python -m tools.sim_ctl live cmd place-leak --pos 15,10,4
python -m tools.sim_ctl live cmd clear-leaks
python -m tools.sim_ctl live cmd toggle-dropout --state on
python -m tools.sim_ctl live cmd pause
python -m tools.sim_ctl live cmd reset-race

# Stream N live ticks as JSONL (to stdout or a file for agent analysis)
python -m tools.sim_ctl live watch --ticks 20 --out /tmp/live_ticks.jsonl
```

---

## 4. Recommended Agent Workflow
1. After editing `sim/`, run `python -m tools.physics_audit --json` and `pytest` to confirm all 10 realism checks and `>500 steps/s` throughput still pass.
2. After editing `agents/tabpfn_agent.py` (or baselines), run `python -m tools.eval_runner --scenarios all --seeds 42,43 --json` to compare `time_to_lock_s`, `samples_to_lock`, and `min_dist_to_leak_m`.
3. If an agent behaves unexpectedly, export `--trace-out /tmp/trace.jsonl` and use `python -m tools.sim_ctl headless probe --pos x,y,z` to inspect the exact local wind, wake cavity, and gas concentration at the drone's coordinates.
