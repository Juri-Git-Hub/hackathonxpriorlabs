# SeeCH4 — Autonomous 3D Industrial Gas-Leak Detection Simulator

**SeeCH4** is a real-time, physically grounded **3D drone methane (CH₄) leak detection simulator** for complex industrial facilities. It couples a mass-consistent 3D airflow solver, Lagrangian Gaussian puff dispersion, 6-DOF rigid-body quadrotor dynamics, and realistic chemical/anemometer sensors with an interactive **Three.js** 3D visualization and our flagship **SeeCH4 Triple-TabPFN** search agent — plus live side-by-side comparison against classical baseline agents.

---

## Highlights

- **Physics-First 3D Simulation (`sim/`):**
  - **Mass-Consistent 3D Airflow (`sim/airflow.py`):** QUIC-URB–style staggered MAC grid (`48 × 48 × 12`, `2.5 m` voxels) with Red-Black SOR Poisson pressure projection (`div(u) = 0`), atmospheric boundary layer (ABL) power-law wind shear, Röckle lee-side wake cavities, cylindrical tank potential flow & Strouhal shedding, porous pipe-rack drag, divergence-free 3D Fourier eddies, and Ornstein–Uhlenbeck wind gusts.
  - **Lagrangian 3D Gaussian Puff Plume (`sim/plume.py`):** SI mass-to-ppmv ideal-gas calibration (default CH₄), Briggs (1973) urban/industrial Pasquill–Gifford dispersion classes (`A`–`F`), RK2 streamline advection around obstacles, buoyant plume rise, specular ground & mixing-lid reflections, and vectorized solid-wall gas occlusion.
  - **6-DOF Rigid-Body Quadrotor (`sim/drone.py`):** 160 Hz internal sub-stepping (`8×` per 20 Hz tick), 4-motor ESC lag (`35 ms`), Lee et al. geometric `SE(3)` attitude controller, gyroscopic precession, anisotropic body + blade-flapping drag, Cheeseman–Bennett ground effect, rotor downwash dilution, soft-penalty obstacle contact physics, and a physical power/battery model.
  - **Realistic Sensor Suite (`sim/sensor.py`):** Asymmetric 1st-order chemical rise/decay lag (`τ_rise = 0.6 s`, `τ_decay = 1.8 s`), `1.9 ppm` atmospheric CH₄ baseline, multiplicative noise, slow zero-drift, `5000 ppm` saturation, tilt-coupled ultrasonic anemometer noise, and toggleable 30% `NaN` sensor-fault injection.
  - **Procedural & Preset Industrial Worlds (`sim/world.py`, `sim/layout.py`):** Five industrial facility presets (`refinery`, `pipe_loop`, `tank_terminal`, `compressor_yard`, `cooling_works`) plus deterministic procedural facility generation with orthogonal A*-routed connected pipe networks and guaranteed obstacle/launch-pad clearance.
- **SeeCH4 Triple-TabPFN Agent (`agents/reliable_tabpfn.py`):**
  - Runs three specialized **TabPFNRegressor** models asynchronously off the 20 Hz physics loop:
    1. **Model A (Post-Contact 3D Source Inversion):** 29 pure-physics features ranking 3D source hypotheses after gas contact.
    2. **Model B (Pre-Contact Spatial Prior):** 12 features combining facility geometry, wind, and clean-air falsification to guide initial 3D downwind catchment.
    3. **Model C (Online 3D Plume & Hazard Mapper):** Predicts `q10`, `q50`, `q90` in `log1p(ppm)` and 4-second `>10 ppm` hazard fields.
  - Works out of the box with the **free Prior Labs TabPFN API** (`tabpfn-client`, no GPU needed) or **locally on your own computer** (`tabpfn`, CUDA/MPS/CPU).
- **Interactive 3D Frontend & Live Agent Comparison (`frontend/`):**
  - Built with **Three.js** + **Vite**, streaming live 20 Hz ticks, 3D wind vector fields, volumetric plume puffs, drone telemetry, a 2D Flight Map, and live Triple-TabPFN model heatmaps over WebSocket.
  - **Live Compare Agents:** Toggle **Wind-Plume (Baseline)**, **Gradient (Baseline)**, and **Lawnmower (Baseline)** drones on or off directly in the UI to race them side-by-side against **SeeCH4 (Triple-TabPFN)** in real time.
  - **Mission Setup & Judge Controls:** Choose worlds or randomize layouts, adjust wind direction/speed, click to place leaks on pipes/valves/ground, inject 30% `NaN` sensor faults, inspect any object, record video, and toggle clean presentation mode (`P`).
- **Headless Benchmarking & CLI Control (`tools/`):**
  - Quantitative physics realism audit (`tools/physics_audit.py`), standardized multi-scenario evaluation (`tools/eval_runner.py`), and a unified CLI (`tools/sim_ctl.py`) for inspecting and steering both headless and live simulations.

---

## Quickstart

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **Node.js 18+ & npm** (for building or developing the Three.js frontend)

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure TabPFN (Pick One)

SeeCH4 uses **Triple-TabPFN** for live leak localization and automatically detects your backend (`TABPFN_BACKEND=auto`):

- **Option A — Free Prior Labs TabPFN API (no GPU required):**
  ```bash
  pip install tabpfn-client
  python -c "import tabpfn_client; tabpfn_client.init()"
  ```
  *(Or export a token directly via `export TABPFN_TOKEN=...`.)*

- **Option B — Run locally on your own computer (CUDA GPU, Apple Silicon MPS, or CPU):**
  ```bash
  pip install tabpfn
  ```
  *(Optional: force a specific device with `export TABPFN_DEVICE=cuda|mps|cpu`.)*

- **Option C — Custom self-hosted HTTP endpoint:**
  ```bash
  export TABPFN_API_URL=http://127.0.0.1:8001/regress
  ```

### 3. Launch SeeCH4

```bash
python launch_demo.py
```

`launch_demo.py` installs frontend dependencies if needed, builds `frontend/dist`, verifies your TabPFN backend, and serves the Three.js UI, `/ws`, and `/health` on **http://localhost:8501**.

Optional flags:

```bash
# Reproducible layout & simulation seeds on a custom port
python launch_demo.py --port 8502 --seed 99 --layout-seed 73
```

### 4. Frontend Hot-Reload Development

To iterate on the Three.js UI with Vite HMR:

```bash
# Terminal 1: start the FastAPI WebSocket backend
python -m server.app --port 8501

# Terminal 2: start the Vite dev server (proxies /ws and /health to :8501)
cd frontend
npm ci
npm run dev
```

Open **http://localhost:5173** (or append `?mock=true` for standalone frontend development without a Python backend).

---

## Architecture & Conventions

### Single Source of Truth (`shared/schemas.py`)

All cross-module payloads are Pydantic v2 models defined in [shared/schemas.py](shared/schemas.py):

- `WorldInitPayload` (`world_init`) — facility bounds, buildings, storage tanks, equipment assets, connected pipe runs, and 3D candidate grid.
- `WindFieldPayload` (`wind_field`) — coarse 3D mass-consistent wind velocity grid for flow visualization.
- `SimTickPayload` (`sim_tick`) — 20 Hz simulation state: drone 6-DOF poses, telemetry, active leaks, Gaussian puffs, live wind vector, setup/pause state, and completion report.
- `BeliefUpdatePayload` (`belief_update`) — agent source estimate, Triple-TabPFN heatmaps (`tabpfn_search_map`, `tabpfn_source_map`, `hazard_probabilities`), planned 3D route, and diagnostics.
- `BackendStatusPayload` (`backend_status`) — agent mode, readiness, mission state, battery, and latency.
- `JudgeCommand` (`judge_cmd`) — interactive WebSocket commands (`start_mission`, `set_wind`, `place_leak`, `clear_leaks`, `toggle_dropout`, `set_comparison_drones`, `pause`, `step`, `reset_race`, `select_map`, `randomize_layout`).

### Coordinate System

- **Python Physics & Agents:** Right-handed **Z-up**, in meters. Facility volume: `x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]` (ground plane at `z = 0`).
- **Three.js Frontend:** Right-handed **Y-up**. Conversion happens strictly at the rendering boundary (`frontend/src/coords.js`):
  ```text
  three_pos = (sim_x, sim_z, -sim_y)
  ```

### Decoupled Headless Engine

`sim/` and `agents/` have **zero dependency** on FastAPI, WebSockets, or Three.js. `SimEngine` runs deterministically at a fixed `dt = 0.05 s` (20 Hz) with 10 Hz agent observations (`OBS_INTERVAL = 0.1 s`) and verifies mission completion independently from agent internals: a leak is confirmed when the drone and its fresh source estimate remain within `5.0 m` of a true leak with `> 15 ppm` measured concentration for `2.0 s` continuously.

---

## Autonomous Agents (`agents/`)

| Agent | Role | Module | Description |
| :--- | :--- | :--- | :--- |
| **SeeCH4 (Triple-TabPFN)** | **Flagship (always active)** | [agents/tabpfn_agent.py](agents/tabpfn_agent.py), [agents/reliable_tabpfn.py](agents/reliable_tabpfn.py) | Three specialized TabPFN regressors (post-contact 3D source inversion, pre-contact spatial prior, online 3D plume/hazard mapper) + obstacle-aware 3D A* navigation. |
| **Wind-Plume (Baseline)** | Live UI toggle / benchmark | [agents/wind_plume.py](agents/wind_plume.py) | Onboard wind-relative downwind transect search with 3D A* routing and upwind/crosswind/vertical probing. |
| **Gradient (Baseline)** | Live UI toggle / benchmark | [agents/gradient.py](agents/gradient.py) | Bio-inspired anemotactic upwind surge and crosswind cast baseline. |
| **Lawnmower (Baseline)** | Live UI toggle / benchmark | [agents/lawnmower.py](agents/lawnmower.py) | Deterministic 3D volumetric serpentine raster sweep baseline. |
| **Mock Agent** | Unit tests only | [agents/mock_agent.py](agents/mock_agent.py) | Sub-millisecond mock belief generator for headless unit tests without ML weights. |

You can force a specific TabPFN backend at any time with `TABPFN_BACKEND=client|local|http`.

---

## Physics Audit, Headless Benchmarking & CLI (`tools/`)

See [tools/README.md](tools/README.md) for full CLI documentation and JSON output schemas.

```bash
# Quantitative physics realism scorecard (exit code 0 on pass)
python -m tools.physics_audit
python -m tools.sim_ctl audit --json

# Headless multi-scenario benchmark (open_corridor, wake_trap, wind_shift_90, ...)
python -m tools.eval_runner --scenarios open_corridor,wake_trap,wind_shift_90 --duration 60

# Run a headless scenario with mid-run wind shift & export a 2 Hz JSONL trace
python -m tools.sim_ctl headless run --scenario wake_trap --duration 45 --wind-at 15:90:5.0 --trace-out /tmp/trace.jsonl

# Instant 3D spatial physics probe (local wind, wake turbulence, roof AGL, ppm)
python -m tools.sim_ctl headless probe --pos 18,24,5 --wind-angle 0 --wind-speed 4.0 --leak 12,24,4

# Inspect & control a live running server
python -m tools.sim_ctl live health
python -m tools.sim_ctl live status
python -m tools.sim_ctl live scenario wake_trap
python -m tools.sim_ctl live cmd set-wind --angle 90 --speed 5.5
```

---

## Repository Structure

```text
.
├── launch_demo.py              # One-command frontend build + SeeCH4 server launcher
├── shared/
│   ├── schemas.py              # Pydantic v2 WebSocket & simulation data contracts
│   └── flight_geometry.py      # Shared 3D flight clearance & collision geometry
├── sim/                        # Headless 3D physics engine (zero web/UI dependencies)
│   ├── world.py                # 120×120×30 m facility geometry, assets, occlusion & grid
│   ├── layout.py               # Procedural facility layout & orthogonal A* pipe router
│   ├── airflow.py              # Mass-consistent 3D MAC/SOR wind solver & turbulence
│   ├── plume.py                # Vectorized Lagrangian 3D Gaussian puff dispersion
│   ├── drone.py                # 6-DOF rigid-body quadrotor dynamics & SE(3) control
│   ├── sensor.py               # Asymmetric chemical lag, anemometer & fault injection
│   └── engine.py               # 20 Hz master loop, multi-drone coordinator & judge
├── agents/                     # SeeCH4 Triple-TabPFN agent & comparison baselines
│   ├── base.py                 # BaseAgent ABC & non-blocking AsyncInferenceAgent
│   ├── tabpfn_agent.py         # SeeCH4 Triple-TabPFN entry point
│   ├── reliable_tabpfn.py      # Triple-TabPFN spatial localization & mapping policy
│   ├── tabpfn_features.py      # Physics & spatial feature builders for TabPFN
│   ├── tabpfn_nav.py           # Fast 3D navigation helper for TabPFN
│   ├── tabpfn_remote.py        # Unified TabPFN backend (Prior Labs API, local, or HTTP)
│   ├── wind_plume.py           # Wind-Plume baseline agent
│   ├── search_control.py       # Observation-only 3D plume acquisition controller
│   ├── navigation.py           # Swept-segment 3D A* planner & RoutedAgent wrapper
│   ├── gradient.py             # Bio-inspired surge-cast baseline
│   ├── lawnmower.py            # 3D volumetric raster sweep baseline
│   └── mock_agent.py           # Fast mock agent for headless unit tests
├── server/
│   └── app.py                  # FastAPI WebSocket broadcaster (/ws, /health) & static UI
├── frontend/                   # Interactive Three.js + Vite 3D visualization
├── tools/                      # Physics realism audit, scenario benchmarks & sim_ctl CLI
├── scripts/                    # 3D asset generator, TabPFN dataset builder & benchmarks
├── data/                       # Active Triple-TabPFN context datasets
└── tests/                      # Physics, agent, layout, WebSocket & CLI test suites
```

---

## Running Tests

```bash
# Python test suite (physics, agents, layouts, WebSocket contract, CLI tools)
pytest -q

# Frontend HUD unit tests & production build check
node --test frontend/test/*.test.js
npm --prefix frontend run build
```

> **Note:** `tests/test_physics.py::test_headless_engine_speed_over_500_steps_per_sec` is a hardware-dependent throughput benchmark; on slower CPUs you can run the functional suite with `pytest -q -k "not test_headless_engine_speed_over_500_steps_per_sec"`.
