# Autonomous 3D Industrial Gas-Leak Detection Simulator

A real-time, physically grounded **3D drone gas-leak detection simulator** for complex industrial facilities. It couples a mass-consistent 3D airflow solver, Lagrangian Gaussian puff dispersion, 6-DOF rigid-body quadrotor dynamics, and realistic chemical/anemometer sensors with an interactive **Three.js** 3D visualization and pluggable autonomous search agents.

---

## Highlights

- **Physics-First 3D Simulation (`sim/`):**
  - **Mass-Consistent 3D Airflow (`sim/airflow.py`):** QUIC-URB–style staggered MAC grid (`48 × 48 × 12`, `2.5 m` voxels) with Red-Black SOR Poisson pressure projection (`div(u) = 0`), atmospheric boundary layer (ABL) power-law wind shear, Röckle lee-side wake cavities, cylindrical tank potential flow & Strouhal shedding, porous pipe-rack drag, divergence-free 3D Fourier eddies, and Ornstein–Uhlenbeck wind gusts.
  - **Lagrangian 3D Gaussian Puff Plume (`sim/plume.py`):** SI mass-to-ppmv ideal-gas calibration (default CH₄), Briggs (1973) urban/industrial Pasquill–Gifford dispersion classes (`A`–`F`), RK2 streamline advection around obstacles, buoyant plume rise, specular ground & mixing-lid reflections, and vectorized solid-wall gas occlusion.
  - **6-DOF Rigid-Body Quadrotor (`sim/drone.py`):** 160 Hz internal sub-stepping (`8×` per 20 Hz tick), 4-motor ESC lag (`35 ms`), Lee et al. geometric `SE(3)` attitude controller, gyroscopic precession, anisotropic body + blade-flapping drag, Cheeseman–Bennett ground effect, rotor downwash dilution, soft-penalty obstacle contact physics, and a physical power/battery model.
  - **Realistic Sensor Suite (`sim/sensor.py`):** Asymmetric 1st-order chemical rise/decay lag (`τ_rise = 0.6 s`, `τ_decay = 1.8 s`), `1.9 ppm` atmospheric CH₄ baseline, multiplicative noise, slow zero-drift, `5000 ppm` saturation, tilt-coupled ultrasonic anemometer noise, and toggleable 30% `NaN` sensor-fault injection.
  - **Procedural & Preset Industrial Worlds (`sim/world.py`, `sim/layout.py`):** Five industrial facility presets (`refinery`, `pipe_loop`, `tank_terminal`, `compressor_yard`, `cooling_works`) plus deterministic procedural facility generation with orthogonal A*-routed connected pipe networks and guaranteed obstacle/launch-pad clearance.
- **Interactive 3D Frontend (`frontend/`):**
  - Built with **Three.js** + **Vite**, streaming live 20 Hz simulation ticks, 3D wind vector fields, plume puffs, drone telemetry, and belief clouds over WebSocket.
  - Live judge controls: adjust wind speed/angle, click to place or relocate leaks at configurable heights, inject 30% `NaN` sensor dropouts, pause/single-step/reset missions, switch worlds or generate randomized layouts, inspect buildings/pipes/valves/drones, and toggle clean presentation mode (`P`).
- **Pluggable Autonomous Agents (`agents/`):**
  - Strict non-blocking async architecture (`AsyncInferenceAgent`) so the 20 Hz physics loop never blocks on inference or 3D A* path planning.
  - Live simulation runs the **Dual-TabPFN** agent (`agents/tabpfn_agent.py` / `agents/reliable_tabpfn.py`) via either the free Prior Labs TabPFN API or local TabPFN on your own computer.
  - Also includes offline benchmark baselines: **Wind-Plume** (`agents/wind_plume.py`), **Bio-Inspired Gradient / Surge-Cast** (`agents/gradient.py`), **3D Volumetric Lawnmower** (`agents/lawnmower.py`), and a fast **Mock** agent (`agents/mock_agent.py`) for unit tests.
- **Headless Benchmarking & CLI Control (`tools/`):**
  - Quantitative 10-check physics realism audit (`tools/physics_audit.py`), standardized multi-scenario evaluation (`tools/eval_runner.py`), and a unified CLI (`tools/sim_ctl.py`) for inspecting and steering both headless and live simulations.

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

The simulator uses **TabPFN** for live leak localization and automatically detects your backend (`TABPFN_BACKEND=auto`):

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

### 3. Launch the Interactive 3D Simulation

```bash
python launch_demo.py
```

`launch_demo.py` installs frontend dependencies if needed, builds `frontend/dist`, verifies your TabPFN backend, and serves the Three.js UI, `/ws`, and `/health` on **http://localhost:8501**.

Choose the scenario in **Situation settings**, then select **Start mission**. Three drones launch from separate positions and split their search targets into west, center, and east sectors at every altitude. After gas contact, a drone can follow the plume beyond its sector. The simulator stops the mission when any drone confirms the source.

For multiple leaks, enable **Keep existing leaks when placing another** in **Situation settings**, then use **Place leak** for each source. Use **Clear leaks** first to start with an empty map. The mission stops at the first confirmed leak.

The **Flight map** shows all three trails and identifies which drone produced the displayed source estimate. **Models** shows the lead drone's search, source, and plume heatmaps.

Run the same layout and simulation seeds in both modes, keeping the leak location and wind settings identical. Compare the confirmed detection times and record any runs that end without confirmation.

```bash
python launch_demo.py --drones 3 --port 8502 --seed 99 --layout-seed 73
python launch_demo.py --drones 1 --port 8503 --seed 99 --layout-seed 73
```

The headless fleet benchmark measures first gas contact (3.5 ppm), first confirmation, and total team samples on paired, seeded pipe-leak scenarios:

```bash
python -m scripts.benchmark_drone_fleet --mock
python -m scripts.benchmark_drone_fleet  # configured TabPFN backend, real-time inference
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
- `SimTickPayload` (`sim_tick`) — 20 Hz simulation state: drone 6-DOF poses, telemetry, active leaks, Gaussian puffs, live wind vector, pause state, and completion report.
- `BeliefUpdatePayload` (`belief_update`) — agent source estimate, belief/uncertainty cloud, planned 3D route, and diagnostics.
- `BackendStatusPayload` (`backend_status`) — agent mode, readiness, mission state, battery, and latency.
- `JudgeCommand` (`judge_cmd`) — interactive WebSocket commands (`set_wind`, `place_leak`, `clear_leaks`, `toggle_dropout`, `pause`, `step`, `reset_race`, `select_map`, `randomize_layout`).

### Coordinate System

- **Python Physics & Agents:** Right-handed **Z-up**, in meters. Facility volume: `x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]` (ground plane at `z = 0`).
- **Three.js Frontend:** Right-handed **Y-up**. Conversion happens strictly at the rendering boundary (`frontend/src/coords.js`):
  ```text
  three_pos = (sim_x, sim_z, -sim_y)
  ```

### Decoupled Headless Engine

`sim/` and `agents/` have **zero dependency** on FastAPI, WebSockets, Three.js, or serial hardware. `SimEngine` runs deterministically at a fixed `dt = 0.05 s` (20 Hz) with 10 Hz agent observations (`OBS_INTERVAL = 0.1 s`) and verifies mission completion independently from agent internals: a leak is confirmed when the drone and its fresh source estimate remain within `5.0 m` of a true leak with `> 15 ppm` measured concentration for `2.0 s` continuously.

---

## Autonomous Agents (`agents/`)

| Agent | Module | Description |
| :--- | :--- | :--- |
| **Dual-TabPFN** *(live standard)* | [agents/tabpfn_agent.py](agents/tabpfn_agent.py), [agents/reliable_tabpfn.py](agents/reliable_tabpfn.py) | Non-blocking TabPFN spatial localization & plume mapping using pre-contact and post-contact physics contexts (`data/`) via the free Prior Labs TabPFN API, local TabPFN, or a custom HTTP endpoint. |
| **Wind-Plume** *(baseline)* | [agents/wind_plume.py](agents/wind_plume.py) | Onboard wind-relative downwind transect search with 3D A* routing and upwind/crosswind/vertical localization. |
| **Gradient / Surge-Cast** *(baseline)* | [agents/gradient.py](agents/gradient.py) | Bio-inspired anemotactic upwind surge and crosswind cast baseline. |
| **3D Lawnmower** *(baseline)* | [agents/lawnmower.py](agents/lawnmower.py) | Deterministic 3D volumetric serpentine raster sweep baseline. |
| **Mock Agent** *(tests)* | [agents/mock_agent.py](agents/mock_agent.py) | Sub-millisecond mock belief generator for headless unit tests without ML weights. |

You can force a specific TabPFN backend at any time with `TABPFN_BACKEND=client|local|http`.

---

## Physics Audit, Headless Benchmarking & CLI (`tools/`)

See [tools/README.md](tools/README.md) for full CLI documentation and JSON output schemas.

```bash
# 10-check quantitative physics realism scorecard (exit code 0 on pass)
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
├── launch_demo.py              # One-command frontend build + FastAPI server launcher
├── shared/
│   └── schemas.py              # Pydantic v2 WebSocket & simulation data contracts
├── sim/                        # Headless 3D physics engine (zero web/UI dependencies)
│   ├── world.py                # 120×120×30 m facility geometry, assets, occlusion & grid
│   ├── layout.py               # Procedural facility layout & orthogonal A* pipe router
│   ├── airflow.py              # Mass-consistent 3D MAC/SOR wind solver & turbulence
│   ├── plume.py                # Vectorized Lagrangian 3D Gaussian puff dispersion
│   ├── drone.py                # 6-DOF rigid-body quadrotor dynamics & SE(3) control
│   ├── sensor.py               # Asymmetric chemical lag, anemometer & fault injection
│   └── engine.py               # 20 Hz master loop, multi-drone coordinator & judge
├── agents/                     # Pluggable autonomous search agents & 3D A* navigation
│   ├── base.py                 # BaseAgent ABC & non-blocking AsyncInferenceAgent
│   ├── wind_plume.py           # Default wind-relative plume tracking agent
│   ├── search_control.py       # Observation-only 3D plume acquisition controller
│   ├── navigation.py           # Swept-segment 3D A* planner & RoutedAgent wrapper
│   ├── tabpfn_agent.py         # TabPFN agent entry point
│   ├── reliable_tabpfn.py      # Dual-TabPFN spatial localization & mapping policy
│   ├── tabpfn_features.py      # Physics & spatial feature builders for TabPFN
│   ├── tabpfn_nav.py           # Fast 3D navigation helper for TabPFN
│   ├── tabpfn_remote.py        # Unified TabPFN backend (Prior Labs API, local, or HTTP)
│   ├── gradient.py             # Bio-inspired surge-cast baseline
│   ├── lawnmower.py            # 3D volumetric raster sweep baseline
│   └── mock_agent.py           # Fast mock agent for headless unit tests
├── server/
│   └── app.py                  # FastAPI WebSocket broadcaster (/ws, /health) & static UI
├── frontend/                   # Interactive Three.js + Vite 3D visualization
├── tools/                      # Physics realism audit, scenario benchmarks & sim_ctl CLI
├── scripts/                    # 3D asset generator, TabPFN dataset builder & benchmarks
├── hardware/
│   └── drone_bridge.py         # Optional isolated WebSocket → serial/ESP32 bridge
├── data/                       # Active TabPFN context datasets
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

---

## Optional Hardware Bridge (`hardware/drone_bridge.py`)

An isolated WebSocket client forwards live telemetry (`PPM` and drone status) over serial to an external microcontroller (e.g., ESP32) without ever blocking or affecting the simulation:

```bash
python hardware/drone_bridge.py --ws-url ws://localhost:8501/ws --port /dev/ttyUSB0
```
