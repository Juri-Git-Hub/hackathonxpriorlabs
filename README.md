# SeeCH4 — Autonomous Methane-Leak Detection Drones Powered by Triple-TabPFN

**SeeCH4** is an autonomous **3D methane (CH₄) leak-hunting drone** powered by **Triple-TabPFN** — three specialized tabular foundation models working together in real time to explore industrial facilities, invert turbulent gas plumes back to their 3D source, and forecast downwind methane hazards. To develop, benchmark, and demonstrate the drone safely, SeeCH4 includes a real-time 3D physics testbed and interactive **Three.js** viewer where you can race **SeeCH4 (Triple-TabPFN)** live against classical baseline drones.

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

SeeCH4 uses **Triple-TabPFN** for live leak localization and automatically detects your backend (`TABPFN_BACKEND=auto`).

> **Performance Recommendation:** Because SeeCH4 runs three `TabPFNRegressor` models continuously during live flight, **we strongly recommend running locally on a capable PC with a dedicated GPU (e.g. NVIDIA CUDA) or connecting over HTTP to a fast GPU machine** for low-latency, responsive inference. The free hosted API is convenient when no GPU is available, but has higher network/queue latency.

- **Option A — Run locally on your own computer (Recommended with GPU):**
  ```bash
  pip install tabpfn
  ```
  *(Optional: force a specific device with `export TABPFN_DEVICE=cuda|mps|cpu`.)*

- **Option B — Custom self-hosted HTTP GPU server (Recommended for remote GPU):**
  ```bash
  export TABPFN_API_URL=http://<gpu-host>:8001/regress
  ```

- **Option C — Free Prior Labs TabPFN API (no GPU required, higher latency):**
  ```bash
  pip install tabpfn-client
  # Log in via browser (saves your token locally):
  python -c "from tabpfn_client import interactive_login; interactive_login()"
  ```
  *(Or generate a token at https://platform.priorlabs.ai/account/api-keys and run `export TABPFN_TOKEN="<your-token>"`.)*

### 3. Launch SeeCH4

```bash
python launch_demo.py
```

`launch_demo.py` installs frontend dependencies if needed, builds `frontend/dist`, verifies your TabPFN backend, picks a free local port (starting at `8501`), and prints the URL to open in your browser.

```bash
# Optional: reproducible layout & simulation seeds on a custom port
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

Open **http://localhost:5173** (or append `?mock=true` for standalone UI development without a Python backend).

---

## Highlights

- **Triple-TabPFN Autonomous Drone (`agents/reliable_tabpfn.py`):** Combines three specialized in-context `TabPFNRegressor` models (post-contact 3D source inversion, pre-contact spatial prior, online 3D plume/hazard forecasting) with causal chemical-sensor deconvolution and 3D A* obstacle-aware flight planning.
- **Zero-Leakage Multi-World Dataset Pipeline (`scripts/generate_tabpfn_datasets.py`):** Offline TabPFN contexts are collected strictly on unseen procedural layouts using source-blind mobile trajectories and shared feature code, preventing facility memorization or train/eval leakage.
- **Live Side-by-Side Drone Comparison (`frontend/`):** Toggle **Wind-Plume (Baseline)**, **Gradient (Baseline)**, and **Lawnmower (Baseline)** drones on or off directly in the UI to compare trajectories, gas readings, and confirmation times against **SeeCH4 (Triple-TabPFN)** in real time (both in 3D and on the 2D Flight Map).
- **High-Fidelity 3D Physics Testbed (`sim/`):** Mass-consistent 3D MAC/SOR wind solver, SI-calibrated Lagrangian Gaussian puff dispersion, 6-DOF rigid-body quadrotor dynamics with 160 Hz sub-stepping, realistic asymmetric chemical/anemometer sensors, and procedural industrial facilities.
- **Headless Evaluation & Physics Audit (`tools/`, `scripts/`):** Quantitative 10-check physics realism scorecard, standardized multi-scenario benchmarks, and a JSON-first CLI (`tools/sim_ctl.py`) for headless and live control.

---

## Technical Deep Dive

### 1. Triple-TabPFN Autonomous Localization Architecture

SeeCH4 replaces brittle heuristic plume-chasing with three coordinated **TabPFNRegressor** models orchestrated by [agents/reliable_tabpfn.py](agents/reliable_tabpfn.py) using the shared feature engine in [agents/tabpfn_features.py](agents/tabpfn_features.py):

| Model | Context Dataset (`data/`) | Context Size | Features | Role & Cadence |
| :--- | :--- | :---: | :---: | :--- |
| **Model A — Post-Contact 3D Source Inversion** | `tabpfn_source_post_warm30_v1.json.gz` | 4,096 rows (240 episodes) | 21 | Ranks 3D source hypotheses after gas contact using pure plume-inversion physics and within-snapshot comparative features (`1.5 s` cadence). |
| **Model B — Pre-Contact Spatial Prior** | `tabpfn_source_pre_v4.json.gz` | 1,600 rows (120 episodes) | 12 | Guides Phase 1 exploration before gas contact using facility geometry, wind shadows, and clean-air falsification (`2.0 s` cadence). |
| **Model C — Online 3D Plume & Hazard Mapper** | `tabpfn_plume_prior.json` + live flight | 157 prior + ≤90 live rows | 14 | Predicts `q10`, `q50`, `q90` in `log1p(ppm)` and 4-second `P(ppm ≥ 10)` hazard fields across the 3D facility grid (`5.0 s` cadence). |

#### Causal Sensor-Lag Deconvolution (`ObservationHistory`)

Real UAV methane analyzers have asymmetric response times (`τ_rise = 0.60 s`, `τ_decay = 1.80 s`) over the `1.90 ppm` Earth atmospheric CH₄ baseline. Raw readings smear plumes downwind and create false positives after exiting a puff.

`ObservationHistory` in [agents/tabpfn_features.py](agents/tabpfn_features.py) causally deconvolves each incoming 10 Hz observation into an instantaneous concentration estimate `ppm_inst` and classifies every sample into:
- **`is_clean`** — confirmed clean air (including low-ppm sensor desorption washout tails `is_washout`), used to falsify upwind source hypotheses whose predicted plumes should have hit clean-air samples.
- **`is_transient`** — rapid concentration drops while still inside high-ppm gas (`≥ 10 ppm`), explicitly protected from being mislabeled as clean air during turbulent puff intermittency.
- **Causal Ambient Wind Estimator** — filters out `NaN` anemometer dropouts and building wind-shadow distortions to maintain a rolling ambient wind vector estimate.

#### Feature Engineering & Zero-Shortcut Physics

- **Model A (21 post-contact features — `POST_SOURCE_FEATURE_COLUMNS`):** Deliberately excludes `dist_to_asset` and `min_obs_dist` shortcuts so TabPFN cannot cheat by hugging equipment or hovering near past samples. Features are grouped into:
  1. *Global calibrated 3-buoyancy MAP Gaussian plume fit (7):* `plume_cos_best`, `plume_r2_q1`, `plume_r2_qhat`, `plume_rmse_qhat`, `hit_rmse_qhat`, `abs_log_q_hat`, `hit_explain_frac` — evaluated across a discrete emission-rate grid `Q ∈ {0.45, 0.70, 1.00, 1.40, 2.00}` against the drone's deconvolved observation history.
  2. *Local-peak-window MAP fit (2):* `local_r2_qhat`, `local_r2_gap_to_best` — resilient to wind shifts and multi-leak interference.
  3. *Strictly monotonic within-snapshot comparative physics (6):* `r2_gap_to_best`, `r2_q1_gap_to_best`, `rmse_ratio_to_best`, `cos_gap_to_best`, `explain_gap_to_best`, `upwind_hit_ratio_to_best` — normalizes candidate scores relative to the best physics explanation in the current snapshot so TabPFN generalizes across unseen wind speeds and emission strengths.
  4. *Directional, peak geometry & bounded infrastructure tie-breaker (6):* `unexplained_hit_log`, `upwind_gas_mass_frac`, `downwind_clear_core_frac`, `peak_cross_norm`, `peak_dz_buoy_min`, `infra_prox`.
- **Model B (12 pre-contact features — `PRE_SOURCE_FEATURE_COLUMNS`):** `nearest_asset_type`, `dist_to_asset`, `asset_age_years`, `dist_to_pipe`, `pipe_pressure_bar`, `pipe_is_gas`, `wall_distance`, `in_wind_shadow`, `z`, `downwind_clear_score`, `onsite_clear_score`, `min_obs_dist`.
- **Model C (14 plume/hazard features — `PLUME_FEATURE_COLUMNS`):** `mission_id`, `x`, `y`, `z`, `wind_x`, `wind_y`, `nearest_asset_type`, `dist_to_asset`, `asset_age_years`, `dist_to_pipe`, `pipe_pressure_bar`, `pipe_is_gas`, `in_wind_shadow`, `upwind_pipe_score`. Prior rows are coordinate-masked so TabPFN transfers spatial-wind structure to arbitrary layouts while conditioning on the live mission's own `mission_id` observations.

#### Two-Scale Upwind Candidate Generation & Active Probing

After gas contact, `build_post_source_candidates` constructs a non-truncating two-scale 3D candidate set (coarse facility grid + fine upwind corridor around peak gas detections, preserving exact indices back to the world display grid). Model A scores all candidates with quantile/mean predictions, and `ReliableTabPFNAgent` plans either:
- **Disambiguation / confirmation probes** (upwind, crosswind, downwind, or vertical) when multiple hypotheses remain competitive, or
- **High-ppm core hold & lock** when the estimated source converges within confirmation range.

---

### 2. Offline Data Collection & Context Generation (`scripts/generate_tabpfn_datasets.py`)

Because TabPFN performs **in-context learning** (fitting directly on the tabular context at inference time rather than gradient-trained weights), the quality, diversity, and leakage-freedom of the context datasets in `data/` are critical:

- **Strict Train / Evaluation Separation:** Context datasets are generated **exclusively** on procedural `generated` layouts (`layout_seed = 1000..1419`). The 5 named evaluation facilities (`refinery`, `pipe_loop`, `tank_terminal`, `compressor_yard`, `cooling_works`) and held-out procedural test seeds are never seen during data collection.
- **Exact Train/Live Feature Parity:** Both offline collection and live flight call the exact same `FacilityContext`, `ObservationHistory`, `build_post_source_candidates`, `build_post_source_feature_rows`, and `build_pre_source_feature_rows` functions in [agents/tabpfn_features.py](agents/tabpfn_features.py).
- **Source-Blind Multi-Behavior Collector (`_TrainingCollector`):** Collects trajectories across 5 flight regimes — `converge`, `linger` (simulating realistic downwind plume traps), `gradient`, `mixed`, and `lawnmower` — across varied wind speeds, wind shifts, leak heights, and emission rates.
- **Continuous Proximity Targets (`source_proximity_target`):** Regression targets encode smooth 3D proximity to the true leak, achieving `94.8%` candidate coverage across the 240 post-contact episodes (`4,096` stratified rows for Model A, `1,600` rows for Model B, and `157` prior rows for Model C).

To regenerate the datasets from scratch:

```bash
python -m scripts.generate_tabpfn_datasets --episodes 240 --workers 10
```

---

### 3. 3D Obstacle-Aware Navigation & Non-Blocking Execution

- **3D A* Planner (`agents/tabpfn_nav.py`):** `FastPlanner3D` builds a collision-free 3D roadmap over `WorldInitPayload` using shared clearance geometry (`shared/flight_geometry.py`, `FLIGHT_CLEARANCE + 0.5 m`), automatically adds rooftop altitude layers for elevated leaks and face-midpoint portals around buildings to keep narrow corridors connected, and uses swept-segment AABB visibility checks plus a `0.75 m` dense local lattice (`safe_goal_3d`) for tight-space goal projection.
- **Strict Async Isolation (`agents/base.py`):** `SimEngine` steps physics at a fixed **20 Hz** (`dt = 0.05 s`) with **10 Hz** agent observations (`OBS_INTERVAL = 0.1 s`). `ReliableTabPFNAgent` subclasses `AsyncInferenceAgent`, running all TabPFN inference and A* path planning in a dedicated background worker thread and atomically swapping `target_pos` and `BeliefUpdatePayload` so physics never stalls.
- **Unified TabPFN Backend Adapter (`agents/tabpfn_remote.py`):** Supports three backends via `TABPFN_BACKEND=auto|local|http|client`:
  - `local`: In-process `tabpfn.TabPFNRegressor` with in-memory `(cache_key, data_hash)` fit caching.
  - `http`: Custom remote GPU `/regress` server with SHA-256 context-reference caching (avoids resending static 4,096-row contexts after the first call).
  - `client`: Prior Labs hosted API (`tabpfn-client`) with the same in-memory fit cache.

---

### 4. High-Fidelity 3D Physics Testbed (`sim/`)

The simulation engine has **zero dependency** on FastAPI, WebSockets, or Three.js and runs deterministically headless for fast evaluation:

- **Mass-Consistent 3D Airflow (`sim/airflow.py`):**
  - Staggered MAC 3D grid (`48 × 48 × 12`, `2.5 m` cubic voxels over `120 × 120 × 30 m`).
  - Exact zero-normal-flow solid masks for rectangular buildings and cylindrical tanks, plus porous drag for open-truss pipe racks and standalone pipes.
  - Industrial ABL power-law shear (`α = 0.18`, `z_ref = 10 m`), Röckle/QUIC-URB lee-side recirculation cavities, and vectorized Red-Black SOR 3D Poisson pressure projection (`ω = 1.72`, `75` iterations) enforcing `div(u) ≈ 0` (cached per wind angle).
  - Superimposed with 6 strictly divergence-free 3D Fourier eddy modes, cylindrical Strouhal vortex shedding, and Ornstein–Uhlenbeck temporal gusts (`τ_w = 6 s`, `σ_w = 0.6 m/s`).
- **Lagrangian 3D Gaussian Puff Dispersion (`sim/plume.py`):**
  - Vectorized ring buffer (`1,200` physics puffs, capped at `≤ 400` render puffs for 60 FPS WebGL).
  - SI mass-to-ppmv ideal-gas calibration (`M = 0.01604 kg/mol` for CH₄ at `293.15 K`, `101,325 Pa`), Briggs (1973) urban/industrial Pasquill–Gifford dispersion classes (`A`–`F`), RK2 midpoint streamline advection around obstacles, buoyant plume rise, specular ground & mixing-lid (`300 m`) reflections, and vectorized solid-wall ray occlusion so gas cannot teleport through walls.
- **6-DOF Rigid-Body Quadrotor (`sim/drone.py`):**
  - `1.5 kg` industrial inspection quadrotor (`450 mm` wheelbase, 10-inch props) integrated at **160 Hz** (`8×` internal sub-stepping per 20 Hz tick).
  - 4-motor 1st-order ESC lag (`τ_motor = 35 ms`), Lee et al. geometric `SE(3)` attitude controller on `SO(3)` (unit quaternions), rotor gyroscopic precession, anisotropic body drag + blade-flapping H-force, Cheeseman–Bennett ground effect, rotor-downwash sample-point displacement & dilution, soft-penalty obstacle contact physics, and a `27 kJ` physical power/battery model (~3.5 min active endurance).
- **Chemical, Thermal & Anemometer Sensors (`sim/sensor.py`):**
  - Asymmetric 1st-order chemical lag (`τ_rise = 0.6 s`, `τ_decay = 1.8 s`), `1.9 ppm` CH₄ background, `±8%` multiplicative noise, slow zero-drift, `5,000 ppm` saturation, `0.01 ppm` ADC quantization, tilt-coupled ultrasonic anemometer noise, and toggleable 30% `NaN` dropout fault injection.
- **World Geometry & Procedural Layouts (`sim/world.py`, `sim/layout.py`):**
  - 5 named industrial facilities (`refinery`, `pipe_loop`, `tank_terminal`, `compressor_yard`, `cooling_works`) plus deterministic procedural facility generation with orthogonal A*-routed connected pipe networks and guaranteed obstacle/launch-pad clearances.

---

### 5. System Contracts, Coordinate Convention & Mission Judge

- **Single Source of Truth (`shared/schemas.py`):** Frozen Pydantic v2 WebSocket payloads — `WorldInitPayload`, `WindFieldPayload`, `SimTickPayload`, `BeliefUpdatePayload`, `BackendStatusPayload`, and `JudgeCommand` (`start_mission`, `set_wind`, `place_leak`, `clear_leaks`, `toggle_dropout`, `set_comparison_drones`, `pause`, `step`, `reset_race`, `select_map`, `randomize_layout`).
- **Coordinate System:** Python physics & agents operate in right-handed **Z-up** meters (`x, y ∈ [-60, 60]`, `z ∈ [0, 30]`). Conversion to Three.js **Y-up** happens strictly at the rendering boundary (`frontend/src/coords.js`):
  ```text
  three_pos = (sim_x, sim_z, -sim_y)
  ```
- **Independent Ground-Truth Confirmation (`sim/engine.py`):** Agents never receive leak coordinates or judge feedback. `SimEngine._check_completion()` confirms a find only when the drone and its fresh source estimate (`≤ 5 s` old) both remain within `5.0 m` of a true leak while measuring `> 15 ppm` for `2.0 s` continuously.

---

## Autonomous Agents (`agents/`)

| Agent | Role | Module | Description |
| :--- | :--- | :--- | :--- |
| **SeeCH4 (Triple-TabPFN)** | **Flagship (always active)** | [agents/tabpfn_agent.py](agents/tabpfn_agent.py), [agents/reliable_tabpfn.py](agents/reliable_tabpfn.py) | Three specialized TabPFN regressors (post-contact 3D source inversion, pre-contact spatial prior, online 3D plume/hazard mapper) + obstacle-aware 3D A* navigation. |
| **Wind-Plume (Baseline)** | Live UI toggle / benchmark | [agents/wind_plume.py](agents/wind_plume.py) | Onboard wind-relative downwind transect search with 3D A* routing and upwind/crosswind/vertical probing. |
| **Gradient (Baseline)** | Live UI toggle / benchmark | [agents/gradient.py](agents/gradient.py) | Bio-inspired anemotactic upwind surge and crosswind cast baseline. |
| **Lawnmower (Baseline)** | Live UI toggle / benchmark | [agents/lawnmower.py](agents/lawnmower.py) | Deterministic 3D volumetric serpentine raster sweep baseline. |
| **Mock Agent** | Unit tests only | [agents/mock_agent.py](agents/mock_agent.py) | Sub-millisecond mock belief generator for headless unit tests without ML weights. |

Force a specific TabPFN backend at any time with `TABPFN_BACKEND=local|http|client`.

---

## Physics Audit, Headless Benchmarking & CLI (`tools/`, `scripts/`)

See [tools/README.md](tools/README.md) for full CLI documentation and JSON output schemas.

```bash
# Quantitative physics realism scorecard (exit code 0 on pass)
python -m tools.physics_audit
python -m tools.sim_ctl audit --json

# Headless multi-scenario benchmark (open_corridor, wake_trap, wind_shift_90, ...)
python -m tools.eval_runner --scenarios open_corridor,wake_trap,wind_shift_90 --duration 60

# Full Triple-TabPFN validation / held-out 44-case leak detection benchmark
python -m scripts.benchmark_leak_detection --tier val
python -m scripts.benchmark_leak_detection --tier test

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
├── sim/                        # Headless 3D physics testbed (zero web/UI dependencies)
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
│   ├── tabpfn_features.py      # Physics & spatial feature builders + deconvolution
│   ├── tabpfn_nav.py           # Fast 3D A* navigation & safe goal projection
│   ├── tabpfn_remote.py        # Unified TabPFN backend (local, HTTP, or Prior Labs API)
│   ├── wind_plume.py           # Wind-Plume baseline agent
│   ├── search_control.py       # Observation-only 3D plume acquisition controller
│   ├── navigation.py           # Swept-segment 3D A* planner & RoutedAgent wrapper
│   ├── gradient.py             # Bio-inspired surge-cast baseline
│   ├── lawnmower.py            # 3D volumetric raster sweep baseline
│   └── mock_agent.py           # Fast mock agent for headless unit tests
├── server/
│   └── app.py                  # FastAPI WebSocket broadcaster (/ws, /health) & static UI
├── frontend/                   # Interactive Three.js + Vite 3D viewer & model dashboard
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
