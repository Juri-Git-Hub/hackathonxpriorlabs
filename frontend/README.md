# SeeCH4 Frontend (Three.js + Vite)

Interactive 3D visualization for the **SeeCH4** autonomous industrial gas-leak detection simulator.

## Running

- **Integrated demo (builds and serves via FastAPI):**
  ```bash
  python launch_demo.py
  # Open http://localhost:8501
  ```
- **Development with Vite HMR:**
  ```bash
  # Terminal 1 (repo root)
  python -m server.app --port 8501

  # Terminal 2 (frontend/)
  npm ci
  npm run dev
  # Open http://localhost:5173
  ```
  Vite proxies `/ws` and `/health` to the backend on port `8501`. Append `?mock=true` to run with synthetic payloads without a Python backend, or `?ws=ws://host:port/ws` to point at a custom server.

## Coordinate Convention

- Python physics uses right-handed **Z-up** (`x ∈ [-60, 60]`, `y ∈ [-60, 60]`, `z ∈ [0, 30]`).
- Three.js uses right-handed **Y-up**. Conversion is performed strictly at the rendering boundary in `src/coords.js`:
  ```text
  threePos = (sim_x, sim_z, -sim_y)
  ```

## WebSocket Contract

See [`shared/schemas.py`](../shared/schemas.py) for all message schemas (`WorldInitPayload`, `WindFieldPayload`, `SimTickPayload`, `BeliefUpdatePayload`, `BackendStatusPayload`, `JudgeCommand`).

## Tests & Build

```bash
node --test test/*.test.js
npm run build
```
