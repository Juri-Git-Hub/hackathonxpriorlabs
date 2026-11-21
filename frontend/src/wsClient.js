/**
 * WebSocket client connecting to FastAPI backend (`/ws`), with `?mock=true`
 * standalone synthetic stream generator so UI never blocks on Python.
 */
export class SimWsClient {
  constructor({ onWorldInit, onWindField, onSimTick, onBeliefUpdate, onStatus }) {
    this.onWorldInit = onWorldInit;
    this.onWindField = onWindField;
    this.onSimTick = onSimTick;
    this.onBeliefUpdate = onBeliefUpdate;
    this.onStatus = onStatus;
    this.ws = null;
    this._reconnectTimer = null;
    this._mockTimer = null;

    const params = new URLSearchParams(window.location.search);
    this.mockMode = params.get("mock") === "true";
    this.wsUrl =
      params.get("ws") ||
      `ws://${window.location.hostname || "localhost"}:8000/ws`;
  }

  start() {
    if (this.mockMode) {
      this._startMockStream();
    } else {
      this._connectWs();
    }
  }

  _connectWs() {
    this.onStatus?.("connecting", "Connecting...");
    try {
      this.ws = new WebSocket(this.wsUrl);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.onStatus?.("connected", "Live WS");
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "world_init") {
          this.onWorldInit?.(msg);
        } else if (msg.type === "wind_field") {
          this.onWindField?.(msg);
        } else if (msg.type === "sim_tick") {
          this.onSimTick?.(msg);
        } else if (msg.type === "belief_update") {
          this.onBeliefUpdate?.(msg);
        }
      } catch (err) {
        console.warn("Malformed WS payload", err);
      }
    };

    this.ws.onclose = () => {
      this.onStatus?.("disconnected", "Reconnecting...");
      this._scheduleReconnect();
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  _scheduleReconnect() {
    if (this._reconnectTimer) return;
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._connectWs();
    }, 1500);
  }

  sendCommand(cmd) {
    const payload = { type: "judge_cmd", ...cmd };
    if (this.mockMode) {
      this._handleMockCommand(payload);
      return;
    }
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  // --- Standalone ?mock=true synthetic stream ---
  _startMockStream() {
    this.onStatus?.("mock", "Mock Stream (?mock=true)");
    const gridCoords = [];
    for (let z = 3; z <= 18; z += 5) {
      for (let y = -48; y <= 48; y += 12) {
        for (let x = -48; x <= 48; x += 12) {
          gridCoords.push([x, y, z]);
        }
      }
    }

    this._mockState = {
      simTime: 0,
      windAngle: 15,
      windSpeed: 3.5,
      dropout: false,
      paused: false,
      leaks: [{ id: "leak_01", pos: [-6, -7.2, 4], emission_rate: 1, buoyancy_vz: 0.25 }],
    };

    const worldInit = {
      type: "world_init",
      bounds_min: [-60, -60, 0],
      bounds_max: [60, 60, 30],
      buildings: [
        { id: "bldg_control", kind: "building", min_xyz: [-38, -28, 0], max_xyz: [-20, -12, 12] },
        { id: "tank_north", kind: "storage_tank", min_xyz: [-10, 14, 0], max_xyz: [10, 34, 16] },
        { id: "tank_east", kind: "storage_tank", min_xyz: [22, -16, 0], max_xyz: [40, 2, 14] },
        { id: "pipe_rack_center", kind: "pipe_rack", min_xyz: [-16, -6, 0], max_xyz: [14, 4, 8] },
        { id: "bldg_compressor", kind: "building", min_xyz: [-35, 16, 0], max_xyz: [-18, 32, 10] },
      ],
      assets: [
        { id: "flange_01", asset_type: "flange", pos: [-6, -7.2, 4], age_years: 22 },
        { id: "valve_01", asset_type: "valve", pos: [-17.5, -2, 3.5], age_years: 18.5 },
        { id: "tank_port_e1", asset_type: "storage_tank", pos: [20.5, -7, 6], age_years: 24 },
      ],
      grid_coords: gridCoords,
    };

    this.onWorldInit?.(worldInit);

    let tickCount = 0;
    this._mockTimer = setInterval(() => {
      const s = this._mockState;
      if (!s.paused) {
        s.simTime += 0.05;
        tickCount += 1;
      }
      const rad = (s.windAngle * Math.PI) / 180;
      const wx = s.windSpeed * Math.cos(rad);
      const wy = s.windSpeed * Math.sin(rad);
      const t = s.simTime;

      const puffs = [];
      for (const lk of s.leaks) {
        for (let i = 0; i < 40; i++) {
          const age = ((i * 0.5 + t) % 20);
          puffs.push({
            pos: [
              lk.pos[0] + wx * age * 0.7 + Math.sin(age + i) * 1.2,
              lk.pos[1] + wy * age * 0.7 + Math.cos(age + i) * 1.2,
              lk.pos[2] + 0.2 * age,
            ],
            radius_xy: 0.8 + 0.35 * Math.pow(age, 0.85),
            radius_z: 0.5 + 0.2 * Math.pow(age, 0.75),
            opacity: Math.max(0.08, 0.75 / (1 + 0.09 * age)),
          });
        }
      }

      this.onSimTick?.({
        type: "sim_tick",
        sim_time: t,
        wind_vec: [wx, wy],
        leaks: s.leaks,
        sensor_dropout_active: s.dropout,
        puffs,
        drones: [
          {
            id: "tabpfn",
            pos: [Math.cos(t * 0.35) * 20 - 6, Math.sin(t * 0.35) * 18 - 7, 6 + Math.sin(t) * 1.5],
            rot_rpy: [0.08, 0.12, t * 0.35 + Math.PI / 2],
            target_pos: [-6, -7, 5],
            ppm: 18 + Math.sin(t) * 8,
            battery_pct: Math.max(5, 100 - t * 0.4),
            samples_taken: Math.floor(t * 2),
            dist_to_nearest_leak: 12 + Math.sin(t * 0.5) * 4,
            status: "tracking",
          },
          {
            id: "gradient",
            pos: [Math.cos(t * 0.25 + 1) * 28, Math.sin(t * 0.25 + 1) * 25, 5],
            rot_rpy: [-0.05, 0.1, t * 0.25],
            target_pos: [10, 10, 5],
            ppm: 5 + Math.max(0, Math.cos(t) * 6),
            battery_pct: Math.max(5, 100 - t * 0.42),
            samples_taken: Math.floor(t * 2),
            dist_to_nearest_leak: 24,
            status: "searching",
          },
          {
            id: "lawnmower",
            pos: [-45 + ((t * 6) % 90), -35 + Math.floor((t * 6) / 90) * 14, 4],
            rot_rpy: [0, 0.08, 0],
            target_pos: [45, -35, 4],
            ppm: 0.7,
            battery_pct: Math.max(5, 100 - t * 0.38),
            samples_taken: Math.floor(t * 2),
            dist_to_nearest_leak: 38,
            status: "searching",
          },
        ],
      });

      if (tickCount % 20 === 0) {
        const q50 = gridCoords.map(([x, y, z]) => {
          const d = Math.hypot(x + 6, y + 7.2, z - 4);
          return Math.max(0.05, 3.8 * Math.exp(-d / 20));
        });
        const uncertainty = gridCoords.map((_, i) => 0.4 + 0.6 * Math.abs(Math.sin(t + i * 0.1)));
        this.onBeliefUpdate?.({
          type: "belief_update",
          sim_time: t,
          drone_id: "tabpfn",
          q50,
          uncertainty,
          estimated_source: [-6, -7.2, 4],
          next_waypoint: [-4, -5, 5],
        });
      }
    }, 50);
  }

  _handleMockCommand(cmd) {
    const s = this._mockState;
    if (!s) return;
    if (cmd.action === "set_wind") {
      if (cmd.wind_angle_deg != null) s.windAngle = cmd.wind_angle_deg;
      if (cmd.wind_speed != null) s.windSpeed = cmd.wind_speed;
    } else if (cmd.action === "place_leak" && cmd.leak_pos) {
      s.leaks.push({
        id: `leak_${s.leaks.length + 1}`,
        pos: cmd.leak_pos,
        emission_rate: 1.0,
        buoyancy_vz: 0.25,
      });
    } else if (cmd.action === "clear_leaks") {
      s.leaks = [];
    } else if (cmd.action === "toggle_dropout") {
      s.dropout = cmd.dropout_enabled ?? !s.dropout;
    } else if (cmd.action === "pause") {
      s.paused = !s.paused;
    } else if (cmd.action === "reset_race") {
      s.simTime = 0;
    }
  }
}
