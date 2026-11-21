const DRONE_HEX = {
  tabpfn: "#22d3ee",
  gradient: "#fb923c",
  lawnmower: "#94a3b8",
  manual: "#a3e635",
};

/**
 * Manages DOM HUD: live scoreboard, wind dial, judge controls, and toggles.
 */
export class HudController {
  constructor({
    onLayer,
    onRoute,
    onSendCommand,
    onTogglePlume,
    onToggleBelief,
    onTogglePlaceLeakMode,
    onResetTrails,
  }) {
    this.onSendCommand = onSendCommand;
    this.onLayer = onLayer;
    this.onRoute = onRoute;
    this.onTogglePlume = onTogglePlume;
    this.onToggleBelief = onToggleBelief;
    this.onTogglePlaceLeakMode = onTogglePlaceLeakMode;
    this.onResetTrails = onResetTrails;

    this.plumeVisible = false;
    this.beliefVisible = true;
    this.placeLeakMode = false;
    this.paused = false;

    this.elConnDot = document.getElementById("conn-dot");
    this.elConnLabel = document.getElementById("conn-label");
    this.elSimTime = document.getElementById("sim-time");
    this.elWindReadout = document.getElementById("wind-readout");
    this.elDropoutBadge = document.getElementById("dropout-badge");
    this.elScoreBody = document.getElementById("scoreboard-body");
    this.elWindArrow = document.getElementById("wind-arrow-group");
    this.elHint = document.getElementById("hint-text");

    this.elWindAngle = document.getElementById("wind-angle");
    this.elWindAngleVal = document.getElementById("wind-angle-val");
    this.elWindSpeed = document.getElementById("wind-speed");
    this.elWindSpeedVal = document.getElementById("wind-speed-val");

    this.btnPlume = document.getElementById("btn-reveal-plume");
    this.btnBelief = document.getElementById("btn-toggle-belief");
    this.btnDropout = document.getElementById("btn-dropout");
    this.btnPlaceLeak = document.getElementById("btn-place-leak");
    this.btnClearLeaks = document.getElementById("btn-clear-leaks");
    this.btnPause = document.getElementById("btn-pause");
    this.btnReset = document.getElementById("btn-reset");

    this._bindEvents();
  }

  _bindEvents() {
    document.getElementById("prediction-layer").addEventListener("change", e => this.onLayer(e.target.value));
    document.getElementById("show-route").addEventListener("change", e => this.onRoute(e.target.checked));
    document.getElementById("btn-step").addEventListener("click", () => this.onSendCommand({action:"step"}));
    document.getElementById("btn-seed-reset").addEventListener("click", () => {
      const seed=Number(document.getElementById("episode-seed").value);
      if (Number.isInteger(seed) && seed>=0) this.onSendCommand({action:"reset_race",seed});
    });
    const emitWind = () => {
      const angle = Number(this.elWindAngle.value);
      const speed = Number(this.elWindSpeed.value);
      this.elWindAngleVal.textContent = String(angle);
      this.elWindSpeedVal.textContent = speed.toFixed(1);
      this.onSendCommand({
        action: "set_wind",
        wind_angle_deg: angle,
        wind_speed: speed,
      });
    };

    this.elWindAngle.addEventListener("input", emitWind);
    this.elWindSpeed.addEventListener("input", emitWind);

    this.btnPlume.addEventListener("click", () => {
      this.plumeVisible = !this.plumeVisible;
      this.btnPlume.classList.toggle("btn-active", this.plumeVisible);
      this.btnPlume.textContent = this.plumeVisible
        ? "Hide True Plume"
        : "Reveal True Plume";
      this.onTogglePlume(this.plumeVisible);
    });

    this.btnBelief.addEventListener("click", () => {
      this.beliefVisible = !this.beliefVisible;
      this.btnBelief.classList.toggle("btn-active", this.beliefVisible);
      this.onToggleBelief(this.beliefVisible);
    });

    this.btnDropout.addEventListener("click", () => {
      this.onSendCommand({ action: "toggle_dropout" });
    });

    this.btnPlaceLeak.addEventListener("click", () => {
      this.setPlaceLeakMode(!this.placeLeakMode);
    });

    this.btnClearLeaks.addEventListener("click", () => {
      this.onSendCommand({ action: "clear_leaks" });
    });

    this.btnPause.addEventListener("click", () => {
      this.paused = !this.paused;
      this.btnPause.textContent = this.paused ? "Resume" : "Pause";
      this.btnPause.classList.toggle("btn-active", this.paused);
      this.onSendCommand({ action: "pause" });
    });

    this.btnReset.addEventListener("click", () => {
      this.paused = false;
      this.btnPause.textContent = "Pause";
      this.btnPause.classList.remove("btn-active");
      this.onResetTrails();
      this.onSendCommand({ action: "reset_race" });
    });
  }

  setPlaceLeakMode(active) {
    this.placeLeakMode = Boolean(active);
    this.btnPlaceLeak.classList.toggle("btn-danger-active", this.placeLeakMode);
    this.elHint.textContent = this.placeLeakMode
      ? "Click anywhere on the refinery ground to place a hidden gas leak..."
      : "Drag to orbit · Scroll to zoom · Right-drag to pan";
    this.onTogglePlaceLeakMode(this.placeLeakMode);
  }

  setConnectionStatus(state, label) {
    this.elConnDot.className = `dot ${state}`;
    this.elConnLabel.textContent = label;
  }

  updateBackendStatus(status) {
    document.getElementById("model-status").textContent = `${status.mode} · ${status.state}${status.error ? ": " + status.error : ""}`;
    document.getElementById("model-metrics").textContent = `${(status.inference_seconds || 0).toFixed(2)} s inference · ${status.measurement_count || 0} measurements · ${(status.exposure_ppm_seconds || 0).toFixed(1)} ppm·s exposure`;
  }

  updateBelief(belief) {
    document.getElementById("source-estimate").textContent = `Estimated source (x,y,z): ${belief.estimated_source.map(v => v.toFixed(1)).join(", ")} m`;
  }

  updateTick(tick) {
    if (tick.paused !== undefined) {
      this.paused=tick.paused;
      this.btnPause.textContent=this.paused ? "Resume" : "Pause";
      this.btnPause.classList.toggle("btn-active",this.paused);
    }
    document.getElementById("btn-step").disabled=!this.paused;
    this.elSimTime.textContent = `${tick.sim_time.toFixed(1)}s`;

    const [wx, wy] = tick.wind_vec;
    const speed = Math.hypot(wx, wy);
    const angleDeg = ((Math.atan2(wy, wx) * 180) / Math.PI + 360) % 360;
    this.elWindReadout.textContent = `${speed.toFixed(1)} m/s @ ${Math.round(angleDeg)}°`;

    // Compass SVG points UP for +Y (90 deg sim), RIGHT for +X (0 deg sim)
    const svgDeg = 90 - angleDeg;
    this.elWindArrow.setAttribute("transform", `rotate(${svgDeg.toFixed(1)} 40 40)`);

    this.elDropoutBadge.classList.toggle("hidden", !tick.sensor_dropout_active);
    this.btnDropout.classList.toggle("btn-danger-active", tick.sensor_dropout_active);

    // Update scoreboard rows
    this.elScoreBody.innerHTML = tick.drones
      .map((d) => {
        const col = DRONE_HEX[d.id] ?? "#e2e8f0";
        const distStr = d.dist_to_nearest_leak > 500 ? "—" : `${d.dist_to_nearest_leak.toFixed(1)}m`;
        return `
          <tr>
            <td><span class="drone-name"><span class="swatch" style="background:${col}"></span>${d.id}</span></td>
            <td><span class="status-pill status-${d.status}">${d.status}</span></td>
            <td>${d.ppm.toFixed(1)}</td>
            <td>${distStr}</td>
            <td>${d.samples_taken}</td>
            <td>${d.battery_pct.toFixed(0)}%</td>
          </tr>
        `;
      })
      .join("");
  }
}
