const DRONE_HEX = {
  tabpfn: "#22d3ee",
  gradient: "#fb923c",
  lawnmower: "#e2e8f0",
  manual: "#a3e635",
};

/**
 * Manages DOM HUD: live scoreboard, wind dial, judge controls, toast notifications,
 * and interactive Object Inspector (drones, pipes, valves, buildings).
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
    onSelectDrone,
    onFollowDrone,
    onFocusPosition,
    onPlaceLeakAt,
  }) {
    this.onSendCommand = onSendCommand;
    this.onLayer=onLayer;this.onRoute=onRoute;
    this.onTogglePlume = onTogglePlume;
    this.onToggleBelief = onToggleBelief;
    this.onTogglePlaceLeakMode = onTogglePlaceLeakMode;
    this.onResetTrails = onResetTrails;
    this.onSelectDrone = onSelectDrone;
    this.onFollowDrone = onFollowDrone;
    this.onFocusPosition = onFocusPosition;
    this.onPlaceLeakAt = onPlaceLeakAt;

    this.plumeVisible = false;
    this.beliefVisible = true;
    this.placeLeakMode = false;
    this.paused = false;

    this._selectedMeta = null;
    this._followingDroneId = null;
    this._latestDrones = new Map();
    this._toastTimer = null;

    this.elConnDot = document.getElementById("conn-dot");
    this.elConnLabel = document.getElementById("conn-label");
    this.elSimTime = document.getElementById("sim-time");
    this.elWindReadout = document.getElementById("wind-readout");
    this.elLeakCount = document.getElementById("leak-count");
    this.elDropoutBadge = document.getElementById("dropout-badge");
    this.elScoreBody = document.getElementById("scoreboard-body");
    this.elWindArrow = document.getElementById("wind-arrow-group");
    this.elHint = document.getElementById("hint-text");
    this.elToast = document.getElementById("toast");

    this.elInspector = document.getElementById("inspector");
    this.elInspectorType = document.getElementById("inspector-type");
    this.elInspectorTitle = document.getElementById("inspector-title");
    this.elInspectorBody = document.getElementById("inspector-body");
    this.elInspectorActions = document.getElementById("inspector-actions");
    this.btnInspectorClose = document.getElementById("inspector-close");

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
    document.getElementById("btn-new-mission").addEventListener("click",()=>{this.onResetTrails();this.onSendCommand({action:"reset_race"});});
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
      this.setPlumeVisible(!this.plumeVisible);
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
      this.showToast("Cleared all gas leaks");
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
      this.showToast("Race reset");
    });

    this.btnInspectorClose.addEventListener("click", () => {
      this.clearInspector();
    });

    // Click a drone row in the scoreboard to inspect/focus it
    this.elScoreBody.addEventListener("click", (evt) => {
      const tr = evt.target.closest("tr[data-drone-id]");
      if (tr) {
        this.onSelectDrone?.(tr.dataset.droneId);
      }
    });
  }

  setPlumeVisible(visible) {
    this.plumeVisible = Boolean(visible);
    this.btnPlume.classList.toggle("btn-active", this.plumeVisible);
    this.btnPlume.textContent = this.plumeVisible
      ? "Hide True Plume"
      : "Reveal True Plume";
    this.onTogglePlume(this.plumeVisible);
  }

  showToast(message, durationMs = 2600) {
    this.elToast.textContent = message;
    this.elToast.classList.remove("hidden");
    if (this._toastTimer) clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      this.elToast.classList.add("hidden");
    }, durationMs);
  }

  setPlaceLeakMode(active) {
    this.placeLeakMode = Boolean(active);
    this.btnPlaceLeak.classList.toggle("btn-danger-active", this.placeLeakMode);
    this.btnPlaceLeak.textContent = this.placeLeakMode
      ? "Click Pipe / Ground... (Cancel)"
      : "Click Ground: Place Leak";
    this.elHint.textContent = this.placeLeakMode
      ? "Click any pipe, valve, or ground point to place a leak (Esc to cancel)"
      : "Click drones, pipes, valves, or buildings to inspect · Drag to orbit";
    this.onTogglePlaceLeakMode(this.placeLeakMode);
  }

  setConnectionStatus(state, label) {
    this.elConnDot.className = `dot ${state}`;
    this.elConnLabel.textContent = label;
  }

  clearInspector() {
    this._selectedMeta = null;
    if (this._followingDroneId) {
      this._followingDroneId = null;
      this.onFollowDrone?.(null);
    }
    this.elInspector.classList.add("hidden");
  }

  showInspector(meta, hitSimPos = null) {
    this._selectedMeta = { ...meta, hitSimPos };
    this.elInspector.classList.remove("hidden");
    this._renderInspector();
  }

  _renderInspector() {
    const m = this._selectedMeta;
    if (!m) return;

    const rows = [];
    this.elInspectorActions.innerHTML = "";

    if (m.kind === "drone") {
      const d = this._latestDrones.get(m.id);
      this.elInspectorType.textContent = "DRONE";
      this.elInspectorTitle.textContent = m.id.toUpperCase();
      if (d) {
        rows.push(["Status", d.status.toUpperCase()]);
        rows.push(["Gas Reading", `${d.ppm.toFixed(1)} ppm`]);
        rows.push(["Battery", `${d.battery_pct.toFixed(1)}%`]);
        rows.push(["Samples", String(d.samples_taken)]);
        rows.push([
          "Dist to Leak",
          d.dist_to_nearest_leak > 500 ? "—" : `${d.dist_to_nearest_leak.toFixed(1)} m`,
        ]);
        rows.push([
          "Position",
          `(${d.pos[0].toFixed(1)}, ${d.pos[1].toFixed(1)}, ${d.pos[2].toFixed(1)})`,
        ]);
      }

      const isFollowing = this._followingDroneId === m.id;
      const btnFollow = document.createElement("button");
      btnFollow.className = `btn ${isFollowing ? "btn-active" : "btn-accent"}`;
      btnFollow.textContent = isFollowing ? "Stop Following" : "Follow Drone";
      btnFollow.onclick = () => {
        this._followingDroneId = isFollowing ? null : m.id;
        this.onFollowDrone?.(this._followingDroneId);
        this._renderInspector();
      };
      this.elInspectorActions.appendChild(btnFollow);
    } else if (m.kind === "pipe") {
      this.elInspectorType.textContent = "PIPELINE";
      this.elInspectorTitle.textContent = m.name || m.id;
      rows.push(["Line ID", m.id]);
      rows.push(["Service", m.service || "Process"]);
      rows.push(["Pressure", `${m.pressureBar ?? 12.0} bar`]);
      if (m.hitSimPos) {
        rows.push([
          "Click Point",
          `(${m.hitSimPos[0].toFixed(1)}, ${m.hitSimPos[1].toFixed(1)}, ${m.hitSimPos[2].toFixed(1)})`,
        ]);
      }

      if (m.hitSimPos) {
        const btnLeak = document.createElement("button");
        btnLeak.className = "btn btn-danger-active";
        btnLeak.textContent = "Trigger Leak Here";
        btnLeak.onclick = () => this.onPlaceLeakAt?.(m.hitSimPos);
        this.elInspectorActions.appendChild(btnLeak);

        const btnFocus = document.createElement("button");
        btnFocus.className = "btn";
        btnFocus.textContent = "Focus";
        btnFocus.onclick = () => this.onFocusPosition?.(m.hitSimPos);
        this.elInspectorActions.appendChild(btnFocus);
      }
    } else if (m.kind === "asset") {
      this.elInspectorType.textContent = m.assetType.toUpperCase();
      this.elInspectorTitle.textContent = m.id;
      const risk = m.ageYears >= 18 ? "HIGH" : m.ageYears >= 10 ? "MEDIUM" : "LOW";
      rows.push(["Asset Type", m.assetType]);
      rows.push(["Age", `${m.ageYears.toFixed(1)} yrs`]);
      rows.push(["Leak Risk", risk]);
      rows.push([
        "Location",
        `(${m.posSim[0].toFixed(1)}, ${m.posSim[1].toFixed(1)}, ${m.posSim[2].toFixed(1)})`,
      ]);

      const btnLeak = document.createElement("button");
      btnLeak.className = "btn btn-danger-active";
      btnLeak.textContent = "Trigger Leak at Valve";
      btnLeak.onclick = () => this.onPlaceLeakAt?.(m.posSim);
      this.elInspectorActions.appendChild(btnLeak);

      const btnFocus = document.createElement("button");
      btnFocus.className = "btn";
      btnFocus.textContent = "Focus";
      btnFocus.onclick = () => this.onFocusPosition?.(m.posSim);
      this.elInspectorActions.appendChild(btnFocus);
    } else if (m.kind === "building") {
      this.elInspectorType.textContent = m.structureKind.toUpperCase();
      this.elInspectorTitle.textContent = m.id;
      rows.push(["Dimensions", `${m.size[0].toFixed(0)} × ${m.size[1].toFixed(0)} × ${m.size[2].toFixed(0)} m`]);
      rows.push(["Wake Cavity", `${(m.size[2] * 2.5).toFixed(0)} m downwind`]);

      const btnFocus = document.createElement("button");
      btnFocus.className = "btn";
      btnFocus.textContent = "Focus Camera";
      btnFocus.onclick = () => this.onFocusPosition?.(m.centerSim);
      this.elInspectorActions.appendChild(btnFocus);
    }

    this.elInspectorBody.innerHTML = rows
      .map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`)
      .join("");
  }

  updateBackendStatus(status) {
    this.modelMode=status.mode;
    this.updateReport(status.mission_report);
    const mission=document.getElementById("mission-status");
    const states={found:"Leck gefunden — Suche beendet",no_leak:"No active leak — restart mission or place a leak",depleted:"Battery empty — restart mission",paused:"Paused",searching:"Searching for leak"};
    mission.textContent=states[status.mission_state] || (status.mode==="mock" ? "Mock mission" : "Searching");
    mission.style.color=["no_leak","depleted"].includes(status.mission_state) ? "#fca5a5" : "#cbd5e1";
    document.getElementById("model-status").textContent = `${status.mode} · ${status.state}${status.error ? ": " + status.error : ""}`;
    document.getElementById("model-metrics").textContent = `${(status.inference_seconds || 0).toFixed(2)} s inference · ${status.measurement_count || 0} measurements · ${(status.exposure_ppm_seconds || 0).toFixed(1)} ppm·s exposure`;
  }

  updateBelief(belief) {
    document.getElementById("source-estimate").textContent = `Estimated source (x,y,z): ${belief.estimated_source.map(v => v.toFixed(1)).join(", ")} m`;
  }

  updateReport(report) {
    this.completed=Boolean(report);
    document.getElementById("mission-report").classList.toggle("hidden",!report);
    for (const id of ["btn-pause","btn-step","btn-clear-leaks","btn-place-leak","btn-dropout","wind-angle","wind-speed"]) document.getElementById(id).disabled=this.completed;
    if (!report)return;
    document.getElementById("report-duration").textContent=`Suchdauer: ${report.elapsed_sim_seconds.toFixed(1)} s Simulationszeit · ${report.measurement_count} Messungen`;
    document.getElementById("report-position").textContent=`Geschätzte Quelle (x,y,z): ${report.estimated_source.map(x=>x.toFixed(1)).join(", ")} m`;
    document.getElementById("report-score").textContent=report.model_score==null ? "Modellbewertung: nicht verfügbar (z. B. Mock-Modus)" : `Modellbewertung: ${(report.model_score*100).toFixed(1)} % Klassifikationsscore (unkalibriert)`;
    document.getElementById("report-evidence").textContent=`${this.modelMode==="mock" ? "Mock-Fund" : "Im Simulator bestätigt"} · Ortungsfehler ${report.estimated_error_m.toFixed(1)} m · ${report.ppm_at_detection.toFixed(1)} ppm vor Ort`;
    document.getElementById("report-wall").textContent=`Echtzeit seit Reset: ${report.elapsed_wall_seconds.toFixed(1)} s (inkl. Pausen) · Schätzung ${report.estimate_age_seconds.toFixed(1)} s alt`;
  }

  updateTick(tick) {
    this.updateReport(tick.mission_report);
    if (tick.paused !== undefined) {
      this.paused=tick.paused;
      this.btnPause.textContent=this.paused ? "Resume" : "Pause";
      this.btnPause.classList.toggle("btn-active",this.paused);
    }
    document.getElementById("btn-step").disabled=this.completed || !this.paused;

    this.elSimTime.textContent = `${tick.sim_time.toFixed(1)}s`;
    this.elLeakCount.textContent = String(tick.leaks.length);

    const [wx, wy] = tick.wind_vec;
    const speed = Math.hypot(wx, wy);
    const angleDeg = ((Math.atan2(wy, wx) * 180) / Math.PI + 360) % 360;
    this.elWindReadout.textContent = `${speed.toFixed(1)} m/s @ ${Math.round(angleDeg)}°`;

    const svgDeg = 90 - angleDeg;
    this.elWindArrow.setAttribute("transform", `rotate(${svgDeg.toFixed(1)} 40 40)`);

    this.elDropoutBadge.classList.toggle("hidden", !tick.sensor_dropout_active);
    this.btnDropout.classList.toggle("btn-danger-active", tick.sensor_dropout_active);

    for (const d of tick.drones) {
      this._latestDrones.set(d.id, d);
    }
    if (this._selectedMeta?.kind === "drone") {
      this._renderInspector();
    }

    this.elScoreBody.innerHTML = tick.drones
      .map((d) => {
        const col = DRONE_HEX[d.id] ?? "#e2e8f0";
        const distStr = d.dist_to_nearest_leak > 500 ? "—" : `${d.dist_to_nearest_leak.toFixed(1)}m`;
        return `
          <tr data-drone-id="${d.id}" style="cursor:pointer" title="Click to inspect ${d.id}">
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
