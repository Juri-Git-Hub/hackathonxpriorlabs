const DRONE_HEX = {
  tabpfn: "#22d3ee",
  gradient: "#fb923c",
  lawnmower: "#e2e8f0",
  manual: "#a3e635",
};

/**
 * Manages DOM HUD: live benchmark scoreboard, TabPFN advantage strip,
 * synced base wind setpoints vs live gusts, Presentation Mode, and Object Inspector.
 */
export class HudController {
  constructor({
    onLayer,
    onRoute,
    onSendCommand,
    onTogglePlume,
    onToggleBelief,
    onToggleWind,
    onTogglePlaceLeakMode,
    onResetTrails,
    onSelectDrone,
    onClearSelection,
    onFollowDrone,
    onFocusPosition,
    onPlaceLeakAt,
    onToggleRecording,
    onCaptureScreenshot,
    onSelectCameraPreset,
    onToggleAutoOrbit,
    onCycleFollowDrone,
  }) {
    this.onSendCommand = onSendCommand;
    this.onLayer=onLayer;this.onRoute=onRoute;
    this.onTogglePlume = onTogglePlume;
    this.onToggleBelief = onToggleBelief;
    this.onToggleWind = onToggleWind;
    this.onTogglePlaceLeakMode = onTogglePlaceLeakMode;
    this.onResetTrails = onResetTrails;
    this.onSelectDrone = onSelectDrone;
    this.onClearSelection = onClearSelection;
    this.onFollowDrone = onFollowDrone;
    this.onFocusPosition = onFocusPosition;
    this.onPlaceLeakAt = onPlaceLeakAt;
    this.onToggleRecording = onToggleRecording;
    this.onCaptureScreenshot = onCaptureScreenshot;
    this.onSelectCameraPreset = onSelectCameraPreset;
    this.onToggleAutoOrbit = onToggleAutoOrbit;
    this.onCycleFollowDrone = onCycleFollowDrone;

    this.plumeVisible = false;
    this.beliefVisible = true;
    this.windVisible = true;
    this.placeLeakMode = false;
    this.paused = false;
    this.presentationMode = false;
    this.hudHidden = false;
    this.shortcutsOpen = false;
    this._isRecording = false;

    this._selectedMeta = null;
    this._followingDroneId = null;
    this._latestDrones = new Map();
    this._latestLeaks = [];
    this._latestBelief = null;
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
    this.elControls = document.getElementById("controls");
    this.elTabpfnMetric = document.getElementById("tabpfn-metric");
    this.elTabpfnCallout = document.getElementById("tabpfn-callout");
    this.elTabpfnCalloutText = document.getElementById("tabpfn-callout-text");

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

    this.btnPresent = document.getElementById("btn-present");
    this.btnPlume = document.getElementById("btn-reveal-plume");
    this.btnBelief = document.getElementById("btn-toggle-belief");
    this.btnWind = document.getElementById("btn-toggle-wind");
    this.btnDropout = document.getElementById("btn-dropout");
    this.btnPlaceLeak = document.getElementById("btn-place-leak");
    this.btnClearLeaks = document.getElementById("btn-clear-leaks");
    this.btnPause = document.getElementById("btn-pause");
    this.btnReset = document.getElementById("btn-reset");

    // Viewer toolbar & overlays
    this.btnRecord = document.getElementById("btn-record");
    this.elRecDot = document.getElementById("rec-dot");
    this.elRecLabel = document.getElementById("btn-record-label");
    this.btnSnapshot = document.getElementById("btn-snapshot");
    this.btnCamPresets = Array.from(document.querySelectorAll("[data-cam-preset]"));
    this.btnFollowCycle = document.getElementById("btn-follow-cycle");
    this.btnAutoOrbit = document.getElementById("btn-auto-orbit");
    this.btnHideHud = document.getElementById("btn-hide-hud");
    this.btnShortcuts = document.getElementById("btn-shortcuts");
    this.elRecFloating = document.getElementById("rec-floating-badge");
    this.elRecFloatingTime = document.getElementById("rec-floating-time");
    this.elShutterFlash = document.getElementById("shutter-flash");
    this.elShortcutsModal = document.getElementById("shortcuts-modal");
    this.btnShortcutsClose = document.getElementById("shortcuts-close");

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

    // Keep wind direction and wind speed sliders independent so changing speed NEVER turns the wind
    this.elWindAngle.addEventListener("input", () => {
      const angle = Number(this.elWindAngle.value);
      this.elWindAngleVal.textContent = String(angle);
      this.onSendCommand({
        action: "set_wind",
        wind_angle_deg: angle,
      });
    });

    this.elWindSpeed.addEventListener("input", () => {
      const speed = Number(this.elWindSpeed.value);
      this.elWindSpeedVal.textContent = speed.toFixed(1);
      this.onSendCommand({
        action: "set_wind",
        wind_speed: speed,
      });
    });

    this.btnPresent.addEventListener("click", () => {
      this.setPresentationMode(!this.presentationMode);
    });

    this.btnPlume.addEventListener("click", () => {
      this.setPlumeVisible(!this.plumeVisible);
    });

    this.btnBelief.addEventListener("click", () => {
      this.setBeliefVisible(!this.beliefVisible);
    });

    this.btnWind?.addEventListener("click", () => {
      this.setWindVisible(!this.windVisible);
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
      this.togglePause();
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
      this.onClearSelection?.();
    });

    this.elScoreBody.addEventListener("click", (evt) => {
      const tr = evt.target.closest("tr[data-drone-id]");
      if (tr) {
        this.onSelectDrone?.(tr.dataset.droneId);
      }
    });

    // Viewer toolbar events
    this.btnRecord?.addEventListener("click", () => {
      this.onToggleRecording?.();
    });
    this.elRecFloating?.addEventListener("click", () => {
      this.onToggleRecording?.();
    });
    this.btnSnapshot?.addEventListener("click", () => {
      this.onCaptureScreenshot?.();
    });
    for (const btn of this.btnCamPresets) {
      btn.addEventListener("click", () => {
        this.onSelectCameraPreset?.(btn.dataset.camPreset);
      });
    }
    this.btnFollowCycle?.addEventListener("click", () => {
      this.onCycleFollowDrone?.();
    });
    this.btnAutoOrbit?.addEventListener("click", () => {
      this.onToggleAutoOrbit?.();
    });
    this.btnHideHud?.addEventListener("click", () => {
      this.setHudHidden(!this.hudHidden);
    });
    this.btnShortcuts?.addEventListener("click", () => {
      this.setShortcutsOpen(!this.shortcutsOpen);
    });
    this.btnShortcutsClose?.addEventListener("click", () => {
      this.setShortcutsOpen(false);
    });
    this.elShortcutsModal?.addEventListener("click", (evt) => {
      if (evt.target === this.elShortcutsModal) {
        this.setShortcutsOpen(false);
      }
    });
  }

  togglePause() {
    this.paused = !this.paused;
    this.btnPause.textContent = this.paused ? "Resume" : "Pause";
    this.btnPause.classList.toggle("btn-active", this.paused);
    this.onSendCommand({ action: "pause" });
    this.showToast(this.paused ? "Simulation paused" : "Simulation resumed", 1500);
  }

  setBeliefVisible(visible) {
    this.beliefVisible = Boolean(visible);
    this.btnBelief.classList.toggle("btn-active", this.beliefVisible);
    this.onToggleBelief(this.beliefVisible);
  }

  setHudHidden(hidden) {
    this.hudHidden = Boolean(hidden);
    document.body.classList.toggle("hud-hidden", this.hudHidden);
    this._syncFloatingRecBadge();
    if (this.hudHidden) {
      this.showToast("HUD hidden — press H to restore", 2200);
    }
  }

  setShortcutsOpen(open) {
    this.shortcutsOpen = Boolean(open);
    this.elShortcutsModal?.classList.toggle("hidden", !this.shortcutsOpen);
    this.btnShortcuts?.classList.toggle("tool-btn-active", this.shortcutsOpen);
  }

  triggerShutterFlash() {
    if (!this.elShutterFlash) return;
    this.elShutterFlash.classList.remove("flash");
    void this.elShutterFlash.offsetWidth;
    this.elShutterFlash.classList.add("flash");
  }

  setRecordingState({ recording, elapsedSec }) {
    this._isRecording = Boolean(recording);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
    const secs = String(Math.floor(elapsedSec % 60)).padStart(2, "0");
    const timeStr = `${mins}:${secs}`;

    this.btnRecord?.classList.toggle("tool-btn-recording", this._isRecording);
    this.elRecDot?.classList.toggle("active", this._isRecording);
    if (this.elRecLabel) {
      this.elRecLabel.textContent = this._isRecording ? `REC ${timeStr}` : "Rec";
    }
    if (this.elRecFloatingTime) {
      this.elRecFloatingTime.textContent = `REC ${timeStr}`;
    }
    this._syncFloatingRecBadge();
  }

  _syncFloatingRecBadge() {
    if (!this.elRecFloating) return;
    this.elRecFloating.classList.toggle("hidden", !(this._isRecording && this.hudHidden));
  }

  setCameraRigState({ autoOrbit, followDroneId, activePreset }) {
    this._followingDroneId = followDroneId || null;

    for (const btn of this.btnCamPresets) {
      btn.classList.toggle("tool-btn-active", btn.dataset.camPreset === activePreset);
    }
    this.btnAutoOrbit?.classList.toggle("tool-btn-active", Boolean(autoOrbit));
    this.btnFollowCycle?.classList.toggle("tool-btn-active", Boolean(followDroneId));
    if (this.btnFollowCycle) {
      this.btnFollowCycle.textContent = followDroneId
        ? `Follow: ${followDroneId.toUpperCase()}`
        : "Follow";
    }

    if (this._selectedMeta?.kind === "drone") {
      this._renderInspector(true);
    }
  }

  getKnownDroneIds() {
    return Array.from(this._latestDrones.keys());
  }

  setPresentationMode(enabled) {
    this.presentationMode = Boolean(enabled);
    this.elControls.classList.toggle("hidden", this.presentationMode);
    document.getElementById("legend")?.classList.toggle("compact", this.presentationMode);
    this.btnPresent.classList.toggle("active", this.presentationMode);
    this.btnPresent.textContent = this.presentationMode ? "Controls" : "Present";
  }

  syncWindSetpoint(angleDeg, speed) {
    if (angleDeg != null) {
      const a = Math.round(((angleDeg % 360) + 360) % 360);
      this.elWindAngle.value = String(a);
      this.elWindAngleVal.textContent = String(a);
    }
    if (speed != null) {
      this.elWindSpeed.value = String(speed);
      this.elWindSpeedVal.textContent = Number(speed).toFixed(1);
    }
  }

  setPlumeVisible(visible) {
    this.plumeVisible = Boolean(visible);
    this.btnPlume.classList.toggle("btn-active", this.plumeVisible);
    this.btnPlume.textContent = this.plumeVisible
      ? "Hide True Plume"
      : "Reveal True Plume";
    this.onTogglePlume(this.plumeVisible);
  }

  setWindVisible(visible) {
    this.windVisible = Boolean(visible);
    this.btnWind?.classList.toggle("btn-active", this.windVisible);
    this.btnWind?.setAttribute("aria-pressed", String(this.windVisible));
    this.onToggleWind?.(this.windVisible);
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
      ? "Click Pipe/Ground (Cancel)"
      : "Place Leak";
    this.elHint.textContent = this.placeLeakMode
      ? "Click any pipe, valve, or ground point to place a leak (Esc to cancel)"
      : "WASD/QE fly (+Shift fast) · Drag orbit · R record · F snap · ? shortcuts";
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
    this._renderInspector(true);
  }

  updateBelief(belief) {
    this._latestBelief = belief;
    this._refreshTabpfnSummary();
  }

  _refreshTabpfnSummary() {
    if (!this._latestBelief || this._latestLeaks.length === 0) {
      this.elTabpfnMetric.textContent = "TabPFN Est: —";
      return;
    }
    const [ex, ey, ez] = this._latestBelief.estimated_source;
    let bestErr = Infinity;
    for (const lk of this._latestLeaks) {
      const d = Math.hypot(ex - lk.pos[0], ey - lk.pos[1], ez - lk.pos[2]);
      if (d < bestErr) bestErr = d;
    }
    this.elTabpfnMetric.textContent = `TabPFN Est Err: ${bestErr.toFixed(1)}m`;
  }

  _renderInspector(rebuildActions = false) {
    const m = this._selectedMeta;
    if (!m) return;

    const rows = [];
    if (rebuildActions) {
      this.elInspectorActions.innerHTML = "";
    }

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

      if (rebuildActions) {
        const isFollowing = this._followingDroneId === m.id;
        const btnFollow = document.createElement("button");
        btnFollow.className = `btn ${isFollowing ? "btn-active" : "btn-accent"}`;
        btnFollow.textContent = isFollowing ? "Stop Following" : "Follow Drone";
        btnFollow.onclick = () => {
          const next = this._followingDroneId === m.id ? null : m.id;
          this._followingDroneId = next;
          this.onFollowDrone?.(next);
          this._renderInspector(true);
        };
        this.elInspectorActions.appendChild(btnFollow);
      }
    } else if (m.kind === "pipe") {
      this.elInspectorType.textContent = "PIPELINE";
      this.elInspectorTitle.textContent = m.name || m.id;
      rows.push(["Line ID", m.id]);
      rows.push(["Service", m.service || "Process"]);
      rows.push(["Pressure", `${m.pressureBar ?? 12.0} bar`]);
      if (m.hitSimPos) {
        rows.push([
          "Point",
          `(${m.hitSimPos[0].toFixed(1)}, ${m.hitSimPos[1].toFixed(1)}, ${m.hitSimPos[2].toFixed(1)})`,
        ]);
        if (rebuildActions) {
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

      if (rebuildActions) {
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
      }
    } else if (m.kind === "building") {
      this.elInspectorType.textContent = m.structureKind.toUpperCase();
      this.elInspectorTitle.textContent = m.id;
      rows.push(["Dimensions", `${m.size[0].toFixed(0)} × ${m.size[1].toFixed(0)} × ${m.size[2].toFixed(0)} m`]);
      rows.push(["Wake Cavity", `${(m.size[2] * 2.5).toFixed(0)} m downwind`]);

      if (rebuildActions) {
        const btnFocus = document.createElement("button");
        btnFocus.className = "btn";
        btnFocus.textContent = "Focus Camera";
        btnFocus.onclick = () => this.onFocusPosition?.(m.centerSim);
        this.elInspectorActions.appendChild(btnFocus);
      }
    }

    this.elInspectorBody.innerHTML = rows
      .map(([k, v]) => `<span class="k">${k}</span><span class="v">${v}</span>`)
      .join("");
  }

  updateBackendStatus(status) {
    this.modelMode=status.mode;
    this.updateReport(status.mission_report);
    const mission=document.getElementById("mission-status");
    const phases={tabpfn_source_prediction:"TabPFN bestimmt das 3D-Messziel",waiting_for_tabpfn:"Messung bereit · warte auf TabPFN",sensor_settling:"Gaskontakt · Sensor beruhigt sich",wind_adapted_acquisition:"Wind geändert · Suchziel angepasst"};
    const states={found:"Leck gefunden — Suche beendet",no_leak:"No active leak — restart mission or place a leak",depleted:"Battery empty — restart mission",paused:"Paused",searching:phases[status.navigation_basis] || (status.search_phase==="track" ? "Gaskontakt · lokale 3D-Quellensuche" : "Suche nach Gaskontakt")};
    mission.textContent=states[status.mission_state] || (status.mode==="mock" ? "Mock mission" : "Searching");
    mission.style.color=["no_leak","depleted"].includes(status.mission_state) ? "#fca5a5" : "#cbd5e1";
    const features=status.source_features ? ` · ${status.source_features} source / ${status.hazard_features} risk inputs` : "";
    document.getElementById("model-status").textContent = `${status.mode} · ${status.state}${features}${status.error ? ": " + status.error : ""}`;
    const plans={preparing:"Map plan: preparing before flight",coverage_fallback:"Map coverage route · TabPFN map dataset pending",tabpfn_map:"Map route scored by TabPFN",injected_test_model:"Map route · test model",mock:"Mock stream · no map-search model",error:"Map planning failed · holding position"};
    document.getElementById("map-plan-status").textContent=`${plans[status.plan_state] || "Map plan: unavailable"}${status.search_candidate_count ? ` · ${status.search_candidate_count} cells · revision ${status.plan_revision || 0}` : ""}`;
    const wind=status.adaptive_wind ? ` · ${status.wind_replans || 0} Windanpassungen` : "";
    document.getElementById("model-metrics").textContent = `${(status.inference_seconds || 0).toFixed(2)} s inference · ${status.measurement_count || 0} measurements · ${status.completed_probes || 0} Messstopps · ${status.prediction_decisions || 0} TabPFN-Ziele${wind} · ${(status.exposure_ppm_seconds || 0).toFixed(1)} ppm·s exposure`;
  }

  updateBelief(belief) {
    document.getElementById("source-estimate").textContent = belief.source_evidence===false ? "Source estimate: awaiting gas measurements" : `Estimated source (x,y,z): ${belief.estimated_source.map(v => v.toFixed(1)).join(", ")} m`;
    this.hazardForecastTime=belief.hazard_forecast_time;
    this.updateRiskTime(belief.sim_time);
  }

  updateRiskTime(simTime) {
    const node=document.getElementById("risk-forecast-time");
    if (!node)return;
    node.textContent=this.hazardForecastTime==null ? "Gefahrenprognose: Zeitbezug nicht verfügbar" : `Risiko für t=${this.hazardForecastTime.toFixed(1)} s${simTime>this.hazardForecastTime ? " · veraltet" : ""}`;
  }

  updateReport(report) {
    this.completed=Boolean(report);
    document.getElementById("mission-report").classList.toggle("hidden",!report);
    for (const id of ["btn-pause","btn-step","btn-clear-leaks","btn-dropout","wind-angle","wind-speed"]) document.getElementById(id).disabled=this.completed;
    if (!report)return;
    document.getElementById("report-duration").textContent=`Suchdauer: ${report.elapsed_sim_seconds.toFixed(1)} s Simulationszeit · ${report.measurement_count} Messungen`;
    document.getElementById("report-position").textContent=`Geschätzte Quelle (x,y,z): ${report.estimated_source.map(x=>x.toFixed(1)).join(", ")} m`;
    document.getElementById("report-score").textContent=report.model_score==null ? "Modellbewertung: nicht verfügbar (z. B. Mock-Modus)" : `Modellbewertung: ${(report.model_score*100).toFixed(1)} % Klassifikationsscore (unkalibriert)`;
    document.getElementById("report-evidence").textContent=`${this.modelMode==="mock" ? "Mock-Fund" : "Im Simulator bestätigt"} · Ortungsfehler ${report.estimated_error_m.toFixed(1)} m · ${report.ppm_at_detection.toFixed(1)} ppm vor Ort`;
    document.getElementById("report-wall").textContent=`Echtzeit seit Reset: ${report.elapsed_wall_seconds.toFixed(1)} s (inkl. Pausen) · Schätzung ${report.estimate_age_seconds.toFixed(1)} s alt`;
  }

  updateTick(tick) {
    this.updateReport(tick.mission_report);
    this.updateRiskTime(tick.sim_time);
    if (tick.paused !== undefined) {
      this.paused=tick.paused;
      this.btnPause.textContent=this.paused ? "Resume" : "Pause";
      this.btnPause.classList.toggle("btn-active",this.paused);
    }
    document.getElementById("btn-step").disabled=this.completed || !this.paused;

    this.elSimTime.textContent = `${tick.sim_time.toFixed(1)}s`;
    this.elLeakCount.textContent = String(tick.leaks.length);
    this._latestLeaks = tick.leaks;
    this._refreshTabpfnSummary();

    const [wx, wy] = tick.wind_vec;
    const speed = Math.hypot(wx, wy);
    const angleDeg = ((Math.atan2(wy, wx) * 180) / Math.PI + 360) % 360;
    this.elWindReadout.textContent = `${speed.toFixed(1)} m/s @ ${Math.round(angleDeg)}°`;

    const svgDeg = 90 - angleDeg;
    this.elWindArrow.setAttribute("transform", `rotate(${svgDeg.toFixed(1)} 40 40)`);

    this.elDropoutBadge.classList.toggle("hidden", !tick.sensor_dropout_active);
    this.btnDropout.classList.toggle("btn-danger-active", tick.sensor_dropout_active);

    if (tick.sensor_dropout_active) {
      this.elTabpfnCallout.classList.add("warn");
      this.elTabpfnCalloutText.textContent =
        "30% NaN Faults Active — TabPFN handles missing wind/temp natively";
    } else {
      this.elTabpfnCallout.classList.remove("warn");
      this.elTabpfnCalloutText.textContent =
        "TabPFN active spatial search — quantile UCB over 3D facility priors";
    }

    for (const d of tick.drones) {
      this._latestDrones.set(d.id, d);
    }
    if (this._selectedMeta?.kind === "drone") {
      this._renderInspector(false);
    }

    // Highlight the leading drone (closest to leak) so jury comparison is instant
    let leaderId = null;
    let bestDist = Infinity;
    for (const d of tick.drones) {
      if (d.dist_to_nearest_leak < bestDist) {
        bestDist = d.dist_to_nearest_leak;
        leaderId = d.id;
      }
    }

    this.elScoreBody.innerHTML = tick.drones
      .map((d) => {
        const col = DRONE_HEX[d.id] ?? "#e2e8f0";
        const distStr = d.dist_to_nearest_leak > 500 ? "—" : `${d.dist_to_nearest_leak.toFixed(1)}m`;
        const isLeader = d.id === leaderId && bestDist < 500;
        const distCls = isLeader || d.dist_to_nearest_leak < 12 ? "dist-close" : "";
        const proxPct = Math.max(4, Math.min(100, 100 - d.dist_to_nearest_leak * 1.4));
        const batPct = Math.max(0, Math.min(100, d.battery_pct));
        const batCls = batPct < 20 ? "low" : "";
        const rowCls = d.id === "tabpfn" ? "row-tabpfn" : "";
        const leaderTag = isLeader ? `<span class="leader-tag">LEAD</span>` : "";
        return `
          <tr class="${rowCls}" data-drone-id="${d.id}" style="cursor:pointer" title="Click to inspect ${d.id}">
            <td><span class="drone-name"><span class="swatch" style="background:${col}"></span>${d.id}${leaderTag}</span></td>
            <td><span class="status-pill status-${d.status}">${d.status}</span></td>
            <td>${d.ppm.toFixed(1)}</td>
            <td class="${distCls}">
              <div class="prox-cell">
                <span>${distStr}</span>
                <div class="prox-bar"><div class="prox-fill" style="width:${proxPct.toFixed(0)}%;background:${col}"></div></div>
              </div>
            </td>
            <td>${d.samples_taken}</td>
            <td>
              <div class="bat-wrap">
                <div class="bat-bar"><div class="bat-fill ${batCls}" style="width:${batPct.toFixed(0)}%"></div></div>
                <span>${batPct.toFixed(0)}%</span>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");
  }
}
