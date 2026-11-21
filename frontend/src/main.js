import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";

import { simToThree, threeToSim } from "./coords.js";
import { loadModelAssets } from "./assets.js";
import { setupEnvironment } from "./environment.js";
import { SimWsClient } from "./wsClient.js";
import { RefineryView } from "./refinery.js";
import { PlumeView } from "./plumeView.js";
import { WindView } from "./windView.js";
import { BeliefView } from "./beliefView.js";
import { DroneView } from "./droneView.js";
import industrialMaps from "./industrialMaps.json";
import { HudController } from "./hud.js";
import { CameraRig } from "./cameraRig.js";
import { CanvasRecorder } from "./recorder.js";

async function bootstrap() {
  // 1. Scene, camera, renderer
  const container = document.getElementById("canvas-container");
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    48,
    window.innerWidth / window.innerHeight,
    0.5,
    600
  );
  camera.position.set(-64, 46, 68);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  setupEnvironment(scene, renderer);

  // Subtle cinema bloom (high threshold so only lights, windows, flare, & neon LEDs glow)
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.18, // restrained cinema bloom (never washes out close-ups)
    0.30,
    0.90
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;
  controls.minDistance = 8;
  controls.maxDistance = 190;
  controls.target.set(0, 6, 0);

  // 2. Load Blender PBR GLB assets
  const models = await loadModelAssets();

  // 3. Scene views
  const refineryView = new RefineryView(scene, models);
  const plumeView = new PlumeView(scene);
  const windView = new WindView(scene);
  const beliefView = new BeliefView(scene);
  const droneView = new DroneView(scene, models.quadrotor);

  // Live 3D placement reticle ring (visible when Place Leak mode is active)
  const reticle = new THREE.Mesh(
    new THREE.RingGeometry(1.1, 1.65, 28),
    new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    })
  );
  reticle.rotation.x = -Math.PI / 2;
  reticle.visible = false;
  scene.add(reticle);

  // Expanding pulse rings when a leak is placed (instant visual confirmation)
  const pulseRings = [];
  function spawnLeakPulse(simPos) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.8, 1.35, 28),
      new THREE.MeshBasicMaterial({
        color: 0xef4444,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.95,
        depthTest: false,
      })
    );
    ring.rotation.x = -Math.PI / 2;
    simToThree(simPos, ring.position);
    scene.add(ring);
    pulseRings.push({ mesh: ring, age: 0 });
  }

  // Glowing 3D selection highlight ring (connects Inspector panel to exact 3D object!)
  const selectionRing = new THREE.Mesh(
    new THREE.TorusGeometry(2.4, 0.16, 12, 32),
    new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.92,
      depthTest: false,
    })
  );
  selectionRing.rotation.x = Math.PI / 2;
  selectionRing.visible = false;
  scene.add(selectionRing);
  let selectedDroneId = null;

  let placeLeakMode = false;

  const cameraRig = new CameraRig(camera, controls, {
    onStateChange: (state) => hud.setCameraRigState(state),
  });

  const recorder = new CanvasRecorder(renderer.domElement, {
    fps: 60,
    bitrate: 10_000_000,
    onStateChange: (state) => hud.setRecordingState(state),
    onSaved: (filename, _size, kind) => {
      hud.showToast(
        kind === "screenshot"
          ? `Screenshot saved: ${filename}`
          : `Recording saved: ${filename}`,
        3000
      );
    },
    onError: (msg) => hud.showToast(msg, 3200),
  });

  function cycleFollowDrone() {
    const ids = hud.getKnownDroneIds();
    const order = ids.length > 0 ? ids : ["tabpfn", "gradient", "lawnmower"];
    const curIdx = cameraRig.followDroneId ? order.indexOf(cameraRig.followDroneId) : -1;
    const nextId = curIdx + 1 < order.length ? order[curIdx + 1] : null;
    cameraRig.setFollowDrone(nextId);
    hud.showToast(
      nextId ? `Following drone: ${nextId.toUpperCase()}` : "Camera: Free flight",
      1800
    );
  }

  function triggerRecordingToggle() {
    const started = recorder.toggle();
    if (started) hud.showToast("Recording started (press R to stop)", 2000);
  }

  function triggerScreenshot() {
    hud.triggerShutterFlash();
    recorder.captureScreenshot(() => composer.render());
  }

  function placeLeakAt(simPos) {
    const pos = [
      Number(simPos[0].toFixed(2)),
      Number(simPos[1].toFixed(2)),
      Math.max(1.0, Number(simPos[2].toFixed(2))),
    ];
    wsClient.sendCommand({
      action: "place_leak",
      leak_pos: pos,
      replace_existing: true,
    });
    spawnLeakPulse(pos);
    // Automatically reveal plume if hidden so the user immediately sees the new leak
    if (!hud.plumeVisible) {
      hud.setPlumeVisible(true);
    }
    hud.showToast(
      `Starting a new leak search at (${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)}m)`
    );
  }

  const worldButtons = document.getElementById("world-buttons");
  const generateWorld = document.getElementById("btn-new-layout");
  const mapButtons = new Map();
  for (const [id, { name }] of Object.entries(industrialMaps)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn";
    button.textContent = name;
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      if (id !== currentMap) wsClient.sendCommand({ action: "select_map", map_id: id });
    });
    worldButtons.insertBefore(button, generateWorld);
    mapButtons.set(id, button);
  }
  let currentMap = null;
  let currentLayoutSeed = null;

  // 4. WebSocket client & HUD
  const wsClient = new SimWsClient({
    onWorldInit: (worldInit) => {
      const mapId = worldInit.map_id ?? "refinery";
      const layoutSeed = worldInit.layout_seed ?? null;
      const layoutStatus = document.getElementById("layout-status");
      const worldName = industrialMaps[mapId]?.name ?? "Generated World";
      layoutStatus.textContent = layoutSeed === null ? worldName : `${worldName} · Seed ${layoutSeed}`;
      if (mapId !== currentMap || layoutSeed !== currentLayoutSeed) {
        const presentation = industrialMaps[mapId]?.presentation;
        refineryView.build(worldInit, presentation);
        hud.clearInspector();
        hud.setPlaceLeakMode(false);
        cameraRig.setImmediate(
          presentation?.camera ?? [-64, 46, 68],
          presentation?.target ?? [0, 6, 0],
        );
        for (const { mesh } of pulseRings.splice(0)) {
          scene.remove(mesh);
          mesh.geometry.dispose();
          mesh.material.dispose();
        }
        plumeView.update([], [], [0, 0]);
        currentMap = mapId;
        currentLayoutSeed = layoutSeed;
      }
      for (const [id, button] of mapButtons) {
        const selected = id === mapId;
        button.disabled = false;
        button.classList.toggle("btn-active", selected);
        button.setAttribute("aria-pressed", String(selected));
      }
      generateWorld.disabled = wsClient.mockMode;
      beliefView.initGrid(worldInit.grid_coords);
      droneView.clearTrails();
      document.getElementById("source-estimate").textContent="Source estimate: —";
      hud.hazardForecastTime=null;
      hud.updateRiskTime(0);
    },
    onWindField: (windField) => {
      windView.setWindField(windField);
    },
    onSimTick: (tick) => {
      plumeView.update(tick.puffs, tick.leaks, tick.wind_vec);
      windView.setAmbientWind(tick.wind_vec);
      droneView.update(tick.drones);
      hud.updateTick(tick);
    },
    onBeliefUpdate: (belief) => {
      beliefView.update(belief);
      hud.updateBelief(belief);
    },
    onBackendStatus: status => { hud.updateBackendStatus(status);if(status.planned_route)beliefView.setRoute(status.planned_route); },
    onError: (error) => hud.showToast(error.detail),
    onStatus: (state, label) => {
      hud.setConnectionStatus(state, label);
      if (state !== "mock" && state !== "connected") {
        for (const button of mapButtons.values()) button.disabled = true;
        generateWorld.disabled = true;
      }
    },
  });

  const hud = new HudController({
    onSendCommand: (cmd) => wsClient.sendCommand(cmd.action==="reset_race" ? {...cmd,keep_leaks:true} : cmd),
    onLayer: layer => beliefView.setLayer(layer),
    onRoute: visible => beliefView.setRouteVisible(visible),
    onTogglePlume: (visible) => plumeView.setVisible(visible),
    onToggleBelief: (visible) => beliefView.setVisible(visible),
    onToggleWind: (visible) => windView.setVisible(visible),
    onTogglePlaceLeakMode: (active) => {
      placeLeakMode = active;
      reticle.visible = false;
      renderer.domElement.style.cursor = active ? "crosshair" : "default";
    },
    onResetTrails: () => droneView.clearTrails(),
    onSelectDrone: (droneId) => {
      selectedDroneId = droneId;
      hud.showInspector({ kind: "drone", id: droneId });
      const pos = droneView.getDroneWorldPos(droneId);
      if (pos) {
        cameraRig.focusPoint(pos);
      }
    },
    onFollowDrone: (droneId) => {
      cameraRig.setFollowDrone(droneId);
    },
    onFocusPosition: (simPos) => {
      cameraRig.focusPoint(simToThree(simPos, new THREE.Vector3()));
    },
    onPlaceLeakAt: (simPos) => {
      placeLeakAt(simPos);
    },
    onToggleRecording: () => triggerRecordingToggle(),
    onCaptureScreenshot: () => triggerScreenshot(),
    onSelectCameraPreset: (presetKey) => cameraRig.setPreset(presetKey),
    onToggleAutoOrbit: () => {
      const active = cameraRig.toggleAutoOrbit();
      hud.showToast(active ? "Auto-Orbit enabled" : "Auto-Orbit stopped", 1600);
    },
    onCycleFollowDrone: () => cycleFollowDrone(),
  });

  // 5. Interactive Raycasting (hover reticle + click drones/pipes/valves/buildings + place leaks)
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downX = 0;
  let downY = 0;

  function setPointerFromEvent(evt) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function raycastInteractive(evt) {
    setPointerFromEvent(evt);
    raycaster.setFromCamera(pointer, camera);
    const targets = [
      ...droneView.getInteractiveObjects(),
      ...refineryView.getInteractiveObjects(),
    ];
    return raycaster.intersectObjects(targets, false);
  }

  function raycastLeakPlacement(evt) {
    setPointerFromEvent(evt);
    raycaster.setFromCamera(pointer, camera);
    const targets = [...refineryView.getInteractiveObjects()];
    if (refineryView.groundMesh) targets.push(refineryView.groundMesh);
    return raycaster.intersectObjects(targets, false);
  }

  renderer.domElement.addEventListener("pointermove", (evt) => {
    if (placeLeakMode) {
      const hits = raycastLeakPlacement(evt);
      if (hits.length > 0) {
        const hit = hits[0];
        reticle.position.copy(hit.point);
        reticle.position.y = Math.max(0.15, hit.point.y + 0.08);
        reticle.visible = true;
      } else {
        reticle.visible = false;
      }
      return;
    }

    const hits = raycastInteractive(evt);
    renderer.domElement.style.cursor = hits.length > 0 ? "pointer" : "default";
  });

  renderer.domElement.addEventListener("pointerdown", (evt) => {
    downX = evt.clientX;
    downY = evt.clientY;
  });

  renderer.domElement.addEventListener("pointerup", (evt) => {
    if (evt.button !== 0) return;
    // Ignore orbit/pan drags
    if (Math.hypot(evt.clientX - downX, evt.clientY - downY) > 6) return;

    if (placeLeakMode) {
      const hits = raycastLeakPlacement(evt);
      if (hits.length > 0) {
        const hit = hits[0];
        const simPos = threeToSim(hit.point);
        // If clicked on ground, elevate to realistic pipe height; on pipe/valve use exact 3D height
        if (hit.object === refineryView.groundMesh) {
          simPos[2] = Math.max(1,Math.min(28,Number(document.getElementById("leak-height").value)||3.5));
        }
        placeLeakAt(simPos);
        hud.setPlaceLeakMode(false);
      }
      return;
    }

    const hits = raycastInteractive(evt);
    if (hits.length > 0) {
      const hit = hits[0];
      const meta = hit.object.userData.inspect;
      if (meta) {
        const hitSimPos = threeToSim(hit.point);
        hud.showInspector(meta, hitSimPos);
        if (meta.kind === "drone") {
          selectedDroneId = meta.id;
        } else {
          selectedDroneId = null;
          selectionRing.position.copy(hit.point);
          selectionRing.scale.setScalar(meta.kind === "building" ? 3.5 : 1.0);
          selectionRing.visible = true;
        }
      }
    } else {
      selectedDroneId = null;
      selectionRing.visible = false;
      hud.clearInspector();
    }
  });

  window.addEventListener("keydown", (evt) => {
    const activeEl = document.activeElement;
    const isEditable =
      activeEl &&
      (activeEl.tagName === "INPUT" ||
        activeEl.tagName === "TEXTAREA" ||
        activeEl.tagName === "SELECT" ||
        activeEl.isContentEditable);

    if (evt.key === "Escape") {
      if (hud.shortcutsOpen) {
        hud.setShortcutsOpen(false);
      } else if (placeLeakMode) {
        hud.setPlaceLeakMode(false);
      } else {
        selectedDroneId = null;
        selectionRing.visible = false;
        hud.clearInspector();
      }
      return;
    }

    if (evt.metaKey || evt.ctrlKey || isEditable) return;

    const k = evt.key.toLowerCase();
    if (evt.key === "?" || (evt.key === "/" && evt.shiftKey)) {
      evt.preventDefault();
      hud.setShortcutsOpen(!hud.shortcutsOpen);
    } else if (k === "p") {
      hud.setPresentationMode(!hud.presentationMode);
    } else if (k === "h") {
      hud.setHudHidden(!hud.hudHidden);
    } else if (k === "r") {
      triggerRecordingToggle();
    } else if (k === "f") {
      triggerScreenshot();
    } else if (k === "o") {
      const active = cameraRig.toggleAutoOrbit();
      hud.showToast(active ? "Auto-Orbit enabled" : "Auto-Orbit stopped", 1600);
    } else if (k === "v") {
      hud.setPlumeVisible(!hud.plumeVisible);
    } else if (k === "b") {
      hud.setBeliefVisible(!hud.beliefVisible);
    } else if (evt.code === "Space") {
      evt.preventDefault();
      hud.togglePause();
    } else if (evt.key === "Tab") {
      evt.preventDefault();
      cycleFollowDrone();
    } else if (evt.key === "1" || evt.key === "0" || evt.key === "Home") {
      cameraRig.setPreset("overview");
    } else if (evt.key === "2") {
      cameraRig.setPreset("top");
    } else if (evt.key === "3") {
      cameraRig.setPreset("low");
    }
  });

  // 6. Resize & render loop
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
  });

  generateWorld.addEventListener("click", () => {
    wsClient.sendCommand({ action: "randomize_layout", map_id: "generated" });
  });
  if (wsClient.mockMode) generateWorld.title = "World generation requires the simulation server.";
  wsClient.start();

  // Lightweight debug/camera helper for deterministic multi-angle inspection
  window.__simDebug = {
    cameraRig,
    recorder,
    setCamera: (pos, target) => cameraRig.setImmediate(pos, target),
    setPreset: (key) => cameraRig.setPreset(key),
    toggleOrbit: () => cameraRig.toggleAutoOrbit(),
    toggleRecording: () => triggerRecordingToggle(),
    captureScreenshot: () => triggerScreenshot(),
    selectDrone: (id) => {
      selectedDroneId = id;
      hud.showInspector({ kind: "drone", id });
    },
    focusOnDrone: (id, offset = [-10, 6, 12]) => {
      const p = droneView.getDroneWorldPos(id);
      if (p) {
        cameraRig.setImmediate(
          [p.x + offset[0], p.y + offset[1], p.z + offset[2]],
          [p.x, p.y, p.z]
        );
      }
    },
    placeLeak: (pos) => placeLeakAt(pos),
    setPlume: (v) => hud.setPlumeVisible(v),
    setBelief: (v) => hud.setBeliefVisible(v),
    setWindField: (v) => hud.setWindVisible(v),
    sendCommand: (cmd) => {
      if (cmd.action === "set_wind") {
        hud.syncWindSetpoint(cmd.wind_angle_deg, cmd.wind_speed);
      }
      wsClient.sendCommand(cmd);
    },
    setPresentationMode: (v) => hud.setPresentationMode(v),
    setHudHidden: (v) => hud.setHudHidden(v),
    clearInspector: () => {
      selectedDroneId = null;
      selectionRing.visible = false;
      hud.clearInspector();
    },
    inspectAsset: () => {
      const obj = refineryView.getInteractiveObjects().find((o) => o.userData.inspect?.kind === "asset");
      if (obj) {
        selectedDroneId = null;
        hud.showInspector(obj.userData.inspect, obj.userData.inspect.posSim);
        simToThree(obj.userData.inspect.posSim, selectionRing.position);
        selectionRing.scale.setScalar(1.0);
        selectionRing.visible = true;
      }
    },
    inspectPipe: () => {
      const obj = refineryView.getInteractiveObjects().find((o) => o.userData.inspect?.kind === "pipe");
      if (obj) {
        selectedDroneId = null;
        const pt = [-6, -2, 4.4];
        hud.showInspector(obj.userData.inspect, pt);
        simToThree(pt, selectionRing.position);
        selectionRing.scale.setScalar(1.0);
        selectionRing.visible = true;
      }
    },
  };

  const clock = new THREE.Clock();
  const tmpFollow = new THREE.Vector3();

  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.elapsedTime;

    cameraRig.update(dt, (id, out) => droneView.getDroneWorldPos(id, out));

    // Animate leak placement pulse rings
    for (let i = pulseRings.length - 1; i >= 0; i--) {
      const pr = pulseRings[i];
      pr.age += dt;
      const t = pr.age / 1.2;
      if (t >= 1.0) {
        scene.remove(pr.mesh);
        pr.mesh.geometry.dispose();
        pr.mesh.material.dispose();
        pulseRings.splice(i, 1);
      } else {
        pr.mesh.scale.setScalar(1.0 + t * 5.0);
        pr.mesh.material.opacity = (1.0 - t) * 0.9;
      }
    }

    controls.update();
    droneView.animate(dt);
    const tabPos = droneView.getDroneWorldPos("tabpfn", tmpFollow);
    if (tabPos) beliefView.setTabpfnDroneWorldPos(tabPos);
    if (selectedDroneId) {
      const sp = droneView.getDroneWorldPos(selectedDroneId, tmpFollow);
      if (sp) {
        selectionRing.position.copy(sp);
        selectionRing.scale.setScalar(1.35 + 0.12 * Math.sin(elapsed * 5.0));
        selectionRing.visible = true;
      }
    } else if (selectionRing.visible) {
      selectionRing.rotation.z = elapsed * 1.4;
    }
    plumeView.animate(dt);
    windView.animate(dt);
    beliefView.animate(elapsed);
    composer.render();
  }
  animate();
}

bootstrap().catch((err) => {
  console.error("Failed to initialize 3D simulation:", err);
});
