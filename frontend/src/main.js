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
import { PlumeView, VolumetricPlumePass } from "./plumeView.js";
import { WindView } from "./windView.js";
import { BeliefView } from "./beliefView.js";
import { DroneView } from "./droneView.js";
import industrialMaps from "./industrialMaps.json";
import { HudController } from "./hud.js";
import { CameraRig } from "./cameraRig.js";
import { CanvasRecorder } from "./recorder.js";
import { SelectionHighlighter } from "./selectionHighlight.js";

async function bootstrap() {
  // 1. Scene, camera, renderer
  const container = document.getElementById("canvas-container");
  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    48,
    window.innerWidth / window.innerHeight,
    0.5,
    1400
  );
  camera.position.set(-64, 46, 68);

  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const envController = setupEnvironment(scene, renderer);

  // Create WebGLRenderTarget with a DepthTexture so the 3D volumetric gas cloud raymarcher
  // occludes pixel-perfectly around every tank, pipe, building, and drone!
  const pr = renderer.getPixelRatio();
  const rtWidth = Math.max(1, Math.floor(window.innerWidth * pr));
  const rtHeight = Math.max(1, Math.floor(window.innerHeight * pr));
  const composerTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
    type: THREE.HalfFloatType,
    depthBuffer: true,
    depthTexture: new THREE.DepthTexture(rtWidth, rtHeight, THREE.UnsignedIntType),
  });

  // Create WebGLRenderTarget with a DepthTexture so the 3D volumetric gas cloud raymarcher
  // occludes pixel-perfectly around every tank, pipe, building, and drone!
  const pr = renderer.getPixelRatio();
  const rtWidth = Math.max(1, Math.floor(window.innerWidth * pr));
  const rtHeight = Math.max(1, Math.floor(window.innerHeight * pr));
  const composerTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
    type: THREE.HalfFloatType,
    depthBuffer: true,
    depthTexture: new THREE.DepthTexture(rtWidth, rtHeight, THREE.UnsignedIntType),
  });

  // Subtle cinema bloom (high threshold so only lights, windows, flare, & neon LEDs glow)
  const composer = new EffectComposer(renderer, composerTarget);
  // Give renderTarget2 its own DepthTexture so ping-pong passes never hit a WebGL2 depth feedback loop!
  composer.renderTarget2.depthTexture = new THREE.DepthTexture(
    rtWidth,
    rtHeight,
    THREE.UnsignedIntType
  );
  composer.addPass(new RenderPass(scene, camera));

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

  // Add depth-aware 3D Volumetric Gas Cloud raymarching pass, followed by bloom & output
  const plumePass = new VolumetricPlumePass(plumeView, camera);
  composer.addPass(plumePass);
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.18,
    0.30,
    0.90
  );
  composer.addPass(bloomPass);

  // Whole-object hover & selection silhouette + subtle surface highlight
  const selectionHighlighter = new SelectionHighlighter(scene, camera, rtWidth, rtHeight);
  for (const pass of selectionHighlighter.getPasses()) {
    composer.addPass(pass);
  }

  composer.addPass(new OutputPass());

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

  let placeLeakMode = false;

  function clearSelection() {
    selectionHighlighter.clear();
    hud.clearInspector();
  }

  function selectDroneById(droneId, focusCamera = false) {
    const meta = { kind: "drone", id: droneId };
    hud.showInspector(meta);
    const highlightObj = droneView.getDroneHighlightObject(droneId);
    selectionHighlighter.select(highlightObj, meta);
    if (focusCamera) {
      const pos = droneView.getDroneWorldPos(droneId);
      if (pos) {
        cameraRig.focusPoint(pos);
      }
    }
  }

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
        const themeSurface = {
          refinery_dusk: { surface: "concrete", zone_label: "GENERATED · PETROCHEMICAL", flare_pos: [48, 46, 0] },
          desert_sun: { surface: "desert", zone_label: "GENERATED · DESERT BOOSTER" },
          arctic_frost: { surface: "frost", zone_label: "GENERATED · CRYOGENIC DEPOT" },
          alpine_steel: { surface: "asphalt", zone_label: "GENERATED · ALPINE STATION" },
          overcast_steam: { surface: "wet_slate", zone_label: "GENERATED · THERMAL PLANT" },
        };
        const basePresentation = industrialMaps[mapId]?.presentation ?? {};
        const resolvedTheme = worldInit.theme ?? basePresentation.theme ?? "refinery_dusk";
        const presentation = {
          ...(themeSurface[resolvedTheme] ?? {}),
          ...basePresentation,
          theme: resolvedTheme,
        };
        envController.setTheme(presentation.theme);
        selectionHighlighter.clearAll();
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
      plumeView.setWindField(windField);
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
      if (active) selectionHighlighter.setHovered(null);
      renderer.domElement.style.cursor = active ? "crosshair" : "default";
    },
    onResetTrails: () => droneView.clearTrails(),
    onSelectDrone: (droneId) => {
      selectDroneById(droneId, true);
    },
    onClearSelection: () => {
      selectionHighlighter.clear();
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
    if (hits.length > 0) {
      const hitObj = hits[0].object;
      selectionHighlighter.setHovered(hitObj.userData.inspectRoot ?? hitObj);
      renderer.domElement.style.cursor = "pointer";
    } else {
      selectionHighlighter.setHovered(null);
      renderer.domElement.style.cursor = "default";
    }
  });

  renderer.domElement.addEventListener("pointerleave", () => {
    selectionHighlighter.setHovered(null);
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
        const rootObj = hit.object.userData.inspectRoot ?? hit.object;
        if (selectionHighlighter.isSelected(rootObj)) {
          clearSelection();
          selectionHighlighter.setHovered(rootObj);
        } else {
          const hitSimPos = threeToSim(hit.point);
          hud.showInspector(meta, hitSimPos);
          selectionHighlighter.select(rootObj, meta, hit.point);
        }
      }
    } else {
      clearSelection();
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
        clearSelection();
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
      selectDroneById(id, false);
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
      clearSelection();
    },
    inspectAsset: () => {
      const obj = refineryView.getInteractiveObjects().find((o) => o.userData.inspect?.kind === "asset");
      if (obj) {
        const meta = obj.userData.inspect;
        const rootObj = obj.userData.inspectRoot ?? obj;
        hud.showInspector(meta, meta.posSim);
        selectionHighlighter.select(rootObj, meta, simToThree(meta.posSim));
      }
    },
    inspectPipe: () => {
      const obj = refineryView.getInteractiveObjects().find((o) => o.userData.inspect?.kind === "pipe");
      if (obj) {
        const meta = obj.userData.inspect;
        const rootObj = obj.userData.inspectRoot ?? obj;
        const pt = [-6, -2, 4.4];
        hud.showInspector(meta, pt);
        selectionHighlighter.select(rootObj, meta, simToThree(pt));
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
    selectionHighlighter.animate(elapsed);
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
