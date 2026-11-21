import "./style.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { threeToSim } from "./coords.js";
import { loadModelAssets } from "./assets.js";
import { setupEnvironment } from "./environment.js";
import { SimWsClient } from "./wsClient.js";
import { RefineryView } from "./refinery.js";
import { PlumeView } from "./plumeView.js";
import { BeliefView } from "./beliefView.js";
import { DroneView } from "./droneView.js";
import { HudController } from "./hud.js";

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

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;
  controls.minDistance = 12;
  controls.maxDistance = 190;
  controls.target.set(0, 6, 0);

  // 2. Load Blender PBR GLB assets
  const models = await loadModelAssets();

  // 3. Scene views
  const refineryView = new RefineryView(scene, models);
  const plumeView = new PlumeView(scene);
  const beliefView = new BeliefView(scene);
  const droneView = new DroneView(scene, models.quadrotor);

  let placeLeakMode = false;

  // 4. WebSocket client & HUD
  const wsClient = new SimWsClient({
    onWorldInit: (worldInit) => {
      refineryView.build(worldInit);
      beliefView.initGrid(worldInit.grid_coords);
      droneView.clearTrails();
    },
    onSimTick: (tick) => {
      plumeView.update(tick.puffs, tick.leaks, tick.wind_vec);
      droneView.update(tick.drones);
      hud.updateTick(tick);
    },
    onBeliefUpdate: (belief) => {
      beliefView.update(belief);
    },
    onStatus: (state, label) => {
      hud.setConnectionStatus(state, label);
    },
  });

  const hud = new HudController({
    onSendCommand: (cmd) => wsClient.sendCommand(cmd),
    onTogglePlume: (visible) => plumeView.setVisible(visible),
    onToggleBelief: (visible) => beliefView.setVisible(visible),
    onTogglePlaceLeakMode: (active) => {
      placeLeakMode = active;
      renderer.domElement.style.cursor = active ? "crosshair" : "default";
    },
    onResetTrails: () => droneView.clearTrails(),
  });

  // 5. Click-on-ground raycast to place a hidden leak
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();

  renderer.domElement.addEventListener("pointerdown", (evt) => {
    if (!placeLeakMode || evt.button !== 0 || !refineryView.groundMesh) return;

    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((evt.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((evt.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(refineryView.groundMesh);
    if (hits.length > 0) {
      const simPos = threeToSim(hits[0].point);
      simPos[2] = 3.5;
      wsClient.sendCommand({
        action: "place_leak",
        leak_pos: simPos,
      });
      hud.setPlaceLeakMode(false);
    }
  });

  // 6. Resize & render loop
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  wsClient.start();

  const clock = new THREE.Clock();
  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.elapsedTime;
    controls.update();
    droneView.animate(dt);
    plumeView.animate(dt);
    beliefView.animate(elapsed);
    renderer.render(scene, camera);
  }
  animate();
}

bootstrap().catch((err) => {
  console.error("Failed to initialize 3D simulation:", err);
});
