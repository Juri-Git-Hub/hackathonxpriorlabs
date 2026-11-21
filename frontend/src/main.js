import "./style.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { threeToSim } from "./coords.js";
import { SimWsClient } from "./wsClient.js";
import { RefineryView } from "./refinery.js";
import { PlumeView } from "./plumeView.js";
import { BeliefView } from "./beliefView.js";
import { DroneView } from "./droneView.js";
import { HudController } from "./hud.js";

// 1. Scene, camera, renderer
const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f17);
scene.fog = new THREE.FogExp2(0x0b0f17, 0.0042);

const camera = new THREE.PerspectiveCamera(
  52,
  window.innerWidth / window.innerHeight,
  0.5,
  500
);
camera.position.set(-72, 68, 76);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI / 2 - 0.04;
controls.target.set(0, 6, 0);

// 2. Lighting
const hemiLight = new THREE.HemisphereLight(0x94a3b8, 0x0f172a, 0.9);
scene.add(hemiLight);

const dirLight = new THREE.DirectionalLight(0xf8fafc, 1.1);
dirLight.position.set(55, 85, 45);
scene.add(dirLight);

// 3. Views
const refineryView = new RefineryView(scene);
const plumeView = new PlumeView(scene);
const beliefView = new BeliefView(scene);
const droneView = new DroneView(scene);

let placeLeakMode = false;

// 4. WebSocket client & HUD
const wsClient = new SimWsClient({
  onWorldInit: (worldInit) => {
    refineryView.build(worldInit);
    beliefView.initGrid(worldInit.grid_coords);
    droneView.clearTrails();
    document.getElementById("source-estimate").textContent="Source estimate: —";
  },
  onSimTick: (tick) => {
    plumeView.update(tick.puffs, tick.leaks);
    droneView.update(tick.drones);
    hud.updateTick(tick);
  },
  onBeliefUpdate: (belief) => {
    beliefView.update(belief);
    hud.updateBelief(belief);
  },
  onBackendStatus: (status) => hud.updateBackendStatus(status),
  onStatus: (state, label) => {
    hud.setConnectionStatus(state, label);
  },
});

const hud = new HudController({
  onSendCommand: (cmd) => wsClient.sendCommand(cmd),
  onLayer: (layer) => beliefView.setLayer(layer),
  onRoute: (visible) => beliefView.setRouteVisible(visible),
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
    simPos[2] = Math.max(1,Math.min(28,Number(document.getElementById("leak-height").value)||3.5));
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
  const dt = clock.getDelta();
  const elapsed = clock.elapsedTime;
  controls.update();
  droneView.animate(dt);
  beliefView.animate(elapsed);
  renderer.render(scene, camera);
}
animate();
