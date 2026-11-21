import * as THREE from "three";
import { simToThree } from "./coords.js";

const DRONE_COLORS = {
  tabpfn: 0x22d3ee,
  gradient: 0xfb923c,
  lawnmower: 0xe2e8f0,
  manual: 0xa3e635,
};

const DRONE_LABELS = {
  tabpfn: "TABPFN",
  gradient: "GRADIENT",
  lawnmower: "LAWNMOWER",
};

const TRAIL_LENGTH = 70;

function createLabelSprite(text, hexColor) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(11, 15, 23, 0.78)";
  ctx.beginPath();
  ctx.roundRect(12, 10, 232, 44, 10);
  ctx.fill();

  const colorStr = `#${hexColor.toString(16).padStart(6, "0")}`;
  ctx.strokeStyle = colorStr;
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = colorStr;
  ctx.font = "bold 24px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 33);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(5.4, 1.35, 1);
  sprite.position.set(0, 2.3, 0);
  return sprite;
}

/**
 * Renders Blender PBR industrial quadrotors with per-agent glowing accents,
 * spinning rotors, smooth 60 FPS interpolation, altitude drop-lines, and trails.
 */
export class DroneView {
  constructor(scene, quadrotorTemplate) {
    this.scene = scene;
    this.quadrotorTemplate = quadrotorTemplate;
    this.drones = new Map();
    this.interactiveObjects = [];
    this._tmpVec = new THREE.Vector3();
  }

  getInteractiveObjects() {
    return this.interactiveObjects;
  }

  getDroneWorldPos(id, target = new THREE.Vector3()) {
    const entry = this.drones.get(id);
    if (!entry || !entry.initialized) return null;
    return target.copy(entry.root.position);
  }

  _createDroneRig(id) {
    const colorHex = DRONE_COLORS[id] ?? 0x38bdf8;
    const root = new THREE.Group();

    // Inner tilt rig so label sprite stays upright
    const bodyRig = this.quadrotorTemplate.clone(true);
    bodyRig.rotation.order = "YXZ";
    bodyRig.scale.setScalar(1.35);

    const rotors = [];
    const accentMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      emissive: colorHex,
      emissiveIntensity: 2.4,
      roughness: 0.2,
      metalness: 0.1,
    });

    bodyRig.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = true;
        if (obj.name.startsWith("AccentLED") || obj.name.startsWith("NavLED")) {
          obj.material = accentMat;
        }
        if (obj.name.startsWith("Rotor_")) {
          rotors.push(obj);
        }
      }
    });
    root.add(bodyRig);

    // Floating crisp callsign badge
    const label = createLabelSprite(DRONE_LABELS[id] ?? id.toUpperCase(), colorHex);
    root.add(label);

    // Generous invisible click-target sphere so flying drones are easy to click
    const hitSphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.2, 10, 8),
      new THREE.MeshBasicMaterial({ visible: false })
    );
    hitSphere.userData.inspect = { kind: "drone", id };
    root.add(hitSphere);
    this.interactiveObjects.push(hitSphere);

    this.scene.add(root);

    // Subtle vertical altitude drop-line to ground
    const dropGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, -5, 0),
    ]);
    const dropMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.25,
    });
    const dropLine = new THREE.Line(dropGeo, dropMat);
    dropLine.frustumCulled = false;
    this.scene.add(dropLine);

    // 70-point flight trail
    const trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    trailGeo.setDrawRange(0, 0);
    const trailMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.78,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    trailLine.frustumCulled = false;
    this.scene.add(trailLine);

    const entry = {
      root,
      bodyRig,
      rotors,
      dropLine,
      trailLine,
      trailPoints: [],
      targetPos: new THREE.Vector3(),
      targetRpy: [0, 0, 0],
      initialized: false,
    };
    this.drones.set(id, entry);
    return entry;
  }

  clearTrails() {
    for (const entry of this.drones.values()) {
      entry.trailPoints = [];
      entry.trailLine.geometry.setDrawRange(0, 0);
    }
  }

  update(droneStates) {
    for (const d of droneStates) {
      let entry = this.drones.get(d.id);
      if (!entry) {
        entry = this._createDroneRig(d.id);
      }

      simToThree(d.pos, this._tmpVec);
      entry.targetPos.copy(this._tmpVec);
      entry.targetRpy = d.rot_rpy;

      if (!entry.initialized) {
        entry.root.position.copy(this._tmpVec);
        entry.initialized = true;
      }

      // Append trail point
      const pts = entry.trailPoints;
      const last = pts[pts.length - 1];
      if (!last || last.distanceToSquared(this._tmpVec) > 0.45) {
        pts.push(this._tmpVec.clone());
        if (pts.length > TRAIL_LENGTH) {
          pts.shift();
        }
        const arr = entry.trailLine.geometry.attributes.position.array;
        for (let i = 0; i < pts.length; i++) {
          arr[i * 3] = pts[i].x;
          arr[i * 3 + 1] = pts[i].y;
          arr[i * 3 + 2] = pts[i].z;
        }
        entry.trailLine.geometry.attributes.position.needsUpdate = true;
        entry.trailLine.geometry.setDrawRange(0, pts.length);
      }
    }
  }

  animate(dt) {
    const lerpAlpha = Math.min(1.0, dt * 14.0);
    for (const entry of this.drones.values()) {
      if (!entry.initialized) continue;

      entry.root.position.lerp(entry.targetPos, lerpAlpha);

      const [roll, pitch, yaw] = entry.targetRpy;
      entry.bodyRig.rotation.set(roll, yaw, -pitch, "YXZ");

      // Spin propellers rapidly
      for (let i = 0; i < entry.rotors.length; i++) {
        const dir = i % 2 === 0 ? 1 : -1;
        entry.rotors[i].rotation.y += dt * 42.0 * dir;
      }

      // Update altitude drop line from drone to ground (y = 0.05)
      const p = entry.root.position;
      const posAttr = entry.dropLine.geometry.attributes.position;
      posAttr.setXYZ(0, p.x, p.y - 0.5, p.z);
      posAttr.setXYZ(1, p.x, 0.05, p.z);
      posAttr.needsUpdate = true;
    }
  }
}
