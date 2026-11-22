import * as THREE from "three";
import { simToThree } from "./coords.js";

const DRONE_COLORS = {
  tabpfn: 0x22d3ee,
  wind_plume: 0xa78bfa,
  gradient: 0xfb923c,
  lawnmower: 0xe2e8f0,
  manual: 0xa3e635,
};

const DRONE_LABELS = {
  tabpfn: "SEECH4 · TABPFN",
  wind_plume: "WIND-PLUME",
  gradient: "GRADIENT",
  lawnmower: "LAWNMOWER",
};

const TRAIL_LENGTH = 72;

function createLabelSprite(text, hexColor) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 60;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(10, 15, 24, 0.86)";
  ctx.beginPath();
  ctx.roundRect(10, 8, 236, 44, 10);
  ctx.fill();

  const colorStr = `#${hexColor.toString(16).padStart(6, "0")}`;
  ctx.strokeStyle = colorStr;
  ctx.lineWidth = 3.5;
  ctx.stroke();

  ctx.fillStyle = colorStr;
  ctx.font = "800 24px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 31);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(4.8, 1.15, 1);
  sprite.position.set(0, 2.55, 0);
  return sprite;
}

/**
 * Renders prominent Blender PBR industrial quadrotors (2.05x visual scale) with:
 * - High-contrast glowing LEDs & counter-rotating propellers
 * - Glowing ground tracking halo rings + vertical altitude drop-lines
 * - Bold 3D ribbon trails (clearly traceable even from top-down tactical views)
 * - Generous invisible click hit-spheres for effortless interaction
 */
export class DroneView {
  constructor(scene, quadrotorTemplate) {
    this.scene = scene;
    this.quadrotorTemplate = quadrotorTemplate;
    this.drones = new Map();
    this.interactiveObjects = [];
    this._tmpVec = new THREE.Vector3();
    this._tmpDir = new THREE.Vector3();
    this._tmpSide = new THREE.Vector3();
    this._up = new THREE.Vector3(0, 1, 0);
  }

  getInteractiveObjects() {
    return this.interactiveObjects;
  }

  getDroneWorldPos(id, target = new THREE.Vector3()) {
    const entry = this.drones.get(id);
    if (!entry || !entry.initialized) return null;
    return target.copy(entry.root.position);
  }

  getDroneHighlightObject(id) {
    const entry = this.drones.get(id);
    return entry ? entry.bodyRig : null;
  }

  _createDroneRig(id) {
    const colorHex = DRONE_COLORS[id] ?? 0x38bdf8;
    const root = new THREE.Group();

    const bodyRig = this.quadrotorTemplate.clone(true);
    bodyRig.rotation.order = "YXZ";
    bodyRig.scale.setScalar(2.30);

    const rotors = [];
    const accentMat = new THREE.MeshStandardMaterial({
      color: colorHex,
      emissive: colorHex,
      emissiveIntensity: 3.2,
      roughness: 0.18,
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

    const label = createLabelSprite(DRONE_LABELS[id] ?? id.toUpperCase(), colorHex);
    root.add(label);

    // Downward translucent chemical sensor sampling cone (brightens when detecting gas)
    const scanMat = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const scanCone = new THREE.Mesh(new THREE.ConeGeometry(1.8, 3.8, 16, 1, true), scanMat);
    scanCone.position.set(0, -2.1, 0);
    root.add(scanCone);

    const hitSphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.8, 10, 8),
      new THREE.MeshBasicMaterial({ visible: false })
    );
    hitSphere.visible = false;
    hitSphere.userData.inspect = { kind: "drone", id };
    hitSphere.userData.inspectRoot = bodyRig;
    root.add(hitSphere);
    this.interactiveObjects.push(hitSphere);

    this.scene.add(root);

    // Glowing ground tracking ring beneath the drone (makes position super clear top-down!)
    const groundRing = new THREE.Mesh(
      new THREE.RingGeometry(1.15, 1.75, 24),
      new THREE.MeshBasicMaterial({
        color: colorHex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.72,
        depthWrite: false,
      })
    );
    groundRing.rotation.x = -Math.PI / 2;
    groundRing.position.y = 0.08;
    this.scene.add(groundRing);

    // Vertical altitude drop-line to ground ring
    const dropGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, -5, 0),
    ]);
    const dropMat = new THREE.LineBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity: 0.42,
    });
    const dropLine = new THREE.Line(dropGeo, dropMat);
    dropLine.frustumCulled = false;
    this.scene.add(dropLine);

    // Bold 3D ribbon trail (2 vertices per sample -> triangle strip via index buffer)
    const maxVerts = TRAIL_LENGTH * 2;
    const ribbonPos = new Float32Array(maxVerts * 3);
    const ribbonColors = new Float32Array(maxVerts * 3);
    const indices = [];
    for (let i = 0; i < TRAIL_LENGTH - 1; i++) {
      const a = i * 2;
      const b = a + 1;
      const c = a + 2;
      const d = a + 3;
      indices.push(a, b, c, b, d, c);
    }
    const ribbonGeo = new THREE.BufferGeometry();
    ribbonGeo.setAttribute("position", new THREE.BufferAttribute(ribbonPos, 3));
    ribbonGeo.setAttribute("color", new THREE.BufferAttribute(ribbonColors, 3));
    ribbonGeo.setIndex(indices);
    ribbonGeo.setDrawRange(0, 0);

    const ribbonMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const ribbonMesh = new THREE.Mesh(ribbonGeo, ribbonMat);
    ribbonMesh.frustumCulled = false;
    this.scene.add(ribbonMesh);

    const entry = {
      root,
      bodyRig,
      rotors,
      scanMat,
      hitSphere,
      groundRing,
      dropLine,
      ribbonMesh,
      baseColor: new THREE.Color(colorHex),
      trailPoints: [],
      targetPos: new THREE.Vector3(),
      targetRpy: [0, 0, 0],
      lastPpm: 0,
      initialized: false,
    };
    this.drones.set(id, entry);
    return entry;
  }

  clearTrails() {
    for (const entry of this.drones.values()) {
      entry.trailPoints = [];
      entry.ribbonMesh.geometry.setDrawRange(0, 0);
    }
  }

  _rebuildRibbon(entry) {
    const pts = entry.trailPoints;
    const n = pts.length;
    if (n < 2) {
      entry.ribbonMesh.geometry.setDrawRange(0, 0);
      return;
    }

    const posArr = entry.ribbonMesh.geometry.attributes.position.array;
    const colArr = entry.ribbonMesh.geometry.attributes.color.array;
    const c = entry.baseColor;

    for (let i = 0; i < n; i++) {
      const p = pts[i];
      const frac = i / (n - 1); // 0 = oldest tail, 1 = newest head
      const halfWidth = 0.12 + 0.34 * frac;

      if (i < n - 1) {
        this._tmpDir.subVectors(pts[i + 1], p);
      } else {
        this._tmpDir.subVectors(p, pts[i - 1]);
      }
      if (this._tmpDir.lengthSq() < 1e-6) {
        this._tmpDir.set(1, 0, 0);
      } else {
        this._tmpDir.normalize();
      }

      this._tmpSide.crossVectors(this._tmpDir, this._up);
      if (this._tmpSide.lengthSq() < 1e-4) {
        this._tmpSide.set(1, 0, 0);
      } else {
        this._tmpSide.normalize();
      }
      this._tmpSide.multiplyScalar(halfWidth);

      const vi = i * 6;
      posArr[vi] = p.x - this._tmpSide.x;
      posArr[vi + 1] = p.y;
      posArr[vi + 2] = p.z - this._tmpSide.z;
      posArr[vi + 3] = p.x + this._tmpSide.x;
      posArr[vi + 4] = p.y;
      posArr[vi + 5] = p.z + this._tmpSide.z;

      const fade = frac * frac;
      colArr[vi] = c.r * fade;
      colArr[vi + 1] = c.g * fade;
      colArr[vi + 2] = c.b * fade;
      colArr[vi + 3] = c.r * fade;
      colArr[vi + 4] = c.g * fade;
      colArr[vi + 5] = c.b * fade;
    }

    entry.ribbonMesh.geometry.attributes.position.needsUpdate = true;
    entry.ribbonMesh.geometry.attributes.color.needsUpdate = true;
    entry.ribbonMesh.geometry.setDrawRange(0, (n - 1) * 6);
  }

  update(droneStates) {
    const active = new Set(droneStates.map((d) => d.id));
    for (const [id, entry] of this.drones) {
      if (!active.has(id)) {
        this.scene.remove(entry.root, entry.groundRing, entry.dropLine, entry.ribbonMesh);
        entry.root.traverse((obj) => {
          obj.geometry?.dispose();
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material?.dispose();
        });
        entry.groundRing.geometry.dispose();
        entry.groundRing.material.dispose();
        entry.dropLine.geometry.dispose();
        entry.dropLine.material.dispose();
        entry.ribbonMesh.geometry.dispose();
        entry.ribbonMesh.material.dispose();
        const hitIdx = this.interactiveObjects.indexOf(entry.hitSphere);
        if (hitIdx !== -1) this.interactiveObjects.splice(hitIdx, 1);
        this.drones.delete(id);
      }
    }
    for (const d of droneStates) {
      let entry = this.drones.get(d.id);
      if (!entry) {
        entry = this._createDroneRig(d.id);
      }

      simToThree(d.pos, this._tmpVec);
      entry.targetPos.copy(this._tmpVec);
      entry.targetRpy = d.rot_rpy;
      entry.lastPpm = d.ppm;
      // Brighten sensor cone when drone sniffs elevated gas concentration
      entry.scanMat.opacity = d.ppm > 5.0 ? 0.38 : 0.14;

      if (!entry.initialized) {
        entry.root.position.copy(this._tmpVec);
        entry.initialized = true;
      }

      const pts = entry.trailPoints;
      const last = pts[pts.length - 1];
      if (!last || last.distanceToSquared(this._tmpVec) > 0.45) {
        pts.push(this._tmpVec.clone());
        if (pts.length > TRAIL_LENGTH) {
          pts.shift();
        }
        this._rebuildRibbon(entry);
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

      for (let i = 0; i < entry.rotors.length; i++) {
        const dir = i % 2 === 0 ? 1 : -1;
        entry.rotors[i].rotation.y += dt * 42.0 * dir;
      }

      const p = entry.root.position;
      entry.groundRing.position.set(p.x, 0.08, p.z);

      const posAttr = entry.dropLine.geometry.attributes.position;
      posAttr.setXYZ(0, p.x, p.y - 0.6, p.z);
      posAttr.setXYZ(1, p.x, 0.08, p.z);
      posAttr.needsUpdate = true;
    }
  }
}
