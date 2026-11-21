import * as THREE from "three";
import { simToThree } from "./coords.js";

const DRONE_COLORS = {
  tabpfn: 0x22d3ee,     // Neon cyan
  gradient: 0xfb923c,   // Neon orange
  lawnmower: 0x94a3b8,  // Cool gray
  manual: 0xa3e635,
};

const TRAIL_LENGTH = 60;

/**
 * Renders 3D quadrotors with roll/pitch/yaw tilt, spinning rotors, and 60-point
 * flight trails.
 */
export class DroneView {
  constructor(scene) {
    this.scene = scene;
    this.drones = new Map();
    this._tmpVec = new THREE.Vector3();
  }

  _createDroneRig(id) {
    const color = DRONE_COLORS[id] ?? 0x38bdf8;
    const root = new THREE.Group();
    root.rotation.order = "YXZ";

    // Central body
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 0.35, 0.75),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.35,
        roughness: 0.3,
        metalness: 0.5,
      })
    );
    root.add(body);

    // X-frame arms + 4 rotor discs
    const armMat = new THREE.MeshBasicMaterial({ color: 0xcbd5e1 });
    const arm1 = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.1, 0.12), armMat);
    arm1.rotation.y = Math.PI / 4;
    root.add(arm1);
    const arm2 = new THREE.Mesh(new THREE.BoxGeometry(2.6, 0.1, 0.12), armMat);
    arm2.rotation.y = -Math.PI / 4;
    root.add(arm2);

    const rotors = [];
    const rotorOffsets = [
      [0.92, 0.16, 0.92],
      [0.92, 0.16, -0.92],
      [-0.92, 0.16, 0.92],
      [-0.92, 0.16, -0.92],
    ];
    const rotorGeo = new THREE.CylinderGeometry(0.48, 0.48, 0.04, 12);
    const rotorMat = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.55,
    });
    for (const [rx, ry, rz] of rotorOffsets) {
      const rotor = new THREE.Mesh(rotorGeo, rotorMat);
      rotor.position.set(rx, ry, rz);
      root.add(rotor);
      rotors.push(rotor);
    }

    // Forward nose indicator (+X in local frame)
    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(0.22, 0.55, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(0.85, 0.05, 0);
    root.add(nose);

    this.scene.add(root);

    // 60-point flight trail
    const trailPositions = new Float32Array(TRAIL_LENGTH * 3);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    trailGeo.setDrawRange(0, 0);
    const trailMat = new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.7,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    trailLine.frustumCulled = false;
    this.scene.add(trailLine);

    const entry = {
      root,
      rotors,
      trailLine,
      trailPoints: [],
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
      entry.root.position.copy(this._tmpVec);

      // Apply thrust-derived roll/pitch/yaw (Z-up -> Y-up)
      const [roll, pitch, yaw] = d.rot_rpy;
      entry.root.rotation.set(roll, yaw, -pitch, "YXZ");

      // Append trail point if moved enough
      const pts = entry.trailPoints;
      const last = pts[pts.length - 1];
      if (!last || last.distanceToSquared(this._tmpVec) > 0.35) {
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
    for (const entry of this.drones.values()) {
      for (const rotor of entry.rotors) {
        rotor.rotation.y += dt * 28.0;
      }
    }
  }
}
