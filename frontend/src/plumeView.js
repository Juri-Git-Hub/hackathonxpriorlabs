import * as THREE from "three";
import { simToThree } from "./coords.js";

function createSoftPuffTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const grad = ctx.createRadialGradient(
    size / 2,
    size / 2,
    2,
    size / 2,
    size / 2,
    size / 2
  );
  grad.addColorStop(0.0, "rgba(255, 190, 60, 0.95)");
  grad.addColorStop(0.35, "rgba(251, 146, 60, 0.55)");
  grad.addColorStop(0.7, "rgba(234, 88, 12, 0.18)");
  grad.addColorStop(1.0, "rgba(234, 88, 12, 0.0)");

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/**
 * Renders true gas puffs + leak source beacons (hidden by default; toggled via
 * "Reveal True Plume" button).
 */
export class PlumeView {
  constructor(scene) {
    this.scene = scene;
    this.maxPuffs = 400;
    this.visible = false;

    this.group = new THREE.Group();
    this.group.visible = this.visible;
    this.scene.add(this.group);

    // Volumetric soft-alpha point cloud for puffs
    this.positions = new Float32Array(this.maxPuffs * 3);
    this.colors = new Float32Array(this.maxPuffs * 3);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    this.geometry.setDrawRange(0, 0);

    const material = new THREE.PointsMaterial({
      size: 6.5,
      map: createSoftPuffTexture(),
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(this.geometry, material);
    this.points.frustumCulled = false;
    this.group.add(this.points);

    // True leak markers
    this.leakGroup = new THREE.Group();
    this.group.add(this.leakGroup);
    this._leakKey = "";
    this._tmpVec = new THREE.Vector3();
  }

  setVisible(flag) {
    this.visible = Boolean(flag);
    this.group.visible = this.visible;
  }

  update(puffs, leaks) {
    const count = Math.min(puffs.length, this.maxPuffs);
    for (let i = 0; i < count; i++) {
      const p = puffs[i];
      simToThree(p.pos, this._tmpVec);
      this.positions[i * 3] = this._tmpVec.x;
      this.positions[i * 3 + 1] = this._tmpVec.y;
      this.positions[i * 3 + 2] = this._tmpVec.z;

      const intensity = Math.min(1.0, p.opacity * 1.4);
      this.colors[i * 3] = 1.0 * intensity;
      this.colors[i * 3 + 1] = 0.62 * intensity;
      this.colors[i * 3 + 2] = 0.14 * intensity;
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.setDrawRange(0, count);

    // Rebuild leak source beacons only if leak list changed
    const key = leaks.map((l) => `${l.id}:${l.pos.join(",")}`).join("|");
    if (key !== this._leakKey) {
      this._leakKey = key;
      while (this.leakGroup.children.length > 0) {
        this.leakGroup.remove(this.leakGroup.children.pop());
      }
      const sphereGeo = new THREE.SphereGeometry(1.1, 16, 16);
      const ringGeo = new THREE.RingGeometry(1.6, 2.2, 24);
      for (const lk of leaks) {
        const sphere = new THREE.Mesh(
          sphereGeo,
          new THREE.MeshBasicMaterial({ color: 0xef4444 })
        );
        simToThree(lk.pos, sphere.position);
        this.leakGroup.add(sphere);

        const ring = new THREE.Mesh(
          ringGeo,
          new THREE.MeshBasicMaterial({
            color: 0xf87171,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.7,
          })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(sphere.position.x, 0.15, sphere.position.z);
        this.leakGroup.add(ring);
      }
    }
  }
}
