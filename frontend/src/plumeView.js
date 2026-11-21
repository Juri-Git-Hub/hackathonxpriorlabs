import * as THREE from "three";
import { simToThree } from "./coords.js";

function createWispySmokeTexture() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const lobes = [
    [64, 64, 58, 0.52],
    [50, 56, 40, 0.36],
    [78, 60, 38, 0.34],
    [62, 76, 36, 0.32],
    [64, 48, 32, 0.28],
  ];
  for (const [cx, cy, r, alpha] of lobes) {
    const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, r);
    grad.addColorStop(0.0, `rgba(255, 210, 105, ${alpha})`);
    grad.addColorStop(0.45, `rgba(249, 138, 42, ${alpha * 0.65})`);
    grad.addColorStop(1.0, "rgba(234, 88, 12, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/**
 * Renders physically-scaled volumetric wispy gas puffs (using per-puff `radius_xy`
 * & `opacity` shaders) + true leak beacons, plus always-on subtle wind streaks.
 */
export class PlumeView {
  constructor(scene) {
    this.scene = scene;
    this.maxPuffs = 400;
    this.visible = false;

    this.group = new THREE.Group();
    this.group.visible = this.visible;
    this.scene.add(this.group);

    this.positions = new Float32Array(this.maxPuffs * 3);
    this.sizes = new Float32Array(this.maxPuffs);
    this.alphas = new Float32Array(this.maxPuffs);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("aSize", new THREE.BufferAttribute(this.sizes, 1));
    this.geometry.setAttribute("aAlpha", new THREE.BufferAttribute(this.alphas, 1));
    this.geometry.setDrawRange(0, 0);

    const smokeTex = createWispySmokeTexture();
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uMap: { value: smokeTex },
      },
      vertexShader: `
        attribute float aSize;
        attribute float aAlpha;
        varying float vAlpha;
        void main() {
          vAlpha = aAlpha;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * (340.0 / max(-mvPos.z, 1.0));
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        varying float vAlpha;
        void main() {
          vec4 tex = texture2D(uMap, gl_PointCoord);
          float a = tex.a * vAlpha;
          if (a < 0.01) discard;
          gl_FragColor = vec4(tex.rgb, a);
        }
      `,
    });

    this.points = new THREE.Points(this.geometry, material);
    this.points.frustumCulled = false;
    this.group.add(this.points);

    this.leakGroup = new THREE.Group();
    this.group.add(this.leakGroup);
    this._leakKey = "";
    this._tmpVec = new THREE.Vector3();

    // Always-visible subtle ambient wind streaks drifting across the refinery
    this._windVec = [3.5, 0.8];
    this._windCount = 75;
    this._windPos = new Float32Array(this._windCount * 3);
    for (let i = 0; i < this._windCount; i++) {
      this._windPos[i * 3] = (Math.random() - 0.5) * 120;
      this._windPos[i * 3 + 1] = 2 + Math.random() * 20;
      this._windPos[i * 3 + 2] = (Math.random() - 0.5) * 120;
    }
    const windGeo = new THREE.BufferGeometry();
    windGeo.setAttribute("position", new THREE.BufferAttribute(this._windPos, 3));
    const windMat = new THREE.PointsMaterial({
      size: 0.45,
      color: 0xdbeafe,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
    });
    this.windParticles = new THREE.Points(windGeo, windMat);
    this.scene.add(this.windParticles);
  }

  setVisible(flag) {
    this.visible = Boolean(flag);
    this.group.visible = this.visible;
  }

  update(puffs, leaks, windVec) {
    if (windVec) {
      this._windVec = windVec;
    }

    const count = Math.min(puffs.length, this.maxPuffs);
    for (let i = 0; i < count; i++) {
      const p = puffs[i];
      simToThree(p.pos, this._tmpVec);
      this.positions[i * 3] = this._tmpVec.x;
      this.positions[i * 3 + 1] = this._tmpVec.y;
      this.positions[i * 3 + 2] = this._tmpVec.z;

      // Drive visual puff diameter directly from physical sigma_xy
      this.sizes[i] = Math.min(26.0, Math.max(3.2, p.radius_xy * 3.4));
      this.alphas[i] = Math.min(0.82, p.opacity * 1.15);
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.aSize.needsUpdate = true;
    this.geometry.attributes.aAlpha.needsUpdate = true;
    this.geometry.setDrawRange(0, count);

    const key = leaks.map((l) => `${l.id}:${l.pos.join(",")}`).join("|");
    if (key !== this._leakKey) {
      this._leakKey = key;
      while (this.leakGroup.children.length > 0) {
        this.leakGroup.remove(this.leakGroup.children.pop());
      }
      const sphereGeo = new THREE.SphereGeometry(0.95, 16, 16);
      const ringGeo = new THREE.RingGeometry(1.5, 2.2, 24);
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
            opacity: 0.8,
          })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(sphere.position.x, 0.12, sphere.position.z);
        this.leakGroup.add(ring);
      }
    }
  }

  animate(dt) {
    const [wx, wy] = this._windVec;
    const vx = wx * 1.6;
    const vz = -wy * 1.6;
    const arr = this._windPos;
    for (let i = 0; i < this._windCount; i++) {
      let x = arr[i * 3] + vx * dt;
      let z = arr[i * 3 + 2] + vz * dt;
      if (x > 60) x -= 120;
      if (x < -60) x += 120;
      if (z > 60) z -= 120;
      if (z < -60) z += 120;
      arr[i * 3] = x;
      arr[i * 3 + 2] = z;
    }
    this.windParticles.geometry.attributes.position.needsUpdate = true;
  }
}
