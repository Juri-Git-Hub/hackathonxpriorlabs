import * as THREE from "three";
import { simToThree } from "./coords.js";

function createLeakBadge(labelText) {
  const canvas = document.createElement("canvas");
  canvas.width = 220;
  canvas.height = 52;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(127, 29, 29, 0.90)";
  ctx.beginPath();
  ctx.roundRect(6, 6, 208, 40, 8);
  ctx.fill();
  ctx.strokeStyle = "#f87171";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.fillStyle = "#fee2e2";
  ctx.font = "800 18px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(labelText, 110, 27);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false })
  );
  sprite.scale.set(6.2, 1.45, 1);
  return sprite;
}

/**
 * High-Contrast Volumetric Turbulent Gas Plume Renderer.
 *
 * Uses a custom GLSL animated FBM + domain-warped volumetric cloud shader with
 * 3D top-lit self-shadowing, high-contrast thermal-vapor color ramp (incandescent
 * gold-amber core -> rich glowing amber-orange billowing cloud -> warm smoky vapor),
 * and smooth 5-sub-puff bridging so the plume is unmistakably readable from every angle.
 */
export class PlumeView {
  constructor(scene) {
    this.scene = scene;
    this.maxPhysicalPuffs = 400;
    this.subPuffs = 3;
    this.maxSprites = this.maxPhysicalPuffs * this.subPuffs;
    this.visible = false;

    this.group = new THREE.Group();
    this.group.visible = this.visible;
    this.scene.add(this.group);

    this.positions = new Float32Array(this.maxSprites * 3);
    this.sizes = new Float32Array(this.maxSprites);
    this.alphas = new Float32Array(this.maxSprites);
    this.densities = new Float32Array(this.maxSprites);
    this.seeds = new Float32Array(this.maxSprites);
    this.angles = new Float32Array(this.maxSprites);

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute("aSize", new THREE.BufferAttribute(this.sizes, 1));
    this.geometry.setAttribute("aAlpha", new THREE.BufferAttribute(this.alphas, 1));
    this.geometry.setAttribute("aDensity", new THREE.BufferAttribute(this.densities, 1));
    this.geometry.setAttribute("aSeed", new THREE.BufferAttribute(this.seeds, 1));
    this.geometry.setAttribute("aAngle", new THREE.BufferAttribute(this.angles, 1));
    this.geometry.setDrawRange(0, 0);

    this._uniforms = {
      uTime: { value: 0.0 },
      uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
    };

    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: this._uniforms,
      vertexShader: `
        attribute float aSize;
        attribute float aAlpha;
        attribute float aDensity;
        attribute float aSeed;
        attribute float aAngle;
        uniform float uTime;
        uniform float uPixelRatio;

        varying float vAlpha;
        varying float vDensity;
        varying vec2 vRot;
        varying vec2 vNoiseOffset;

        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          float camDist = max(-mvPos.z, 1.0);
          float nearFade = smoothstep(3.0, 11.0, camDist);
          vAlpha = aAlpha * nearFade;
          vDensity = aDensity;
          float angle = aAngle + uTime * (0.22 + 0.16 * fract(aSeed * 7.31)) * (fract(aSeed * 3.17) > 0.5 ? 1.0 : -1.0);
          vRot = vec2(cos(angle), sin(angle));
          vNoiseOffset = vec2(aSeed * 23.7 + uTime * 0.25, aSeed * 41.3 - uTime * 0.18);

          // Scale by devicePixelRatio so HiDPI displays render the exact same cohesive cloud size
          gl_PointSize = min(280.0 * uPixelRatio, aSize * (420.0 * uPixelRatio / camDist));
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        varying float vAlpha;
        varying float vDensity;
        varying vec2 vRot;
        varying vec2 vNoiseOffset;

        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          float a = hash(i);
          float b = hash(i + vec2(1.0, 0.0));
          float c = hash(i + vec2(0.0, 1.0));
          float d = hash(i + vec2(1.0, 1.0));
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        float fbm(vec2 p) {
          float v = 0.0;
          float amp = 0.54;
          mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
          for (int i = 0; i < 4; i++) {
            v += amp * noise(p);
            p = rot * p * 2.05 + vec2(1.7, 9.2);
            amp *= 0.48;
          }
          return v;
        }

        void main() {
          vec2 uv = gl_PointCoord * 2.0 - 1.0;
          float r2 = dot(uv, uv);
          if (r2 > 1.0) discard;

          // Unique per-particle UV rotation (precomputed in vertex shader)
          vec2 ruv = vec2(vRot.x * uv.x - vRot.y * uv.y, vRot.y * uv.x + vRot.x * uv.y);

          // Animated domain-warped FBM for billowing turbulent gas
          vec2 p = ruv * 2.2 + vNoiseOffset;
          vec2 warp = vec2(fbm(p), fbm(p + vec2(4.7, 1.9)));
          float n = fbm(p + 1.05 * warp);

          // Rich billowy density silhouette (same lush cloud look, just slightly trimmer)
          float radial = pow(max(0.0, 1.0 - r2), 1.15);
          float billow = radial * (0.45 + 0.95 * n);
          float cloud = smoothstep(0.08, 0.58, billow);
          float alpha = cloud * vAlpha;
          if (alpha < 0.01) discard;

          // 3D volumetric directional lighting (bright sunlit upper-right crests, deeper undersides)
          float lit = clamp(0.45 - 0.36 * uv.y + 0.22 * uv.x + 0.52 * (n - 0.45), 0.12, 1.18);

          // High-contrast Thermal / Optical Gas Imaging palette (pops strongly vs cool twilight & cyan wind):
          // Near leak (high density): incandescent yellow-gold / orange core
          // Mid plume: rich luminous amber-orange billowing cloud
          // Far downwind: warm smoky burnt-amber vapor
          vec3 colDilute = vec3(0.86, 0.44, 0.14);
          vec3 colMid    = vec3(0.99, 0.64, 0.12);
          vec3 colHot    = vec3(1.00, 0.95, 0.52);

          float d = clamp(vDensity, 0.0, 1.0);
          vec3 baseCol = mix(colDilute, colMid, smoothstep(0.0, 0.50, d));
          baseCol = mix(baseCol, colHot, smoothstep(0.45, 0.95, d) * smoothstep(0.25, 0.75, radial));

          vec3 shadowCol = vec3(0.42, 0.16, 0.06);
          vec3 finalCol = mix(shadowCol, baseCol * 1.16, lit);

          gl_FragColor = vec4(finalCol, alpha);
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
    this._windVec = [3.5, 0.8];
    this._pulseMeshes = [];

    // Persistent puff tracker: preserves stable seeds/jitter across ticks & interpolates at 60 FPS
    this._trackedPuffs = [];
    this._nextPuffId = 1;
    this._tickElapsed = 0;
    this._tickInterval = 0.05;
    this._staticAttrsDirty = true;
    this._pendingPuffs = null;
  }

  setVisible(flag) {
    const next = Boolean(flag);
    const turnedOn = next && !this.visible;
    this.visible = next;
    this.group.visible = this.visible;
    if (turnedOn && this._pendingPuffs) {
      const puffs = this._pendingPuffs;
      this._pendingPuffs = null;
      this._updateTrackedPuffs(puffs);
    }
  }

  _makeStableSubParams(id) {
    const subs = [];
    for (let s = 0; s < this.subPuffs; s++) {
      const tNorm = this.subPuffs > 1 ? (s / (this.subPuffs - 1)) * 2.0 - 1.0 : 0.0;
      const h1 = Math.sin(id * 12.9898 + s * 78.233) * 43758.5453;
      const r1 = h1 - Math.floor(h1);
      const h2 = Math.sin(id * 39.346 + s * 11.135) * 24634.6345;
      const r2 = h2 - Math.floor(h2);
      const h3 = Math.sin(id * 73.156 + s * 52.235) * 19532.1234;
      const r3 = h3 - Math.floor(h3);
      const centerWeight = 1.0 - 0.28 * Math.abs(tNorm);
      subs.push({
        tNorm,
        r1,
        r2,
        r3,
        centerWeight,
        seed: r1 * 0.5 + r2 * 0.3 + r3 * 0.2,
        angle: r3 * Math.PI * 2.0,
      });
    }
    return subs;
  }

  _updateTrackedPuffs(puffs) {
    // Match incoming 20 Hz puffs to persistent tracked puffs via 4.0m spatial hash
    // so puff IDs, FBM seeds, and jitter offsets NEVER jump, in O(N) instead of O(N^2)!
    const prevList = this._trackedPuffs;
    const usedPrev = new Uint8Array(prevList.length);
    const nextList = [];
    const maxMatchDist2 = 16.0; // 4.0m max motion per 50ms tick
    const invCell = 0.25; // 1 / 4.0m

    const grid = new Map();
    for (let j = 0; j < prevList.length; j++) {
      const prev = prevList[j];
      const cx = Math.floor(prev.currX * invCell);
      const cy = Math.floor(prev.currY * invCell);
      const cz = Math.floor(prev.currZ * invCell);
      const key = ((cx + 512) * 1024 + (cy + 512)) * 1024 + (cz + 512);
      let bucket = grid.get(key);
      if (!bucket) {
        bucket = [];
        grid.set(key, bucket);
      }
      bucket.push(j);
    }

    const count = Math.min(puffs.length, this.maxPhysicalPuffs);
    for (let i = 0; i < count; i++) {
      const p = puffs[i];
      if (p.opacity < 0.09) continue;
      const tx = p.pos[0];
      const ty = p.pos[2];
      const tz = -p.pos[1];

      const gx = Math.floor(tx * invCell);
      const gy = Math.floor(ty * invCell);
      const gz = Math.floor(tz * invCell);

      let bestIdx = -1;
      let bestD2 = maxMatchDist2;
      for (let ox = -1; ox <= 1; ox++) {
        const kx = (gx + ox + 512) * 1024;
        for (let oy = -1; oy <= 1; oy++) {
          const kxy = (kx + (gy + oy + 512)) * 1024;
          for (let oz = -1; oz <= 1; oz++) {
            const bucket = grid.get(kxy + (gz + oz + 512));
            if (!bucket) continue;
            for (let b = 0; b < bucket.length; b++) {
              const j = bucket[b];
              if (usedPrev[j]) continue;
              const prev = prevList[j];
              const dx = tx - prev.currX;
              const dy = ty - prev.currY;
              const dz = tz - prev.currZ;
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < bestD2) {
                bestD2 = d2;
                bestIdx = j;
              }
            }
          }
        }
      }

      if (bestIdx >= 0) {
        usedPrev[bestIdx] = 1;
        const tr = prevList[bestIdx];
        tr.prevX = tr.currX;
        tr.prevY = tr.currY;
        tr.prevZ = tr.currZ;
        tr.targetX = tx;
        tr.targetY = ty;
        tr.targetZ = tz;
        tr.prevRxy = tr.currRxy;
        tr.prevRz = tr.currRz;
        tr.prevOp = tr.currOp;
        tr.targetRxy = p.radius_xy;
        tr.targetRz = p.radius_z;
        tr.targetOp = p.opacity;
        nextList.push(tr);
      } else {
        const id = this._nextPuffId++;
        nextList.push({
          id,
          ageVis: 0,
          prevX: tx,
          prevY: ty,
          prevZ: tz,
          currX: tx,
          currY: ty,
          currZ: tz,
          targetX: tx,
          targetY: ty,
          targetZ: tz,
          prevRxy: p.radius_xy,
          prevRz: p.radius_z,
          prevOp: p.opacity,
          currRxy: p.radius_xy,
          currRz: p.radius_z,
          currOp: p.opacity,
          targetRxy: p.radius_xy,
          targetRz: p.radius_z,
          targetOp: p.opacity,
          subs: this._makeStableSubParams(id),
        });
      }
    }

    this._trackedPuffs = nextList;
    this._tickElapsed = 0;
    this._staticAttrsDirty = true;
    this._writeBuffers(0);
  }

  update(puffs, leaks, windVec) {
    if (windVec) {
      this._windVec = windVec;
    }

    if (this.visible) {
      this._pendingPuffs = null;
      this._updateTrackedPuffs(puffs);
    } else {
      // Defer puff matching & GPU buffer uploads until the plume is actually revealed
      this._pendingPuffs = puffs;
    }

    const key = leaks.map((l) => `${l.id}:${l.pos.join(",")}`).join("|");
    if (key !== this._leakKey) {
      this._leakKey = key;
      this._pulseMeshes = [];
      while (this.leakGroup.children.length > 0) {
        this.leakGroup.remove(this.leakGroup.children.pop());
      }
      const coreGeo = new THREE.SphereGeometry(0.65, 16, 16);
      const ringGeo = new THREE.RingGeometry(1.3, 1.85, 28);
      for (let idx = 0; idx < leaks.length; idx++) {
        const lk = leaks[idx];
        const core = new THREE.Mesh(
          coreGeo,
          new THREE.MeshBasicMaterial({ color: 0xff3b30 })
        );
        simToThree(lk.pos, core.position);
        this.leakGroup.add(core);

        const ring = new THREE.Mesh(
          ringGeo,
          new THREE.MeshBasicMaterial({
            color: 0xf87171,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
          })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.set(core.position.x, 0.16, core.position.z);
        this.leakGroup.add(ring);
        this._pulseMeshes.push(ring);

        const stem = new THREE.Mesh(
          new THREE.CylinderGeometry(0.05, 0.05, Math.max(0.2, core.position.y), 8),
          new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.55 })
        );
        stem.position.set(core.position.x, core.position.y * 0.5, core.position.z);
        this.leakGroup.add(stem);

        const badge = createLeakBadge(`TRUE LEAK #${idx + 1}`);
        badge.position.set(core.position.x, core.position.y + 4.8, core.position.z);
        this.leakGroup.add(badge);
      }
    }
  }

  _writeBuffers(dt) {
    const [wx, wy] = this._windVec;
    const spd = Math.max(0.2, Math.hypot(wx, wy));
    const ux = wx / spd;
    const uz = -wy / spd;

    const alphaLerp = Math.min(1.0, this._tickElapsed / this._tickInterval);
    let writeIdx = 0;

    const writeStatic = this._staticAttrsDirty;
    for (let i = 0; i < this._trackedPuffs.length; i++) {
      const tr = this._trackedPuffs[i];
      tr.ageVis += dt;
      tr.currX = tr.prevX + (tr.targetX - tr.prevX) * alphaLerp;
      tr.currY = tr.prevY + (tr.targetY - tr.prevY) * alphaLerp;
      tr.currZ = tr.prevZ + (tr.targetZ - tr.prevZ) * alphaLerp;
      tr.currRxy = tr.prevRxy + (tr.targetRxy - tr.prevRxy) * alphaLerp;
      tr.currRz = tr.prevRz + (tr.targetRz - tr.prevRz) * alphaLerp;
      tr.currOp = tr.prevOp + (tr.targetOp - tr.prevOp) * alphaLerp;

      const spawnFade = Math.min(1.0, tr.ageVis / 0.22);
      const density = Math.max(0.0, Math.min(1.0, (tr.currOp - 0.04) / 0.78));
      const baseSize = Math.min(20.0, Math.max(3.2, tr.currRxy * 2.65));
      const baseAlpha = Math.min(0.56, 0.12 + Math.pow(density, 0.72) * 0.44) * spawnFade;

      for (let s = 0; s < tr.subs.length; s++) {
        const sub = tr.subs[s];
        const offsetAlong = sub.tNorm * Math.min(2.4, tr.currRxy * 0.52);
        const offsetLat = (sub.r1 - 0.5) * tr.currRxy * 0.62;
        const offsetY = (sub.r2 - 0.5) * tr.currRz * 0.68;

        this.positions[writeIdx * 3] = tr.currX + ux * offsetAlong - uz * offsetLat;
        this.positions[writeIdx * 3 + 1] = Math.max(0.55, tr.currY + offsetY);
        this.positions[writeIdx * 3 + 2] = tr.currZ + uz * offsetAlong + ux * offsetLat;

        this.sizes[writeIdx] = baseSize * (0.82 + 0.36 * sub.r3) * sub.centerWeight;
        this.alphas[writeIdx] = baseAlpha * (0.60 + 0.40 * sub.centerWeight);
        this.densities[writeIdx] = density * (0.80 + 0.20 * sub.centerWeight);
        if (writeStatic) {
          this.seeds[writeIdx] = sub.seed;
          this.angles[writeIdx] = sub.angle;
        }
        writeIdx++;
      }
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.aSize.needsUpdate = true;
    this.geometry.attributes.aAlpha.needsUpdate = true;
    this.geometry.attributes.aDensity.needsUpdate = true;
    if (writeStatic) {
      this.geometry.attributes.aSeed.needsUpdate = true;
      this.geometry.attributes.aAngle.needsUpdate = true;
      this._staticAttrsDirty = false;
    }
    this.geometry.setDrawRange(0, writeIdx);
  }

  animate(dt) {
    this._uniforms.uTime.value += dt;
    if (this.visible && this._trackedPuffs.length > 0) {
      this._tickElapsed += dt;
      this._writeBuffers(dt);
    }
    const t = this._uniforms.uTime.value;
    for (let i = 0; i < this._pulseMeshes.length; i++) {
      const s = 1.0 + 0.16 * Math.sin(t * 3.5 + i);
      this._pulseMeshes[i].scale.setScalar(s);
    }
  }
}
