import * as THREE from "three";
import { Pass, FullScreenQuad } from "three/examples/jsm/postprocessing/Pass.js";
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
  sprite.scale.set(6.0, 1.4, 1);
  return sprite;
}

/**
 * 3D Volumetric Gas Cloud Field (`48 x 16 x 48` WebGL2 Data3DTexture).
 *
 * Zero particles, zero sprites, zero popping or cotton-ball artifacts:
 * - Physical Lagrangian puffs from the backend are splatted into a continuous 3D scalar
 *   density volume (`x in [-60,60]`, `y in [0,26]`, `z in [-60,60]` in Three.js world space).
 * - Temporally smoothed at 60 FPS so motion between 20 Hz ticks is completely fluid.
 * - Rendered by `VolumetricPlumePass` via depth-aware 3D volumetric raymarching with
 *   animated 3D turbulent wisps and directional sunlight self-shadowing.
 */
export class PlumeView {
  constructor(scene) {
    this.scene = scene;
    this.visible = true;

    // 3D Volume grid dimensions (48 x 16 x 48 = 36,864 voxels -> <0.3 ms GPU upload)
    this.nx = 48;
    this.ny = 16;
    this.nz = 48;
    this.boxMin = new THREE.Vector3(-60, 0, -60);
    this.boxMax = new THREE.Vector3(60, 30, 60);

    const total = this.nx * this.ny * this.nz;
    this._densityTarget = new Float32Array(total);
    this._densityCurr = new Float32Array(total);
    this._texData = new Uint8Array(total);

    this.volumeTexture = new THREE.Data3DTexture(
      this._texData,
      this.nx,
      this.ny,
      this.nz
    );
    this.volumeTexture.format = THREE.RedFormat;
    this.volumeTexture.type = THREE.UnsignedByteType;
    this.volumeTexture.minFilter = THREE.LinearFilter;
    this.volumeTexture.magFilter = THREE.LinearFilter;
    this.volumeTexture.unpackAlignment = 1;
    this.volumeTexture.needsUpdate = true;

    this._hasDensity = false;
    this._time = 0;
    this._flowPhase = 0;
    this._windVec = [3.5, 0.8];

    // 3D Local Wind Velocity Texture (48 x 8 x 48 RGBA, encodes local Three.js vel in [-12, +12] m/s)
    this.windNx = 48;
    this.windNy = 8;
    this.windNz = 48;
    this.windBoxMin = new THREE.Vector3(-60, 0, -60);
    this.windBoxMax = new THREE.Vector3(60, 20, 60);
    this._windTexData = new Uint8Array(this.windNx * this.windNy * this.windNz * 4);
    // Default neutral zero-velocity (128 = 0 m/s) until first wind_field arrives
    this._windTexData.fill(128);
    this.windTexture = new THREE.Data3DTexture(
      this._windTexData,
      this.windNx,
      this.windNy,
      this.windNz
    );
    this.windTexture.format = THREE.RGBAFormat;
    this.windTexture.type = THREE.UnsignedByteType;
    this.windTexture.minFilter = THREE.LinearFilter;
    this.windTexture.magFilter = THREE.LinearFilter;
    this.windTexture.unpackAlignment = 1;
    this.windTexture.needsUpdate = true;

    // Leak source markers & badges in 3D scene
    this.group = new THREE.Group();
    this.group.visible = this.visible;
    this.scene.add(this.group);

    this.leakGroup = new THREE.Group();
    this.group.add(this.leakGroup);
    this._leakKey = "";
    this._pulseMeshes = [];
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
      this._splatPuffs(puffs);
    }
  }

  /**
   * Upload the backend's solved 3D mass-consistent wind field into a 3D velocity texture
   * so volumetric cloud billows move at the exact local 3D wind velocity (curving around
   * tanks/buildings and staying still in dead-wind lee wakes).
   */
  setWindField(payload) {
    if (!payload || !payload.vectors || !payload.dims) return;
    const [nx, ny, nz] = payload.dims; // sim (x, y, z) = (48, 48, 8)
    const [minX, minY, minZ] = payload.bounds_min;
    const [maxX, maxY, maxZ] = payload.bounds_max;

    // Three.js world bounds: x = sim_x, y = sim_z, z = -sim_y
    this.windBoxMin.set(minX, minZ, -maxY);
    this.windBoxMax.set(maxX, maxZ, -minY);

    if (nx !== this.windNx || nz !== this.windNy || ny !== this.windNz) {
      this.windNx = nx;
      this.windNy = nz;
      this.windNz = ny;
      this._windTexData = new Uint8Array(nx * nz * ny * 4);
      this.windTexture.image = { data: this._windTexData, width: nx, height: nz, depth: ny };
    }

    const vecs = payload.vectors;
    const out = this._windTexData;
    const maxRange = 12.0; // encodes [-12, +12] m/s into [0, 255]

    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        // three_z = -sim_y => invert y index into z_three
        const izThree = ny - 1 - iy;
        const zSliceOff = izThree * nz * nx;
        const simXYOff = (ix * ny + iy) * nz;
        for (let iz = 0; iz < nz; iz++) {
          const srcIdx = (simXYOff + iz) * 3;
          const uSim = vecs[srcIdx];
          const vSim = vecs[srcIdx + 1];
          const wSim = vecs[srcIdx + 2];

          // Convert sim velocity (u, v, w) -> Three.js velocity (u, w, -v)
          const vx = Math.max(-1, Math.min(1, uSim / maxRange));
          const vy = Math.max(-1, Math.min(1, wSim / maxRange));
          const vz = Math.max(-1, Math.min(1, -vSim / maxRange));

          const dstIdx = (zSliceOff + iz * nx + ix) * 4;
          out[dstIdx] = Math.round((vx * 0.5 + 0.5) * 255);
          out[dstIdx + 1] = Math.round((vy * 0.5 + 0.5) * 255);
          out[dstIdx + 2] = Math.round((vz * 0.5 + 0.5) * 255);
          out[dstIdx + 3] = 255;
        }
      }
    }
    this.windTexture.needsUpdate = true;
  }

  _splatPuffs(puffs) {
    // 1. Splat live physical puffs into 3D continuous density field (_densityTarget)
    const target = this._densityTarget;
    target.fill(0);

    const nx = this.nx;
    const ny = this.ny;
    const nz = this.nz;
    const minX = this.boxMin.x;
    const minY = this.boxMin.y;
    const minZ = this.boxMin.z;
    const dx = (this.boxMax.x - minX) / nx;
    const dy = (this.boxMax.y - minY) / ny;
    const dz = (this.boxMax.z - minZ) / nz;
    const invDx = 1.0 / dx;
    const invDy = 1.0 / dy;
    const invDz = 1.0 / dz;

    let any = false;
    for (let pIdx = 0; pIdx < puffs.length; pIdx++) {
      const p = puffs[pIdx];
      if (p.opacity < 0.05) continue;

      // Convert sim (x, y, z) -> Three.js (x, z, -y)
      const cx = p.pos[0];
      const cy = p.pos[2];
      const cz = -p.pos[1];

      // Smooth Gaussian kernel radii (compact so cloud stays sculpted and readable)
      const sigXZ = Math.min(5.6, Math.max(1.7, p.radius_xy * 0.92));
      const sigY = Math.min(3.2, Math.max(1.2, p.radius_z * 0.92));
      const cutXZ = sigXZ * 2.0;
      const cutY = sigY * 2.0;

      const ix0 = Math.max(0, Math.floor((cx - cutXZ - minX) * invDx));
      const ix1 = Math.min(nx - 1, Math.ceil((cx + cutXZ - minX) * invDx));
      const iy0 = Math.max(0, Math.floor((cy - cutY - minY) * invDy));
      const iy1 = Math.min(ny - 1, Math.ceil((cy + cutY - minY) * invDy));
      const iz0 = Math.max(0, Math.floor((cz - cutXZ - minZ) * invDz));
      const iz1 = Math.min(nz - 1, Math.ceil((cz + cutXZ - minZ) * invDz));

      if (ix0 > ix1 || iy0 > iy1 || iz0 > iz1) continue;
      any = true;

      const inv2SigXZ2 = 0.5 / (sigXZ * sigXZ);
      const inv2SigY2 = 0.5 / (sigY * sigY);
      const weight = Math.min(1.0, p.opacity * 0.42);

      for (let iz = iz0; iz <= iz1; iz++) {
        const vz = minZ + (iz + 0.5) * dz - cz;
        const zTerm = vz * vz * inv2SigXZ2;
        if (zTerm > 2.4) continue;
        const zOff = iz * ny * nx;
        for (let iy = iy0; iy <= iy1; iy++) {
          const vy = minY + (iy + 0.5) * dy - cy;
          const yzTerm = zTerm + vy * vy * inv2SigY2;
          if (yzTerm > 2.4) continue;
          const yzOff = zOff + iy * nx;
          for (let ix = ix0; ix <= ix1; ix++) {
            const vx = minX + (ix + 0.5) * dx - cx;
            const e = yzTerm + vx * vx * inv2SigXZ2;
            if (e > 2.4) continue;
            // Fast smooth Gaussian splat
            target[yzOff + ix] += weight * Math.exp(-e);
          }
        }
      }
    }
    this._hasDensity = any;
  }

  update(puffs, leaks, windVec) {
    if (windVec) {
      this._windVec = windVec;
    }

    if (this.visible) {
      this._pendingPuffs = null;
      this._splatPuffs(puffs);
    } else {
      this._pendingPuffs = puffs;
    }

    // 2. Rebuild leak nozzle markers & floating badges when leak list changes
    const key = leaks.map((l) => `${l.id}:${l.pos.join(",")}`).join("|");
    if (key !== this._leakKey) {
      this._leakKey = key;
      this._pulseMeshes = [];
      while (this.leakGroup.children.length > 0) {
        this.leakGroup.remove(this.leakGroup.children.pop());
      }
      const coreGeo = new THREE.SphereGeometry(0.55, 16, 16);
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
        badge.position.set(core.position.x, core.position.y + 4.4, core.position.z);
        this.leakGroup.add(badge);
      }
    }
  }

  animate(dt) {
    this._time += dt;
    // Dual-phase flow-map clock (cycles cleanly every 4.5s with zero shear or pop)
    this._flowPhase = (this._flowPhase + dt / 4.5) % 1.0;

    // Smoothly temporally blend 3D density volume at 60 FPS (zero 20 Hz stutter!)
    if (this.visible) {
      const blend = 1.0 - Math.exp(-dt * 4.0);
      const curr = this._densityCurr;
      const target = this._densityTarget;
      const tex = this._texData;
      const len = curr.length;

      for (let i = 0; i < len; i++) {
        const cVal = curr[i];
        const tVal = target[i];
        if (cVal === 0 && tVal === 0) continue;
        const v = cVal + (tVal - cVal) * blend;
        if (v <= 0.002) {
          curr[i] = tVal === 0 && v < 1e-4 ? 0 : v;
          tex[i] = 0;
        } else {
          curr[i] = v;
          const mapped = 1.0 - Math.exp(-v * 0.75);
          tex[i] = Math.min(255, (mapped * 255) | 0);
        }
      }
      this.volumeTexture.needsUpdate = true;
    }

    const t = this._time;
    for (let i = 0; i < this._pulseMeshes.length; i++) {
      const s = 1.0 + 0.16 * Math.sin(t * 3.5 + i);
      this._pulseMeshes[i].scale.setScalar(s);
    }
  }
}

/**
 * Full-screen Depth-Aware 3D Volumetric Raymarching Pass.
 *
 * Composites the continuous 3D gas cloud directly over the rendered scene with
 * pixel-accurate depth occlusion around every tank, pipe, building, and drone.
 */
export class VolumetricPlumePass extends Pass {
  constructor(plumeView, camera) {
    super();
    this.plumeView = plumeView;
    this.camera = camera;
    this.needsSwap = true;

    this.uniforms = {
      tDiffuse: { value: null },
      tDepth: { value: null },
      uVolume: { value: plumeView.volumeTexture },
      uWindVolume: { value: plumeView.windTexture },
      uBoxMin: { value: plumeView.boxMin.clone() },
      uBoxMax: { value: plumeView.boxMax.clone() },
      uWindMin: { value: plumeView.windBoxMin.clone() },
      uWindMax: { value: plumeView.windBoxMax.clone() },
      uInvProjection: { value: new THREE.Matrix4() },
      uCameraWorld: { value: new THREE.Matrix4() },
      uCameraPos: { value: new THREE.Vector3() },
      uFlowPhase: { value: 0 },
      uEnabled: { value: 0 },
      uDesert: { value: 0 },
    };

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: this.uniforms,
      depthTest: false,
      depthWrite: false,
      vertexShader: `
        out vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;
        precision highp sampler3D;

        uniform sampler2D tDiffuse;
        uniform sampler2D tDepth;
        uniform sampler3D uVolume;
        uniform sampler3D uWindVolume;
        uniform vec3 uBoxMin;
        uniform vec3 uBoxMax;
        uniform vec3 uWindMin;
        uniform vec3 uWindMax;
        uniform mat4 uInvProjection;
        uniform mat4 uCameraWorld;
        uniform vec3 uCameraPos;
        uniform float uFlowPhase;
        uniform float uEnabled;
        uniform float uDesert;

        in vec2 vUv;
        out vec4 fragColor;

        // Fast 3D hash & value noise for smooth billowing cloud turbulence
        float hash3(vec3 p) {
          p = fract(p * 0.3183099 + 0.1);
          p *= 17.0;
          return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }

        float noise3(vec3 x) {
          vec3 i = floor(x);
          vec3 f = fract(x);
          f = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(
              mix(hash3(i + vec3(0,0,0)), hash3(i + vec3(1,0,0)), f.x),
              mix(hash3(i + vec3(0,1,0)), hash3(i + vec3(1,1,0)), f.x),
              f.y
            ),
            mix(
              mix(hash3(i + vec3(0,0,1)), hash3(i + vec3(1,0,1)), f.x),
              mix(hash3(i + vec3(0,1,1)), hash3(i + vec3(1,1,1)), f.x),
              f.y
            ),
            f.z
          );
        }

        float fbm3(vec3 p) {
          float v = 0.0;
          v += 0.55 * noise3(p);
          p = p * 2.07 + vec3(13.1, 7.7, 5.3);
          v += 0.30 * noise3(p);
          p = p * 2.03 + vec3(3.7, 11.3, 19.1);
          v += 0.15 * noise3(p);
          return v;
        }

        vec2 intersectAABB(vec3 ro, vec3 rd, vec3 bMin, vec3 bMax) {
          vec3 invD = 1.0 / rd;
          vec3 t0s = (bMin - ro) * invD;
          vec3 t1s = (bMax - ro) * invD;
          vec3 tsmaller = min(t0s, t1s);
          vec3 tbigger = max(t0s, t1s);
          float tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
          float tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
          return vec2(tNear, tFar);
        }

        // Returns vec3(density, billowFactor, sunRelief) using dual-phase local 3D wind advection
        vec3 sampleCloudDensity(
          vec3 worldPos,
          vec3 uvw,
          vec3 invWindSize,
          vec3 sunStep,
          float phase0T,
          float phase1T,
          float blendW
        ) {
          if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) return vec3(0.0);

          float baseD = texture(uVolume, uvw).r;
          if (baseD < 0.04) return vec3(0.0);

          // Sample exact local 3D velocity (m/s) from the solved Poisson wind grid
          vec3 wUvw = clamp((worldPos - uWindMin) * invWindSize, 0.0, 1.0);
          vec3 localVel = (texture(uWindVolume, wUvw).rgb * 2.0 - 1.0) * 12.0;

          // Dual-phase flow-map blending: advects billows along exact local 3D streamlines,
          // curves around tanks, and stays calm in dead-wind lee wakes with zero shear/pop.
          vec3 advectVel = localVel * 0.90 + vec3(0.0, 0.08, 0.0);
          vec3 p0 = (worldPos - advectVel * phase0T) * 0.19;
          vec3 p1 = (worldPos - advectVel * phase1T) * 0.19;

          float b0 = 0.70 * fbm3(p0) + 0.30 * noise3(p0 * 2.1);
          float b1 = 0.70 * fbm3(p1) + 0.30 * noise3(p1 * 2.1);
          float billow = mix(b0, b1, blendW);

          // Directional billow relief along sunlight (matches active blend phase)
          float b0Sun = noise3(p0 + sunStep);
          float b1Sun = noise3(p1 + sunStep);
          float billowSun = mix(b0Sun, b1Sun, blendW);
          float relief = clamp((billow - billowSun) * 1.6 + 0.5, 0.0, 1.0);

          // Carve clear moving eddies & wisps without saturating the core to a flat solid
          float eroded = smoothstep(0.14, 0.80, baseD * (0.34 + 1.18 * billow));
          float internalContrast = 0.42 + 0.88 * smoothstep(0.24, 0.84, billow);

          // Soft fade near ground & box ceiling
          float yFade = smoothstep(0.0, 0.05, uvw.y) * smoothstep(1.0, 0.88, uvw.y);
          return vec3(eroded * internalContrast * yFade, billow, relief);
        }

        void main() {
          vec4 sceneCol = texture(tDiffuse, vUv);
          if (uEnabled < 0.5) {
            fragColor = sceneCol;
            return;
          }

          // Reconstruct exact world ray & opaque scene distance from depth buffer
          float depth = texture(tDepth, vUv).r;
          vec4 clipPos = vec4(vUv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
          vec4 viewPos = uInvProjection * clipPos;
          viewPos /= viewPos.w;

          float sceneDist = (depth >= 0.99999) ? 1e5 : length(viewPos.xyz);
          vec3 rayDir = normalize((uCameraWorld * vec4(viewPos.xyz, 0.0)).xyz);
          vec3 rayOrigin = uCameraPos;

          vec2 hit = intersectAABB(rayOrigin, rayDir, uBoxMin, uBoxMax);
          float tStart = max(0.0, hit.x);
          float tEnd = min(hit.y, sceneDist);

          if (tEnd <= tStart) {
            fragColor = sceneCol;
            return;
          }

          vec3 invBoxSize = 1.0 / (uBoxMax - uBoxMin);
          vec3 invWindSize = 1.0 / (uWindMax - uWindMin);
          vec3 sunDir = normalize(vec3(0.68, 0.52, 0.62));
          vec3 sunStep = sunDir * 0.42;
          vec3 sunOccOffset = sunDir * 3.0 * invBoxSize;
          float phase0 = fract(uFlowPhase);
          float phase0T = phase0 * 4.5;
          float phase1T = fract(uFlowPhase + 0.5) * 4.5;
          float blendW = abs(2.0 * phase0 - 1.0);

          // 36 jittered raymarch steps (smooth, band-free, 60 FPS)
          const int STEPS = 36;
          float marchLen = min(tEnd - tStart, 145.0);
          float stepSize = marchLen / float(STEPS);

          // Interleaved gradient noise dither eliminates slicing bands
          float dither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
          float t = tStart + dither * stepSize;
          vec3 pos = rayOrigin + rayDir * t;
          vec3 posStep = rayDir * stepSize;
          vec3 uvw = (pos - uBoxMin) * invBoxSize;
          vec3 uvwStep = posStep * invBoxSize;

          float transmittance = 1.0;
          vec3 accumLight = vec3(0.0);

          // Warm sunlit industrial thermal-vapor cloud palette
          vec3 colSunCrest = vec3(0.98, 0.78, 0.36);
          vec3 colMidVapor = vec3(0.92, 0.52, 0.16);
          vec3 colShadow   = vec3(0.36, 0.17, 0.09);
          colSunCrest = mix(colSunCrest, vec3(0.58, 0.72, 0.78), uDesert);
          colMidVapor = mix(colMidVapor, vec3(0.24, 0.39, 0.47), uDesert);
          colShadow = mix(colShadow, vec3(0.08, 0.16, 0.22), uDesert);

          for (int i = 0; i < STEPS; i++) {
            if (t > tEnd || transmittance < 0.04) break;

            vec3 sampleVal = sampleCloudDensity(pos, uvw, invWindSize, sunStep, phase0T, phase1T, blendW);
            float d = sampleVal.x;
            float billow = sampleVal.y;
            float relief = sampleVal.z;

            if (d > 0.005) {
              // Directional sunlight self-shadowing (volume occlusion + moving billow relief)
              float sunOcc = texture(uVolume, clamp(uvw + sunOccOffset, 0.0, 1.0)).r;
              float lightEnergy = exp(-sunOcc * 2.0) * (0.52 + 0.62 * relief);

              // Height-based ambient sky scatter + sunlit rolling billow crests
              float hFrac = clamp((pos.y - uBoxMin.y) * (1.0 / 14.0), 0.0, 1.0);
              vec3 scatterCol = mix(colShadow, colMidVapor, clamp(0.28 + 0.72 * lightEnergy, 0.0, 1.0));
              scatterCol = mix(scatterCol, colSunCrest, pow(clamp(lightEnergy, 0.0, 1.0), 1.6) * (0.28 + 0.62 * billow) * (0.35 + 0.65 * hFrac));

              // Beer-Lambert volumetric absorption (translucent, unobtrusive cloud feel)
              float sigma = d * mix(0.165, 0.24, uDesert);
              float stepTrans = exp(-sigma * stepSize);
              accumLight += transmittance * scatterCol * (1.0 - stepTrans);
              transmittance *= stepTrans;
            }

            t += stepSize;
            pos += posStep;
            uvw += uvwStep;
          }

          vec3 finalRgb = sceneCol.rgb * transmittance + accumLight;
          fragColor = vec4(finalRgb, 1.0);
        }
      `,
    });

    this.fsQuad = new FullScreenQuad(material);
  }

  render(renderer, writeBuffer, readBuffer) {
    const pv = this.plumeView;
    const enabled = pv.visible && pv._hasDensity;

    this.uniforms.tDiffuse.value = readBuffer.texture;
    this.uniforms.tDepth.value = readBuffer.depthTexture || null;
    this.uniforms.uEnabled.value = enabled ? 1.0 : 0.0;

    if (enabled) {
      this.uniforms.uInvProjection.value.copy(this.camera.projectionMatrixInverse);
      this.uniforms.uCameraWorld.value.copy(this.camera.matrixWorld);
      this.uniforms.uCameraPos.value.copy(this.camera.position);
      this.uniforms.uFlowPhase.value = pv._flowPhase;
      this.uniforms.uWindMin.value.copy(pv.windBoxMin);
      this.uniforms.uWindMax.value.copy(pv.windBoxMax);
    }

    if (this.renderToScreen) {
      renderer.setRenderTarget(null);
    } else {
      renderer.setRenderTarget(writeBuffer);
    }
    this.fsQuad.render(renderer);
  }

  dispose() {
    this.fsQuad.dispose();
  }
}
