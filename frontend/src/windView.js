import * as THREE from "three";
import { simToThree } from "./coords.js";

/**
 * Builds a sleek merged arrow geometry (thin shaft + crisp cone head) pointing along +Z
 * so `Object3D.lookAt()` aligns it directly with any 3D velocity vector.
 */
function createSleekArrowGeometry() {
  const shaftLen = 1.35;
  const headLen = 0.55;
  const shaft = new THREE.CylinderGeometry(0.048, 0.036, shaftLen, 6);
  shaft.rotateX(Math.PI / 2);
  shaft.translate(0, 0, -headLen * 0.5);

  const head = new THREE.ConeGeometry(0.17, headLen, 8);
  head.rotateX(Math.PI / 2);
  head.translate(0, 0, shaftLen * 0.5);

  const g1 = shaft.toNonIndexed();
  const g2 = head.toNonIndexed();
  const p1 = g1.attributes.position.array;
  const p2 = g2.attributes.position.array;
  const merged = new Float32Array(p1.length + p2.length);
  merged.set(p1, 0);
  merged.set(p2, p1.length);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(merged, 3));
  geo.computeVertexNormals();
  shaft.dispose();
  head.dispose();
  g1.dispose();
  g2.dispose();
  return geo;
}

/**
 * CFD-grade 3D Wind Field Visualizer driven by the backend's 48x48x8 mass-consistent
 * 3D Poisson wind grid (`wind_field` payload).
 *
 * Designed to look crisp and readable from EVERY camera angle (top-down, iso, and low side):
 * 1) Camera-facing cylindrical 3D Streamline Tubes (billboarded in vertex shader along
 *    each curve's 3D tangent so they never go edge-on or twist into bowties) with smooth
 *    animated flow pulses and speed-ratio color coding (blue wakes -> cyan -> gold speedup).
 * 2) Subtle Ground-Level Wake & Speedup Contour Map (`48x48` DataTexture at y = 0.14 m)
 *    revealing lee-side wake cavities and Venturi acceleration corridors.
 * 3) Clean 3D Vector Arrows on a 15x15 slice at z = 4.5 m (pipe/leak height).
 */
export class WindView {
  constructor(scene) {
    this.scene = scene;
    this.visible = true;
    this.group = new THREE.Group();
    this.group.visible = this.visible;
    this.scene.add(this.group);

    this._field = null;
    this._solids = [];
    this._fallbackWind = [0, 0];
    this._renderedAmbientDirection = null;
    this._time = 0;
    this._animSpeed = 1.0;

    // 1) Camera-facing 3D Streamline Tubes (consistent silky thickness from every angle!)
    this._ribbonUniforms = {
      uTime: { value: 0.0 },
      uSpeedScale: { value: 1.0 },
    };
    this._ribbonMat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      uniforms: this._ribbonUniforms,
      vertexShader: `
        attribute vec3 aTangent;
        attribute float aArc;
        attribute float aArcFrac;
        attribute float aSpeedRatio;
        attribute float aActivity;
        attribute float aPhase;
        attribute float aEdge;
        attribute float aWidth;

        varying float vArc;
        varying float vArcFrac;
        varying float vSpeedRatio;
        varying float vActivity;
        varying float vPhase;
        varying float vEdge;

        void main() {
          vArc = aArc;
          vArcFrac = aArcFrac;
          vSpeedRatio = aSpeedRatio;
          vActivity = aActivity;
          vPhase = aPhase;
          vEdge = aEdge;

          // Billboard ribbon cross-section toward the camera so it looks cylindrical from ANY angle
          vec3 worldPos = (modelMatrix * vec4(position, 1.0)).xyz;
          vec3 viewDir = normalize(cameraPosition - worldPos);
          vec3 side = cross(aTangent, viewDir);
          float sLen = length(side);
          side = sLen > 1e-4 ? (side / sLen) : vec3(0.0, 1.0, 0.0);

          vec3 offsetPos = worldPos + side * (aEdge * aWidth);
          gl_Position = projectionMatrix * viewMatrix * vec4(offsetPos, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform float uSpeedScale;
        varying float vArc;
        varying float vArcFrac;
        varying float vSpeedRatio;
        varying float vActivity;
        varying float vPhase;
        varying float vEdge;

        vec3 speedColor(float r) {
          // Deep royal blue (wake/stagnation <0.45) -> vivid sky cyan (nominal ~1.0) -> warm gold (speedup >1.18)
          float c = clamp(r, 0.0, 1.42);
          if (c < 0.55) {
            return mix(vec3(0.16, 0.34, 0.96), vec3(0.14, 0.72, 0.98), c / 0.55);
          } else if (c < 1.05) {
            return mix(vec3(0.14, 0.72, 0.98), vec3(0.28, 0.94, 0.84), (c - 0.55) / 0.50);
          } else {
            return mix(vec3(0.28, 0.94, 0.84), vec3(1.00, 0.82, 0.24), clamp((c - 1.05) / 0.32, 0.0, 1.0));
          }
        }

        void main() {
          // Smooth cylindrical cross-section falloff
          float radial = max(0.0, 1.0 - vEdge * vEdge);
          float crossFade = pow(radial, 0.85);

          // Smooth fade at streamline start and end (zero blunt cut-offs)
          float endFade = smoothstep(0.0, 0.12, vArcFrac) * (1.0 - smoothstep(0.84, 1.0, vArcFrac));

          // Smooth strictly-downwind comet pulses (speed visibly reflects wind speed via uTime & glow)
          float flowFreq = 0.048;
          float wavePos = fract(vArc * flowFreq - uTime * 0.48 + vPhase);
          float tail = smoothstep(0.12, 0.82, wavePos) * (1.0 - smoothstep(0.84, 0.96, wavePos));
          float headGlow = smoothstep(0.74, 0.86, wavePos) * (1.0 - smoothstep(0.87, 0.95, wavePos));

          float energy = clamp(0.60 + 0.45 * uSpeedScale, 0.55, 1.45);
          vec3 baseCol = speedColor(vSpeedRatio);
          vec3 col = mix(baseCol * 0.85, baseCol * 1.15 + vec3(0.20, 0.28, 0.34), tail * 0.55 + headGlow * 0.65);

          float emphasis = mix(0.30, 0.85, smoothstep(0.04, 0.45, vActivity));
          float alpha = crossFade * endFade * (0.10 + 0.30 * tail + 0.18 * headGlow) * energy * emphasis;
          if (alpha < 0.012) discard;
          gl_FragColor = vec4(col, min(alpha, 0.92));
        }
      `,
    });
    this._ribbonMesh = null;

    // 2) Subtle Ground-Level Speed & Wake Contour Plane (at y = 0.14 m)
    this._contourMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this._contourPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120),
      this._contourMat
    );
    this._contourPlane.rotation.x = -Math.PI / 2;
    this._contourPlane.position.set(0, 0.14, 0);
    this._contourPlane.visible = false;
    this.group.add(this._contourPlane);

    // 3) Crisp 3D Vector Arrows on a 15x15 slice at z = 4.5 m
    this._sliceZ = 4.5;
    this._arrowGridN = 15;
    this._maxArrows = this._arrowGridN * this._arrowGridN;
    const arrowGeo = createSleekArrowGeometry();
    const arrowMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.52,
      depthWrite: false,
    });
    this._arrows = new THREE.InstancedMesh(arrowGeo, arrowMat, this._maxArrows);
    this._arrows.frustumCulled = false;
    this._arrows.count = 0;
    this.group.add(this._arrows);

    // Scratch helpers
    this._tmpA = new THREE.Vector3();
    this._tmpB = new THREE.Vector3();
    this._tmpColor = new THREE.Color();
    this._dummy = new THREE.Object3D();
  }

  setVisible(flag) {
    this.visible = Boolean(flag);
    this.group.visible = this.visible;
  }

  setWorld(world) {
    this._solids = (world.buildings || []).filter((b) => b.kind !== "pipe_rack" && b.kind !== "pipe_segment");
    if (this._field || Math.hypot(...this._fallbackWind) > 0.15) this._rebuildStreamlineRibbons();
  }

  _insideSolid(x, y, z) {
    const margin = 0.2;
    return this._solids.some((b) => {
      const lo = b.min_xyz, hi = b.max_xyz;
      if (z < lo[2] - margin || z > hi[2] + margin) return false;
      if (b.kind === "storage_tank") {
        const radius = Math.min(hi[0] - lo[0], hi[1] - lo[1]) / 2 + margin;
        return Math.hypot(x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2) <= radius;
      }
      return x >= lo[0] - margin && x <= hi[0] + margin && y >= lo[1] - margin && y <= hi[1] + margin;
    });
  }

  setAmbientWind(windVec) {
    if (!windVec) return;
    const [wx, wy] = windVec;
    const speed = Math.hypot(wx, wy);
    this._animSpeed = Math.max(0.15, speed / 3.5);
    this._ribbonUniforms.uSpeedScale.value = this._animSpeed;
    if (this._field) return;

    if (speed < 0.15) {
      this._fallbackWind = [0, 0];
      this._renderedAmbientDirection = null;
      if (this._ribbonMesh) this._ribbonMesh.visible = false;
      return;
    }

    const direction = [wx / speed, wy / speed];
    const previous = this._renderedAmbientDirection;
    // Small gusts affect animation speed without regenerating the geometry every tick.
    if (!previous || direction[0] * previous[0] + direction[1] * previous[1] < Math.cos(Math.PI / 12)) {
      this._fallbackWind = windVec;
      this._renderedAmbientDirection = direction;
      this._rebuildStreamlineRibbons();
    }
  }

  setWindField(payload) {
    if (!payload || !payload.vectors || !payload.dims) return;
    const [nx, ny, nz] = payload.dims;
    const [minX, minY, minZ] = payload.bounds_min;
    const [maxX, maxY, maxZ] = payload.bounds_max;

    const baseSpeed = Math.max(0.25, payload.base_speed || 3.5);
    this._animSpeed = Math.max(0.15, baseSpeed / 3.5);
    this._ribbonUniforms.uSpeedScale.value = this._animSpeed;

    this._field = {
      nx,
      ny,
      nz,
      minX,
      minY,
      minZ,
      maxX,
      maxY,
      maxZ,
      dx: (maxX - minX) / nx,
      dy: (maxY - minY) / ny,
      dz: (maxZ - minZ) / nz,
      baseSpeed,
      data: Float32Array.from(payload.vectors),
    };

    // Single clean 3D streamline visualization (no redundant ground heatmap or 225 arrow grid clutter)
    this._rebuildStreamlineRibbons();
    this._contourPlane.visible = false;
    this._arrows.count = 0;
  }

  _sampleField(x, y, z, out) {
    const f = this._field;
    if (!f) {
      out[0] = this._fallbackWind[0];
      out[1] = this._fallbackWind[1];
      out[2] = 0.0;
      return;
    }

    const gx = Math.max(0, Math.min(f.nx - 1.001, (x - f.minX) / f.dx - 0.5));
    const gy = Math.max(0, Math.min(f.ny - 1.001, (y - f.minY) / f.dy - 0.5));
    const gz = Math.max(0, Math.min(f.nz - 1.001, (z - f.minZ) / f.dz - 0.5));

    const i0 = Math.floor(gx);
    const j0 = Math.floor(gy);
    const k0 = Math.floor(gz);
    const i1 = i0 + 1;
    const j1 = j0 + 1;
    const k1 = k0 + 1;

    const fx = gx - i0;
    const fy = gy - j0;
    const fz = gz - k0;
    const omx = 1 - fx;
    const omy = 1 - fy;
    const omz = 1 - fz;

    const ny = f.ny;
    const nz = f.nz;
    const d = f.data;

    const idx = (i, j, k) => ((i * ny + j) * nz + k) * 3;
    const i000 = idx(i0, j0, k0);
    const i100 = idx(i1, j0, k0);
    const i010 = idx(i0, j1, k0);
    const i110 = idx(i1, j1, k0);
    const i001 = idx(i0, j0, k1);
    const i101 = idx(i1, j0, k1);
    const i011 = idx(i0, j1, k1);
    const i111 = idx(i1, j1, k1);

    for (let c = 0; c < 3; c++) {
      const c00 = d[i000 + c] * omx + d[i100 + c] * fx;
      const c10 = d[i010 + c] * omx + d[i110 + c] * fx;
      const c01 = d[i001 + c] * omx + d[i101 + c] * fx;
      const c11 = d[i011 + c] * omx + d[i111 + c] * fx;
      const c0 = c00 * omy + c10 * fy;
      const c1 = c01 * omy + c11 * fy;
      out[c] = c0 * omz + c1 * fz;
    }
  }

  _speedColor(speedRatio, targetColor) {
    const c = Math.max(0, Math.min(1.42, speedRatio));
    if (c < 0.55) {
      const t = c / 0.55;
      targetColor.setRGB(0.16 - 0.02 * t, 0.34 + 0.38 * t, 0.96 + 0.02 * t);
    } else if (c < 1.05) {
      const t = (c - 0.55) / 0.50;
      targetColor.setRGB(0.14 + 0.14 * t, 0.72 + 0.22 * t, 0.98 - 0.14 * t);
    } else {
      const t = Math.min(1, (c - 1.05) / 0.32);
      targetColor.setRGB(0.28 + 0.72 * t, 0.94 - 0.12 * t, 0.84 - 0.60 * t);
    }
  }

  _rebuildStreamlineRibbons() {
    if (this._ribbonMesh) {
      this.group.remove(this._ribbonMesh);
      this._ribbonMesh.geometry.dispose();
      this._ribbonMesh = null;
    }

    const maxZ = this._field?.maxZ ?? 30;
    const baseSpd = this._field?.baseSpeed ?? Math.hypot(...this._fallbackWind);

    // Dominant wind unit vector
    const vRef = [0, 0, 0];
    this._sampleField(-52, -52, 10.0, vRef);
    let refSpd = Math.hypot(vRef[0], vRef[1]);
    if (refSpd < 0.1) {
      vRef[0] = this._fallbackWind[0];
      vRef[1] = this._fallbackWind[1];
      refSpd = Math.max(0.1, Math.hypot(vRef[0], vRef[1]));
    }
    const ux = vRef[0] / refSpd;
    const uy = vRef[1] / refSpd;

    // Uniform full-area inflow seeding along all upwind domain edges for ANY wind angle (0..360 deg).
    // Spacing along each inflow edge is scaled by 1/|u_normal| so perpendicular spacing stays uniform
    // across the entire 120x120m pad with zero empty corners.
    const seeds = [];
    const crossSpacing = 12.0;
    const edgeCoord = 57.5;
    const spanMax = 55.0;

    for (const z of [4.5, 9.5]) {
      if (Math.abs(ux) > 0.08) {
        const sx = ux > 0 ? -edgeCoord : edgeCoord;
        const stepY = Math.min(40.0, crossSpacing / Math.abs(ux));
        for (let sy = -spanMax; sy <= spanMax + 1e-3; sy += stepY) {
          seeds.push([sx, sy, z]);
        }
      }
      if (Math.abs(uy) > 0.08) {
        const sy = uy > 0 ? -edgeCoord : edgeCoord;
        const stepX = Math.min(40.0, crossSpacing / Math.abs(uy));
        // Offset by 0.5 * stepX to avoid duplicating the shared upwind corner
        for (let sx = -spanMax + 0.5 * stepX; sx <= spanMax + 1e-3; sx += stepX) {
          seeds.push([sx, sy, z]);
        }
      }
    }

    // Distribute short traces throughout the site, with offset rows to avoid a rigid grid.
    for (const z of [4.5, 9.5]) {
      for (let x = -48; x <= 48; x += 18) {
        const offset = Math.round((x + 48) / 18) % 2 ? 9 : 0;
        for (let y = -48 + offset; y <= 48; y += 18) {
          seeds.push([x, y, z]);
        }
      }
    }

    const positions = [];
    const tangents = [];
    const arcs = [];
    const arcFracs = [];
    const speedRatios = [];
    const activities = [];
    const phases = [];
    const edges = [];
    const widths = [];
    const indices = [];

    const v1 = [0, 0, 0];
    const v2 = [0, 0, 0];
    const maxSteps = 42; // Keep each trace local rather than spanning the map.
    const ds = 0.85; // Fine spatial step for razor-smooth curves around round tanks & roofs
    const baseWidth = 0.11;

    let vertOffset = 0;

    for (let sIdx = 0; sIdx < seeds.length; sIdx++) {
      let [x, y, z] = seeds[sIdx];
      this._sampleField(x, y, z, v1);
      if (Math.hypot(v1[0], v1[1], v1[2]) < 0.15) continue;

      const pts = [];
      let arcLen = 0;
      let previousDirection = null;

      for (let step = 0; step < maxSteps; step++) {
        if (this._insideSolid(x, y, z)) break;
        this._sampleField(x, y, z, v1);
        const spd1 = Math.hypot(v1[0], v1[1], v1[2]);
        if (spd1 < Math.max(0.08, baseSpd * 0.12)) break;
        const direction = [v1[0] / spd1, v1[1] / spd1, v1[2] / spd1];
        if (previousDirection && direction[0] * previousDirection[0] + direction[1] * previousDirection[1] + direction[2] * previousDirection[2] < 0.7) break;
        previousDirection = direction;

        const directionChange = Math.hypot(v1[0] / spd1 - ux, v1[1] / spd1 - uy, v1[2] / spd1);
        const activity = Math.max(directionChange, Math.abs(spd1 / baseSpd - 1));
        pts.push([x, y, z, arcLen, spd1 / baseSpd, activity]);

        const inv1 = 1.0 / spd1;
        const mx = x + 0.5 * v1[0] * inv1 * ds;
        const my = y + 0.5 * v1[1] * inv1 * ds;
        const mz = Math.max(0.8, Math.min(maxZ - 0.6, z + 0.5 * v1[2] * inv1 * ds));

        if (this._insideSolid(mx, my, mz)) break;
        this._sampleField(mx, my, mz, v2);
        const spd2 = Math.hypot(v2[0], v2[1], v2[2]);
        if (spd2 < 0.08) break;

        const inv2 = 1.0 / spd2;
        x += v2[0] * inv2 * ds;
        y += v2[1] * inv2 * ds;
        z = Math.max(0.8, Math.min(maxZ - 0.6, z + v2[2] * inv2 * ds));
        arcLen += ds;

        if (Math.abs(x) > 59.0 || Math.abs(y) > 59.0) break;
      }

      if (pts.length < 8) continue;

      // Convert to Three.js world positions and apply 1-pass Laplacian smoothing for silky curves
      const threePts = pts.map((pt) => simToThree([pt[0], pt[1], pt[2]], new THREE.Vector3()));
      const smoothPts = threePts;

      const phase = ((sIdx * 0.6180339) % 1.0);
      const totalArc = Math.max(1e-3, pts[pts.length - 1][3]);
      const tangent = new THREE.Vector3();

      for (let k = 0; k < smoothPts.length; k++) {
        const pCur = smoothPts[k];
        const pPrev = smoothPts[Math.max(0, k - 1)];
        const pNext = smoothPts[Math.min(smoothPts.length - 1, k + 1)];
        tangent.subVectors(pNext, pPrev);
        if (tangent.lengthSq() < 1e-6) tangent.set(1, 0, 0);
        else tangent.normalize();

        const frac = pts[k][3] / totalArc;
        const speedR = pts[k][4];
        const w = baseWidth * (0.85 + 0.30 * Math.min(1.3, speedR));

        // Emit 2 vertices at same center position; vertex shader expands along camera-facing side
        positions.push(pCur.x, pCur.y, pCur.z, pCur.x, pCur.y, pCur.z);
        tangents.push(tangent.x, tangent.y, tangent.z, tangent.x, tangent.y, tangent.z);
        arcs.push(pts[k][3], pts[k][3]);
        arcFracs.push(frac, frac);
        speedRatios.push(speedR, speedR);
        activities.push(pts[k][5], pts[k][5]);
        phases.push(phase, phase);
        edges.push(-1.0, 1.0);
        widths.push(w, w);

        if (k < smoothPts.length - 1) {
          const i0 = vertOffset + k * 2;
          const i1 = i0 + 1;
          const i2 = i0 + 2;
          const i3 = i0 + 3;
          indices.push(i0, i2, i1, i1, i2, i3);
        }
      }
      vertOffset += smoothPts.length * 2;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("aTangent", new THREE.Float32BufferAttribute(tangents, 3));
    geo.setAttribute("aArc", new THREE.Float32BufferAttribute(arcs, 1));
    geo.setAttribute("aArcFrac", new THREE.Float32BufferAttribute(arcFracs, 1));
    geo.setAttribute("aSpeedRatio", new THREE.Float32BufferAttribute(speedRatios, 1));
    geo.setAttribute("aActivity", new THREE.Float32BufferAttribute(activities, 1));
    geo.setAttribute("aPhase", new THREE.Float32BufferAttribute(phases, 1));
    geo.setAttribute("aEdge", new THREE.Float32BufferAttribute(edges, 1));
    geo.setAttribute("aWidth", new THREE.Float32BufferAttribute(widths, 1));
    geo.setIndex(indices);

    this._ribbonMesh = new THREE.Mesh(geo, this._ribbonMat);
    this._ribbonMesh.frustumCulled = false;
    this.group.add(this._ribbonMesh);
  }

  _rebuildContourTexture() {
    if (!this._field) return;
    const f = this._field;
    const nx = f.nx;
    const ny = f.ny;
    const baseSpd = f.baseSpeed;
    const pixels = new Uint8Array(nx * ny * 4);
    const v = [0, 0, 0];
    const col = new THREE.Color();

    for (let j = 0; j < ny; j++) {
      const sy = f.minY + (j + 0.5) * f.dy;
      for (let i = 0; i < nx; i++) {
        const sx = f.minX + (i + 0.5) * f.dx;
        this._sampleField(sx, sy, this._sliceZ, v);
        const spd = Math.hypot(v[0], v[1], v[2]);
        const pIdx = (j * nx + i) * 4;

        if (spd < 0.05) {
          pixels[pIdx] = 0;
          pixels[pIdx + 1] = 0;
          pixels[pIdx + 2] = 0;
          pixels[pIdx + 3] = 0;
          continue;
        }

        const ratio = spd / baseSpd;
        if (ratio < 0.72) {
          const wakeIntensity = Math.min(1.0, (0.72 - ratio) / 0.58);
          col.setRGB(0.10, 0.24 + 0.16 * (1 - wakeIntensity), 0.90);
          pixels[pIdx] = Math.round(col.r * 255);
          pixels[pIdx + 1] = Math.round(col.g * 255);
          pixels[pIdx + 2] = Math.round(col.b * 255);
          pixels[pIdx + 3] = Math.round((0.12 + 0.62 * wakeIntensity) * 255);
        } else if (ratio > 1.08) {
          const boost = Math.min(1.0, (ratio - 1.08) / 0.28);
          col.setRGB(0.28 + 0.70 * boost, 0.90 - 0.08 * boost, 0.78 - 0.54 * boost);
          pixels[pIdx] = Math.round(col.r * 255);
          pixels[pIdx + 1] = Math.round(col.g * 255);
          pixels[pIdx + 2] = Math.round(col.b * 255);
          pixels[pIdx + 3] = Math.round((0.10 + 0.50 * boost) * 255);
        } else {
          pixels[pIdx] = 0;
          pixels[pIdx + 1] = 0;
          pixels[pIdx + 2] = 0;
          pixels[pIdx + 3] = 0;
        }
      }
    }

    if (this._contourMat.map) {
      this._contourMat.map.dispose();
    }
    const tex = new THREE.DataTexture(pixels, nx, ny, THREE.RGBAFormat);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    this._contourMat.map = tex;
    this._contourMat.needsUpdate = true;
    this._contourPlane.visible = true;
  }

  _rebuildArrowSlice() {
    if (!this._field) return;
    const n = this._arrowGridN;
    const span = 112;
    const start = -56;
    const step = span / (n - 1);
    const baseSpd = this._field.baseSpeed;
    const v = [0, 0, 0];

    let count = 0;
    for (let ix = 0; ix < n; ix++) {
      const sx = start + ix * step;
      for (let iy = 0; iy < n; iy++) {
        const sy = start + iy * step;
        this._sampleField(sx, sy, this._sliceZ, v);
        const spd = Math.hypot(v[0], v[1], v[2]);
        if (spd < 0.06) continue;

        simToThree([sx, sy, this._sliceZ], this._tmpA);
        simToThree([sx + v[0], sy + v[1], this._sliceZ + v[2]], this._tmpB);

        const ratio = spd / baseSpd;
        const lenScale = Math.max(0.45, Math.min(1.55, 0.45 + ratio * 0.80));
        this._dummy.position.copy(this._tmpA);
        this._dummy.lookAt(this._tmpB);
        this._dummy.scale.set(0.85, 0.85, lenScale);
        this._dummy.updateMatrix();
        this._arrows.setMatrixAt(count, this._dummy.matrix);

        this._speedColor(ratio, this._tmpColor);
        this._arrows.setColorAt(count, this._tmpColor);
        count++;
      }
    }

    this._arrows.count = count;
    this._arrows.instanceMatrix.needsUpdate = true;
    if (this._arrows.instanceColor) {
      this._arrows.instanceColor.needsUpdate = true;
    }
  }

  animate(dt) {
    if (!this.visible) return;
    this._time += dt * this._animSpeed;
    this._ribbonUniforms.uTime.value = this._time;
  }
}
