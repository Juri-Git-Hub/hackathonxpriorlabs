import * as THREE from "three";
import { simToThree } from "./coords.js";

function createEstimateLabel(text = "TABPFN EST. SOURCE") {
  const canvas = document.createElement("canvas");
  canvas.width = 280;
  canvas.height = 52;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "rgba(8, 13, 22, 0.92)";
  ctx.beginPath();
  ctx.roundRect(6, 6, 268, 40, 8);
  ctx.fill();
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  ctx.fillStyle = "#67e8f9";
  ctx.font = "800 15.5px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 140, 27);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/**
 * Renders TabPFN's spatial belief as a crisp, non-saturated scientific GIS contour
 * heatmap (2D scalar field evaluated on a 64x64 grid with explicit iso-probability
 * contour bands & lines — zero white washout, completely distinct from the amber plume!)
 * + uncertainty-labeled `TABPFN EST. SOURCE` laser pin + dashed UCB decision vector.
 */
export class BeliefView {
  constructor(scene) {
    this.scene = scene;
    this.visible = true;
    this.layer="source";this.routeVisible=true;this.lastBelief=null;
    this.routeLine=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x22d3ee}));
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.group.add(this.routeLine);

    this.gridCoords = [];
    this.gridCount = 0;

    // 1. Scientific Iso-Contour Ground Heatmap (64x64 fast scalar grid, GPU-interpolated)
    this.res = 64;
    this.heatCanvas = document.createElement("canvas");
    this.heatCanvas.width = this.res;
    this.heatCanvas.height = this.res;
    this.heatCtx = this.heatCanvas.getContext("2d");
    this.heatImg = this.heatCtx.createImageData(this.res, this.res);

    this.heatTex = new THREE.CanvasTexture(this.heatCanvas);
    this.heatTex.minFilter = THREE.LinearFilter;
    this.heatTex.magFilter = THREE.LinearFilter;

    this.heatPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(120, 120),
      new THREE.MeshBasicMaterial({
        map: this.heatTex,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      })
    );
    this.heatPlane.rotation.x = -Math.PI / 2;
    this.heatPlane.position.y = 0.065;
    this.group.add(this.heatPlane);

    // 2. Estimated Source Laser Pin + Ground Crosshair + Dynamic Uncertainty Label
    this.estGroup = new THREE.Group();
    this.estGroup.visible = false;

    this.estRing = new THREE.Mesh(
      new THREE.RingGeometry(1.6, 2.2, 32),
      new THREE.MeshBasicMaterial({
        color: 0x22d3ee,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
      })
    );
    this.estRing.rotation.x = -Math.PI / 2;
    this.estRing.position.y = 0.12;
    this.estGroup.add(this.estRing);

    this.estDiamond = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.95, 0),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true })
    );
    this.estGroup.add(this.estDiamond);

    this.estBeam = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.16, 15, 12, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0x22d3ee,
        transparent: true,
        opacity: 0.50,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.estBeam.position.y = 7.5;
    this.estGroup.add(this.estBeam);

    this.estLabel = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: createEstimateLabel("TABPFN EST. SOURCE"),
        transparent: true,
        depthTest: false,
      })
    );
    this.estLabel.scale.set(8.0, 1.5, 1);
    this.estLabel.position.set(0, 16.5, 0);
    this.estGroup.add(this.estLabel);
    this.group.add(this.estGroup);

    // 3. Next UCB waypoint target ring + dashed decision vector from TabPFN drone
    this.wpBeacon = new THREE.Mesh(
      new THREE.TorusGeometry(1.15, 0.13, 10, 28),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    this.wpBeacon.rotation.x = Math.PI / 2;
    this.wpBeacon.visible = false;
    this.group.add(this.wpBeacon);

    const vecGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(1, 0, 0),
    ]);
    const vecMat = new THREE.LineDashedMaterial({
      color: 0x22d3ee,
      dashSize: 1.5,
      gapSize: 0.9,
      transparent: true,
      opacity: 0.80,
    });
    this.decisionLine = new THREE.Line(vecGeo, vecMat);
    this.decisionLine.visible = false;
    this.decisionLine.frustumCulled = false;
    this.group.add(this.decisionLine);

    this._tmpVec = new THREE.Vector3();
    this._dronePos = new THREE.Vector3();
    this._hasDronePos = false;
  }

  setLayer(layer) { this.layer=layer;if(this.lastBelief)this.update(this.lastBelief); }
  setRouteVisible(flag) { this.routeVisible=flag;this.routeLine.visible=flag; }
  setRoute(route) {
    this.routeLine.geometry.dispose();
    this.routeLine.geometry=new THREE.BufferGeometry().setFromPoints((route || []).map(p => simToThree(p,new THREE.Vector3())));
    this.routeLine.visible=this.routeVisible;
  }

  setVisible(flag) {
    this.visible = Boolean(flag);
    this.group.visible = this.visible;
  }

  setTabpfnDroneWorldPos(pos) {
    if (!pos) return;
    this._dronePos.copy(pos);
    this._hasDronePos = true;
    if (this.wpBeacon.visible) {
      const attr = this.decisionLine.geometry.attributes.position;
      attr.setXYZ(0, this._dronePos.x, this._dronePos.y, this._dronePos.z);
      attr.setXYZ(1, this.wpBeacon.position.x, this.wpBeacon.position.y, this.wpBeacon.position.z);
      attr.needsUpdate = true;
      this.decisionLine.computeLineDistances();
      this.decisionLine.visible = true;
    }
  }

  initGrid(gridCoords) {
    this.gridCoords = gridCoords;
    this.lastBelief = null;
    this.routeLine.geometry.dispose();
    this.routeLine.geometry = new THREE.BufferGeometry();
    this.gridCount = gridCoords.length;
    this.heatCtx.clearRect(0, 0, this.res, this.res);
    this.heatTex.needsUpdate = true;
    this.estGroup.visible = false;
    this.wpBeacon.visible = false;
    this.decisionLine.visible = false;
  }

  update(belief) {
    if (belief.q50.length !== this.gridCount || this.gridCount === 0) return;

    this.lastBelief = belief;
    const hazard = this.layer === "hazard";
    const q50 = hazard
      ? (belief.hazard_probabilities?.length === this.gridCount ? belief.hazard_probabilities : new Array(this.gridCount).fill(0))
      : (belief.source_scores?.length === this.gridCount ? belief.source_scores : belief.q50);
    const unc = belief.uncertainty;

    // 1. Reduce 3D candidate grid to 2D max-over-Z points for fast crisp contour interpolation
    const byXY = new Map();
    let minQ = hazard ? 0 : Infinity;
    let maxQ = hazard ? 1 : 0;
    let sumUnc = 0.0;
    for (let i = 0; i < this.gridCount; i++) {
      const val = q50[i];
      if (val < minQ) minQ = val;
      if (val > maxQ) maxQ = val;
      sumUnc += unc[i];

      const [sx, sy] = this.gridCoords[i];
      const key = `${Math.round(sx)},${Math.round(sy)}`;
      const prev = byXY.get(key);
      if (!prev || val > prev.q) {
        byXY.set(key, { x: sx, y: sy, q: val, u: unc[i] });
      }
    }
    const meanUnc = sumUnc / this.gridCount;
    const span = Math.max(0.22, maxQ - minQ);
    const pts2d = Array.from(byXY.values());

    // 2. Evaluate smooth polynomial-kernel scalar field on 64x64 grid (<0.4ms total!)
    const R = this.res;
    const data = this.heatImg.data;
    const rMax2 = 484.0; // 22m cutoff squared
    const invRMax2 = 1.0 / rMax2;

    for (let py = 0; py < R; py++) {
      const simY = 60.0 - (py / (R - 1)) * 120.0;
      for (let px = 0; px < R; px++) {
        const simX = -60.0 + (px / (R - 1)) * 120.0;

        let wSum = 0.0;
        let qSum = 0.0;
        for (let k = 0; k < pts2d.length; k++) {
          const p = pts2d[k];
          const dx = simX - p.x;
          const dy = simY - p.y;
          const d2 = dx * dx + dy * dy;
          if (d2 >= rMax2) continue;
          const u = 1.0 - d2 * invRMax2;
          const w = u * u * u;
          wSum += w;
          qSum += w * p.q;
        }

        const idx = (py * R + px) * 4;
        if (wSum < 1e-4) {
          data[idx + 3] = 0;
          continue;
        }

        const qInterp = qSum / wSum;
        const norm = Math.min(1.0, Math.max(0.0, (qInterp - minQ) / span));

        // Smooth edge vignette so heatmap never clips harshly at the [-60, 60] fence boundary
        const edgeDist = Math.min(60.0 - Math.abs(simX), 60.0 - Math.abs(simY));
        const edgeFade = Math.min(1.0, Math.max(0.0, edgeDist / 10.0));

        // Keep background tarmac 100% unobstructed; draw crisp topographic iso-contour rings
        if (norm < 0.50 || edgeFade < 0.02) {
          data[idx + 3] = 0;
          continue;
        }

        const t = (norm - 0.50) / 0.50;
        const contourDist = Math.min(
          Math.abs(t - 0.18),
          Math.abs(t - 0.50),
          Math.abs(t - 0.80)
        );
        const onContour = contourDist < 0.036;

        // Ultra-clean aerospace radar contour lines with minimal subtle interior tint
        let rC, gC, bC, aC;
        if (onContour) {
          rC = t > 0.65 ? 232 : 34;
          gC = t > 0.65 ? 121 : 211;
          bC = 249;
          aC = Math.round(235 * edgeFade);
        } else if (t < 0.5) {
          const s = t * 2.0;
          rC = Math.round(6 + 18 * s);
          gC = Math.round(150 + 55 * s);
          bC = Math.round(205 + 35 * s);
          aC = Math.round((8 + 16 * s) * edgeFade);
        } else {
          const s = (t - 0.5) * 2.0;
          rC = Math.round(24 + 175 * s);
          gC = Math.round(205 - 90 * s);
          bC = Math.round(240 + 10 * s);
          aC = Math.round((24 + 28 * s) * edgeFade);
        }

        data[idx] = rC;
        data[idx + 1] = gC;
        data[idx + 2] = bC;
        data[idx + 3] = aC;
      }
    }

    this.heatCtx.putImageData(this.heatImg, 0, 0);

    this.setRoute(belief.route);
    // Draw crisp dashed uncertainty radius around estimated source
    const ctx = this.heatCtx;
    const [ex, ey] = belief.estimated_source;
    const eu = ((ex + 60) / 120) * R;
    const ev = ((60 - ey) / 120) * R;
    const uncR = Math.max(7, Math.min(20, meanUnc * 10));

    ctx.strokeStyle = "rgba(103, 232, 249, 0.95)";
    ctx.lineWidth = 1.6;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(eu, ev, uncR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    this.heatTex.needsUpdate = true;

    // Update label with live uncertainty metric
    const oldTex = this.estLabel.material.map;
    this.estLabel.material.map = createEstimateLabel(
      `TABPFN EST. (±${(meanUnc * 4.5).toFixed(1)}m)`
    );
    if (oldTex) oldTex.dispose();

    simToThree(belief.estimated_source, this._tmpVec);
    this.estGroup.position.set(this._tmpVec.x, 0.08, this._tmpVec.z);
    this.estDiamond.position.y = Math.max(1.8, this._tmpVec.y);
    this.estGroup.visible = belief.source_evidence !== false;

    simToThree(belief.next_waypoint, this.wpBeacon.position);
    this.wpBeacon.visible = true;
    if (this._hasDronePos) {
      this.setTabpfnDroneWorldPos(this._dronePos);
    }
  }

  animate(elapsedSec) {
    if (this.estGroup.visible) {
      this.estDiamond.rotation.y = elapsedSec * 2.0;
      const s = 1.0 + 0.14 * Math.sin(elapsedSec * 3.5);
      this.estRing.scale.setScalar(s);
    }
    if (this.wpBeacon.visible) {
      const s = 1.0 + 0.14 * Math.cos(elapsedSec * 4.5);
      this.wpBeacon.scale.setScalar(s);
    }
  }
}
