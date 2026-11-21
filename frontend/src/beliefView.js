import * as THREE from "three";
import { simToThree } from "./coords.js";

function createCircleTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const grad = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
  grad.addColorStop(0.0, "rgba(255,255,255,1)");
  grad.addColorStop(0.5, "rgba(255,255,255,0.65)");
  grad.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/**
 * Renders TabPFN 3D belief cloud over `grid_coords` (q50 -> cyan->magenta->red,
 * uncertainty -> brightness/alpha) + estimated source & next waypoint beacons.
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

    this.points = null;
    this.gridCount = 0;

    // Estimated source beacon (magenta diamond)
    this.estBeacon = new THREE.Mesh(
      new THREE.OctahedronGeometry(1.25, 0),
      new THREE.MeshBasicMaterial({
        color: 0xf43f5e,
        wireframe: true,
      })
    );
    this.estBeacon.visible = false;
    this.group.add(this.estBeacon);

    // Next waypoint beacon (cyan ring)
    this.wpBeacon = new THREE.Mesh(
      new THREE.TorusGeometry(1.1, 0.14, 10, 24),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee })
    );
    this.wpBeacon.rotation.x = Math.PI / 2;
    this.wpBeacon.visible = false;
    this.group.add(this.wpBeacon);

    this._tmpColor = new THREE.Color();
  }

  setLayer(layer) { this.layer=layer;if(this.lastBelief)this.update(this.lastBelief); }
  setRouteVisible(flag) { this.routeVisible=flag;this.routeLine.visible=flag; }

  setVisible(flag) {
    this.visible = Boolean(flag);
    this.group.visible = this.visible;
  }

  initGrid(gridCoords) {
    if (this.points) {
      this.group.remove(this.points);
      this.points.geometry.dispose();
    }

    this.lastBelief=null;this.estBeacon.visible=false;this.wpBeacon.visible=false;
    this.routeLine.geometry.dispose();this.routeLine.geometry=new THREE.BufferGeometry();
    this.gridCount = gridCoords.length;
    const positions = new Float32Array(this.gridCount * 3);
    const colors = new Float32Array(this.gridCount * 3);
    const tmp = new THREE.Vector3();

    for (let i = 0; i < this.gridCount; i++) {
      simToThree(gridCoords[i], tmp);
      positions[i * 3] = tmp.x;
      positions[i * 3 + 1] = tmp.y;
      positions[i * 3 + 2] = tmp.z;

      colors[i * 3] = 0.08;
      colors[i * 3 + 1] = 0.25;
      colors[i * 3 + 2] = 0.35;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 3.2,
      map: createCircleTexture(),
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    this.points = new THREE.Points(geo, mat);
    this.group.add(this.points);
  }

  update(belief) {
    if (!this.points || belief.q50.length !== this.gridCount) return;

    const colors = this.points.geometry.attributes.color.array;
    this.lastBelief=belief;
    const q50 = this.layer === "hazard" ? (belief.hazard_probabilities?.length === this.gridCount ? belief.hazard_probabilities : new Array(this.gridCount).fill(0)) : (belief.source_scores?.length === this.gridCount ? belief.source_scores : belief.q50);
    const unc = belief.uncertainty;

    let maxQ = this.layer === "hazard" ? 1 : 1e-9;
    for (let i = 0; i < q50.length; i++) {
      if (q50[i] > maxQ) maxQ = q50[i];
    }

    for (let i = 0; i < this.gridCount; i++) {
      const normQ = Math.min(1.0, Math.max(0.0, q50[i] / maxQ));
      const normU = Math.min(1.0, Math.max(0.0, unc[i] / 2.0));

      // Color ramp: cyan (low) -> magenta (mid) -> bright red (high)
      if (normQ < 0.5) {
        const t = normQ * 2.0;
        this._tmpColor.setRGB(0.08 + 0.82 * t, 0.78 - 0.55 * t, 0.92 - 0.05 * t);
      } else {
        const t = (normQ - 0.5) * 2.0;
        this._tmpColor.setRGB(0.9 + 0.1 * t, 0.23 - 0.18 * t, 0.87 - 0.72 * t);
      }

      // Scale brightness by concentration + uncertainty so hotspots pop
      const brightness = 0.18 + 0.55 * normQ + 0.27 * normU;
      colors[i * 3] = this._tmpColor.r * brightness;
      colors[i * 3 + 1] = this._tmpColor.g * brightness;
      colors[i * 3 + 2] = this._tmpColor.b * brightness;
    }

    this.points.geometry.attributes.color.needsUpdate = true;

    this.routeLine.geometry.dispose();
    this.routeLine.geometry=new THREE.BufferGeometry().setFromPoints((belief.route || []).map(p => simToThree(p,new THREE.Vector3())));
    this.routeLine.visible=this.routeVisible;
    simToThree(belief.estimated_source, this.estBeacon.position);
    this.estBeacon.visible = true;

    simToThree(belief.next_waypoint, this.wpBeacon.position);
    this.wpBeacon.visible = true;
  }

  animate(elapsedSec) {
    if (this.estBeacon.visible) {
      this.estBeacon.rotation.y = elapsedSec * 1.8;
      const s = 1.0 + 0.18 * Math.sin(elapsedSec * 4.0);
      this.estBeacon.scale.setScalar(s);
    }
    if (this.wpBeacon.visible) {
      const s = 1.0 + 0.15 * Math.cos(elapsedSec * 5.0);
      this.wpBeacon.scale.setScalar(s);
    }
  }
}
