import * as THREE from "three";
import { simToThree } from "./coords.js";

/**
 * Renders TabPFN 3D belief cloud as a clean volumetric hologram (using per-point
 * size & alpha shaders so low-concentration background points fade out instead of
 * cluttering the refinery like snow) + pulsing estimated source & waypoint beacons.
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

    // Estimated source beacon (pulsing crimson diamond)
    this.estBeacon = new THREE.Mesh(
      new THREE.OctahedronGeometry(1.35, 0),
      new THREE.MeshBasicMaterial({
        color: 0xf43f5e,
        wireframe: true,
      })
    );
    this.estBeacon.visible = false;
    this.group.add(this.estBeacon);

    // Next waypoint beacon (cyan holographic ring)
    this.wpBeacon = new THREE.Mesh(
      new THREE.TorusGeometry(1.25, 0.14, 10, 24),
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
    const sizes = new Float32Array(this.gridCount);
    const alphas = new Float32Array(this.gridCount);
    const tmp = new THREE.Vector3();

    for (let i = 0; i < this.gridCount; i++) {
      simToThree(gridCoords[i], tmp);
      positions[i * 3] = tmp.x;
      positions[i * 3 + 1] = tmp.y;
      positions[i * 3 + 2] = tmp.z;

      colors[i * 3] = 0.1;
      colors[i * 3 + 1] = 0.7;
      colors[i * 3 + 2] = 0.9;
      sizes[i] = 1.0;
      alphas[i] = 0.0;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
    geo.setAttribute("aAlpha", new THREE.BufferAttribute(alphas, 1));

    const mat = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexShader: `
        attribute float aSize;
        attribute float aAlpha;
        attribute vec3 color;
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          vColor = color;
          vAlpha = aAlpha;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * (280.0 / max(-mvPos.z, 1.0));
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          if (vAlpha < 0.01) discard;
          vec2 c = gl_PointCoord - vec2(0.5);
          float r = length(c) * 2.0;
          if (r > 1.0) discard;
          float falloff = pow(1.0 - r, 2.0);
          gl_FragColor = vec4(vColor, vAlpha * falloff);
        }
      `,
    });

    this.points = new THREE.Points(geo, mat);
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }

  update(belief) {
    if (!this.points || belief.q50.length !== this.gridCount) return;

    const colors = this.points.geometry.attributes.color.array;
    this.lastBelief=belief;
    const sizes = this.points.geometry.attributes.aSize.array;
    const alphas = this.points.geometry.attributes.aAlpha.array;
    const q50 = this.layer === "hazard" ? (belief.hazard_probabilities?.length === this.gridCount ? belief.hazard_probabilities : new Array(this.gridCount).fill(0)) : (belief.source_scores?.length === this.gridCount ? belief.source_scores : belief.q50);
    const unc = belief.uncertainty;
    let minQ = this.layer === "hazard" ? 0 : Infinity;
    let maxQ = this.layer === "hazard" ? 1 : 1e-9;
    for (let i = 0; i < q50.length; i++) {
      if (q50[i] < minQ) minQ = q50[i];
      if (q50[i] > maxQ) maxQ = q50[i];
    }
    const span = Math.max(1e-9, maxQ - minQ);

    for (let i = 0; i < this.gridCount; i++) {
      const normQ = Math.min(1.0, Math.max(0.0, (q50[i] - minQ) / span));
      const normU = Math.min(1.0, Math.max(0.0, unc[i] / 2.0));

      // Color ramp: cyan -> magenta -> hot crimson
      if (normQ < 0.5) {
        const t = normQ * 2.0;
        this._tmpColor.setRGB(0.06 + 0.84 * t, 0.82 - 0.56 * t, 0.96 - 0.08 * t);
      } else {
        const t = (normQ - 0.5) * 2.0;
        this._tmpColor.setRGB(0.90 + 0.10 * t, 0.26 - 0.18 * t, 0.88 - 0.70 * t);
      }

      colors[i * 3] = this._tmpColor.r;
      colors[i * 3 + 1] = this._tmpColor.g;
      colors[i * 3 + 2] = this._tmpColor.b;

      // Only show meaningful probability/interest hotspots; keep background clean
      if (normQ < 0.22) {
        sizes[i] = 1.4;
        alphas[i] = 0.04 * normU;
      } else {
        const emphasis = (normQ - 0.22) / 0.78;
        sizes[i] = 3.5 + 8.5 * emphasis + 2.0 * normU;
        alphas[i] = 0.14 + 0.62 * emphasis;
      }
    }

    this.points.geometry.attributes.color.needsUpdate = true;
    this.points.geometry.attributes.aSize.needsUpdate = true;
    this.points.geometry.attributes.aAlpha.needsUpdate = true;

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
