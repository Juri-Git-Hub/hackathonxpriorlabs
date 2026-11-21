import * as THREE from "three";
import { simToThree } from "./coords.js";

const BUILDING_COLORS = {
  building: 0x1e293b,
  storage_tank: 0x26354d,
  pipe_rack: 0x334155,
};

const ASSET_COLORS = {
  valve: 0xfacc15,
  flange: 0xfb923c,
  storage_tank: 0x38bdf8,
  pipe_junction: 0x94a3b8,
};

/**
 * Builds static refinery ground plane, grid, buildings, storage tanks, pipe racks,
 * and facility asset markers from WorldInitPayload.
 */
export class RefineryView {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.groundMesh = null;
  }

  build(worldInit) {
    // Clear previous geometry if re-initialized
    while (this.group.children.length > 0) {
      const child = this.group.children.pop();
      this.group.remove(child);
    }

    const [minX, minY] = worldInit.bounds_min;
    const [maxX, maxY] = worldInit.bounds_max;
    const width = maxX - minX;
    const depth = maxY - minY;

    // Ground plane for click-to-place-leak raycasting
    const groundGeo = new THREE.PlaneGeometry(width, depth);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.92,
      metalness: 0.08,
    });
    this.groundMesh = new THREE.Mesh(groundGeo, groundMat);
    this.groundMesh.rotation.x = -Math.PI / 2;
    this.groundMesh.position.set(0, 0, 0);
    this.groundMesh.receiveShadow = true;
    this.group.add(this.groundMesh);

    // Subtle high-contrast grid
    const gridHelper = new THREE.GridHelper(width, 24, 0x334155, 0x1e293b);
    gridHelper.position.y = 0.03;
    this.group.add(gridHelper);

    // Perimeter border line
    const borderPts = [
      new THREE.Vector3(minX, 0.08, -minY),
      new THREE.Vector3(maxX, 0.08, -minY),
      new THREE.Vector3(maxX, 0.08, -maxY),
      new THREE.Vector3(minX, 0.08, -maxY),
      new THREE.Vector3(minX, 0.08, -minY),
    ];
    const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPts);
    const borderMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, opacity: 0.35, transparent: true });
    this.group.add(new THREE.Line(borderGeo, borderMat));

    // Buildings, storage tanks, pipe racks
    for (const b of worldInit.buildings) {
      const sx = b.max_xyz[0] - b.min_xyz[0];
      const sy = b.max_xyz[1] - b.min_xyz[1];
      const sz = b.max_xyz[2] - b.min_xyz[2];

      const cx = 0.5 * (b.min_xyz[0] + b.max_xyz[0]);
      const cy = 0.5 * (b.min_xyz[1] + b.max_xyz[1]);
      const cz = 0.5 * (b.min_xyz[2] + b.max_xyz[2]);

      const color = BUILDING_COLORS[b.kind] ?? 0x1e293b;
      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.72,
        metalness: 0.25,
        transparent: b.kind === "pipe_rack",
        opacity: b.kind === "pipe_rack" ? 0.78 : 0.94,
      });

      let geo;
      if (b.kind === "storage_tank") {
        const radius = 0.5 * Math.min(sx, sy);
        geo = new THREE.CylinderGeometry(radius, radius, sz, 28);
      } else {
        geo = new THREE.BoxGeometry(sx, sz, sy);
      }

      const mesh = new THREE.Mesh(geo, mat);
      simToThree([cx, cy, cz], mesh.position);
      this.group.add(mesh);

      // Crisp wireframe outline for dark-mode projector visibility
      const edges = new THREE.EdgesGeometry(geo);
      const edgeMat = new THREE.LineBasicMaterial({
        color: b.kind === "pipe_rack" ? 0x64748b : 0x475569,
        transparent: true,
        opacity: 0.75,
      });
      const wire = new THREE.LineSegments(edges, edgeMat);
      wire.position.copy(mesh.position);
      this.group.add(wire);
    }

    // Facility assets (valves, flanges, junctions)
    const assetGeo = new THREE.OctahedronGeometry(0.65, 0);
    for (const asset of worldInit.assets) {
      const mat = new THREE.MeshBasicMaterial({
        color: ASSET_COLORS[asset.asset_type] ?? 0xfacc15,
        transparent: true,
        opacity: 0.85,
      });
      const marker = new THREE.Mesh(assetGeo, mat);
      simToThree(asset.pos, marker.position);
      this.group.add(marker);
    }
  }
}
