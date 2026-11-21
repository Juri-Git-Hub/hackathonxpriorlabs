import * as THREE from "three";
import { simToThree } from "./coords.js";

function createConcreteTextures() {
  const size = 512;
  const albedoCanvas = document.createElement("canvas");
  albedoCanvas.width = size;
  albedoCanvas.height = size;
  const ctx = albedoCanvas.getContext("2d");

  ctx.fillStyle = "#2b313a";
  ctx.fillRect(0, 0, size, size);

  // Concrete aggregate grain
  for (let i = 0; i < 7500; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const v = 34 + Math.floor(Math.random() * 26);
    ctx.fillStyle = `rgba(${v}, ${v + 4}, ${v + 8}, 0.30)`;
    ctx.fillRect(x, y, 2 + Math.random() * 2, 2 + Math.random() * 2);
  }
  // Weathered industrial oil/water stains
  for (let i = 0; i < 20; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 18 + Math.random() * 48;
    const g = ctx.createRadialGradient(x, y, 2, x, y, r);
    g.addColorStop(0, "rgba(15, 19, 24, 0.28)");
    g.addColorStop(1, "rgba(15, 19, 24, 0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Expansion joint seams
  ctx.strokeStyle = "rgba(12, 16, 22, 0.58)";
  ctx.lineWidth = 2;
  const step = size / 4;
  for (let p = 0; p <= size; p += step) {
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(size, p);
    ctx.stroke();
  }

  const albedo = new THREE.CanvasTexture(albedoCanvas);
  albedo.wrapS = THREE.RepeatWrapping;
  albedo.wrapT = THREE.RepeatWrapping;
  albedo.repeat.set(6, 6);
  albedo.colorSpace = THREE.SRGBColorSpace;
  albedo.anisotropy = 8;

  return { albedo };
}

/**
 * Builds realistic industrial refinery terrain, concrete pad, access roads,
 * grounded valve stands, Blender GLB structures, and distant refinery skyline.
 */
export class RefineryView {
  constructor(scene, models) {
    this.scene = scene;
    this.models = models;
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.groundMesh = null;
    this.interactiveObjects = [];
  }

  _tagInteractive(obj, meta) {
    obj.traverse((child) => {
      if (child.isMesh) {
        child.userData.inspect = meta;
        this.interactiveObjects.push(child);
      }
    });
  }

  getInteractiveObjects() {
    return this.interactiveObjects;
  }

  _addPipeRun(pointsSim, radius, mat, flangeMat, supportMat, pipeMeta) {
    const pts = pointsSim.map((p) => simToThree(p));
    const up = new THREE.Vector3(0, 1, 0);
    const runGroup = new THREE.Group();

    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      if (len < 0.05) continue;

      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      const segGeo = new THREE.CylinderGeometry(radius, radius, len, 12);
      const seg = new THREE.Mesh(segGeo, mat);
      seg.position.copy(mid);
      seg.quaternion.setFromUnitVectors(up, dir.clone().normalize());
      seg.castShadow = true;
      seg.receiveShadow = true;
      runGroup.add(seg);

      // Add bolted flange collars every ~7m along long horizontal runs
      if (len > 4.5) {
        const steps = Math.max(1, Math.floor(len / 7.5));
        const flangeGeo = new THREE.CylinderGeometry(radius * 1.42, radius * 1.42, 0.16, 12);
        for (let k = 1; k <= steps; k++) {
          const t = k / (steps + 1);
          const fPos = new THREE.Vector3().lerpVectors(a, b, t);
          const fl = new THREE.Mesh(flangeGeo, flangeMat);
          fl.position.copy(fPos);
          fl.quaternion.copy(seg.quaternion);
          fl.castShadow = true;
          runGroup.add(fl);
        }
      }

      // Add steel H-post pipe support stanchion at midpoint for elevated horizontal segments
      const isHoriz = Math.abs(dir.y) < 0.35 && len > 3.5;
      if (isHoriz && mid.y > 0.8) {
        const stH = mid.y - radius;
        if (stH > 0.2) {
          const st = new THREE.Mesh(
            new THREE.CylinderGeometry(0.11, 0.13, stH, 8),
            supportMat
          );
          st.position.set(mid.x, stH * 0.5, mid.z);
          st.castShadow = true;
          this.group.add(st);
        }
      }
    }

    // Smooth elbow joints at every bend
    const elbowGeo = new THREE.SphereGeometry(radius * 1.12, 12, 10);
    for (let i = 1; i < pts.length - 1; i++) {
      const elbow = new THREE.Mesh(elbowGeo, mat);
      elbow.position.copy(pts[i]);
      elbow.castShadow = true;
      runGroup.add(elbow);
    }

    this.group.add(runGroup);
    if (pipeMeta) {
      this._tagInteractive(runGroup, { kind: "pipe", radius, ...pipeMeta });
    }
  }

  _buildPipingNetwork() {
    const matYellowGas = new THREE.MeshStandardMaterial({
      color: 0xeab308,
      metalness: 0.32,
      roughness: 0.34,
    });
    const matSteel = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.84,
      roughness: 0.26,
    });
    const matBlueCoolant = new THREE.MeshStandardMaterial({
      color: 0x2563eb,
      metalness: 0.42,
      roughness: 0.34,
    });
    const matFlange = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.82,
      roughness: 0.32,
    });
    const matSupport = new THREE.MeshStandardMaterial({
      color: 0x2b3440,
      metalness: 0.72,
      roughness: 0.45,
    });

    // 1. Main yellow gas line: Control Bldg -> valve_01 -> Pipe Rack -> valve_02 -> Tank East -> valve_03
    this._addPipeRun(
      [
        [-20.0, -14.0, 3.5],
        [-17.5, -14.0, 3.5],
        [-17.5, -2.0, 3.5],
        [-14.0, -2.0, 4.4],
        [14.0, -2.0, 4.4],
        [15.5, -1.0, 4.5],
        [20.5, -1.0, 4.5],
        [20.5, -7.0, 6.0],
        [23.0, -7.0, 6.0],
      ],
      0.34,
      matYellowGas,
      matFlange,
      matSupport,
      { id: "line_gas_main", name: "Main Methane Header", service: "High-Pressure Gas", pressureBar: 18.4 }
    );

    // 2. Tank East south discharge line -> valve_03
    this._addPipeRun(
      [
        [31.0, -15.0, 3.0],
        [31.0, -17.5, 3.0],
        [14.0, -17.5, 3.0],
        [4.0, -17.5, 3.0],
        [4.0, -7.2, 5.0],
        [4.0, -3.0, 5.0],
      ],
      0.28,
      matSteel,
      matFlange,
      matSupport,
      { id: "line_east_transfer", name: "East Tank Transfer Line", service: "Process Feed", pressureBar: 11.2 }
    );

    // 3. Central Pipe Rack -> Tank North (dual silver + yellow lines via flange_02 & tank_port_n1)
    this._addPipeRun(
      [
        [-10.0, 3.0, 4.0],
        [-10.0, 5.5, 4.0],
        [0.0, 5.5, 4.0],
        [0.0, 12.5, 5.0],
        [0.0, 15.0, 5.0],
      ],
      0.32,
      matYellowGas,
      matFlange,
      matSupport,
      { id: "line_north_gas", name: "North Tank Gas Feed", service: "High-Pressure Gas", pressureBar: 16.0 }
    );
    this._addPipeRun(
      [
        [8.0, 3.0, 6.0],
        [8.0, 5.5, 6.0],
        [11.5, 5.5, 6.0],
        [11.5, 24.0, 8.0],
        [9.0, 24.0, 8.0],
      ],
      0.30,
      matBlueCoolant,
      matFlange,
      matSupport,
      { id: "line_north_coolant", name: "North Cooling Return", service: "Coolant Loop", pressureBar: 6.5 }
    );

    // 4. Compressor Building -> Tank North & Pipe Rack (via flange_04 & pipe_junc_03)
    this._addPipeRun(
      [
        [-19.0, 24.0, 4.5],
        [-16.5, 24.0, 4.5],
        [-9.0, 24.0, 4.5],
      ],
      0.28,
      matSteel,
      matFlange,
      matSupport,
      { id: "line_comp_discharge", name: "Compressor Discharge", service: "Compressed Vapor", pressureBar: 22.0 }
    );
    this._addPipeRun(
      [
        [-26.0, 17.0, 5.0],
        [-26.0, 14.5, 5.0],
        [-26.0, 2.0, 4.0],
        [-15.0, 2.0, 4.0],
      ],
      0.26,
      matBlueCoolant,
      matFlange,
      matSupport,
      { id: "line_comp_coolant", name: "Compressor Coolant", service: "Coolant Loop", pressureBar: 5.8 }
    );

    // 5. South header: flange_01 -> valve_05 -> Control Building (flange_03 & valve_04)
    this._addPipeRun(
      [
        [-6.0, -3.5, 4.0],
        [-6.0, -7.2, 4.0],
        [-6.0, -25.0, 2.5],
        [0.0, -25.0, 2.5],
        [0.0, -25.0, 0.2],
      ],
      0.30,
      matYellowGas,
      matFlange,
      matSupport,
      { id: "line_south_header", name: "South Manifold Gas Line", service: "High-Pressure Gas", pressureBar: 17.5 }
    );
    this._addPipeRun(
      [
        [-29.0, -13.0, 3.5],
        [-29.0, -10.5, 3.5],
        [-18.5, -10.5, 3.5],
        [-18.5, -20.0, 4.0],
        [-21.0, -20.0, 4.0],
      ],
      0.26,
      matSteel,
      matFlange,
      matSupport,
      { id: "line_control_bypass", name: "Control Unit Bypass", service: "Instrument Line", pressureBar: 9.0 }
    );
  }

  _addPerimeterFence(minX, maxX, minY, maxY) {
    const railMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      metalness: 0.72,
      roughness: 0.42,
    });
    const postGeo = new THREE.CylinderGeometry(0.08, 0.08, 2.2, 6);

    const w = maxX - minX;
    const d = maxY - minY;
    const spanStep = 10;

    // Horizontal top & mid guardrails along all 4 sides
    const hRailX = new THREE.BoxGeometry(w, 0.08, 0.08);
    const hRailZ = new THREE.BoxGeometry(0.08, 0.08, d);
    for (const ry of [1.1, 2.0]) {
      const north = new THREE.Mesh(hRailX, railMat);
      north.position.set(0, ry, -maxY);
      this.group.add(north);
      const south = new THREE.Mesh(hRailX, railMat);
      south.position.set(0, ry, -minY);
      this.group.add(south);
      const east = new THREE.Mesh(hRailZ, railMat);
      east.position.set(maxX, ry, 0);
      this.group.add(east);
      const west = new THREE.Mesh(hRailZ, railMat);
      west.position.set(minX, ry, 0);
      this.group.add(west);
    }

    // Fence posts every 10m
    for (let x = minX; x <= maxX; x += spanStep) {
      for (const z of [-minY, -maxY]) {
        const post = new THREE.Mesh(postGeo, railMat);
        post.position.set(x, 1.1, z);
        this.group.add(post);
      }
    }
    for (let y = minY + spanStep; y < maxY; y += spanStep) {
      for (const x of [minX, maxX]) {
        const post = new THREE.Mesh(postGeo, railMat);
        post.position.set(x, 1.1, -y);
        this.group.add(post);
      }
    }
  }

  build(worldInit) {
    while (this.group.children.length > 0) {
      this.group.remove(this.group.children.pop());
    }
    this.interactiveObjects = [];

    const [minX, minY] = worldInit.bounds_min;
    const [maxX, maxY] = worldInit.bounds_max;
    const width = maxX - minX;
    const depth = maxY - minY;

    // 1. Surrounding industrial earth/gravel apron
    const outerGeo = new THREE.PlaneGeometry(560, 560);
    const outerMat = new THREE.MeshStandardMaterial({
      color: 0x1a2028,
      roughness: 0.96,
      metalness: 0.02,
    });
    const outerGround = new THREE.Mesh(outerGeo, outerMat);
    outerGround.rotation.x = -Math.PI / 2;
    outerGround.position.y = -0.06;
    outerGround.receiveShadow = true;
    this.group.add(outerGround);

    // 2. Main 120x120m weathered concrete refinery pad (raycast target)
    const { albedo } = createConcreteTextures();
    const groundGeo = new THREE.PlaneGeometry(width, depth);
    const groundMat = new THREE.MeshStandardMaterial({
      map: albedo,
      roughness: 0.88,
      metalness: 0.06,
    });
    this.groundMesh = new THREE.Mesh(groundGeo, groundMat);
    this.groundMesh.rotation.x = -Math.PI / 2;
    this.groundMesh.position.set(0, 0, 0);
    this.groundMesh.receiveShadow = true;
    this.group.add(this.groundMesh);

    // 3. Asphalt access roads + dashed lane markings
    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x1e232a,
      roughness: 0.92,
      metalness: 0.04,
    });
    const roadNS = new THREE.Mesh(new THREE.PlaneGeometry(8.5, depth - 6), roadMat);
    roadNS.rotation.x = -Math.PI / 2;
    roadNS.position.set(-49, 0.015, 0);
    roadNS.receiveShadow = true;
    this.group.add(roadNS);

    const roadEW = new THREE.Mesh(new THREE.PlaneGeometry(width - 6, 8.5), roadMat);
    roadEW.rotation.x = -Math.PI / 2;
    roadEW.position.set(0, 0.016, 46);
    roadEW.receiveShadow = true;
    this.group.add(roadEW);

    // Perimeter yellow boundary line
    const borderPts = [
      new THREE.Vector3(minX + 0.8, 0.04, -minY - 0.8),
      new THREE.Vector3(maxX - 0.8, 0.04, -minY - 0.8),
      new THREE.Vector3(maxX - 0.8, 0.04, -maxY + 0.8),
      new THREE.Vector3(minX + 0.8, 0.04, -maxY + 0.8),
      new THREE.Vector3(minX + 0.8, 0.04, -minY - 0.8),
    ];
    this.group.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(borderPts),
        new THREE.LineBasicMaterial({ color: 0xeab308, opacity: 0.65, transparent: true })
      )
    );

    // Perimeter floodlight poles
    const poleGeo = new THREE.CylinderGeometry(0.14, 0.22, 13, 8);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.72, roughness: 0.38 });
    const lampGeo = new THREE.BoxGeometry(1.1, 0.25, 0.55);
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xfff3d1 });
    const poleCoords = [
      [-57, -57], [0, -57], [57, -57],
      [-57, 57], [0, 57], [57, 57],
      [-57, 0], [57, 0],
    ];
    for (const [px, py] of poleCoords) {
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.set(px, 6.5, -py);
      pole.castShadow = true;
      this.group.add(pole);

      const lamp = new THREE.Mesh(lampGeo, lampMat);
      lamp.position.set(px, 13.1, -py);
      this.group.add(lamp);
    }

    // 4. Blender GLB Structures (exact 1:1 alignment with BuildingBox AABBs)
    const footprintMat = new THREE.LineBasicMaterial({
      color: 0xeab308,
      transparent: true,
      opacity: 0.55,
    });

    for (const b of worldInit.buildings) {
      const sx = b.max_xyz[0] - b.min_xyz[0];
      const sy = b.max_xyz[1] - b.min_xyz[1];
      const sz = b.max_xyz[2] - b.min_xyz[2];
      const cx = 0.5 * (b.min_xyz[0] + b.max_xyz[0]);
      const cy = 0.5 * (b.min_xyz[1] + b.max_xyz[1]);

      const template = this.models[b.kind] ?? this.models.building;
      const instance = template.clone(true);

      simToThree([cx, cy, b.min_xyz[2]], instance.position);
      if (b.kind === "storage_tank") {
        const diam = Math.min(sx, sy);
        instance.scale.set(diam, sz, diam);
      } else {
        instance.scale.set(sx, sz, sy);
      }
      this.group.add(instance);
      this._tagInteractive(instance, {
        kind: "building",
        id: b.id,
        structureKind: b.kind,
        size: [sx, sy, sz],
        centerSim: [cx, cy, 0.5 * sz],
      });

      // Exact axis-aligned safety outline around each building footprint
      const pad = 0.7;
      const x0 = b.min_xyz[0] - pad;
      const x1 = b.max_xyz[0] + pad;
      const z0 = -(b.min_xyz[1] - pad);
      const z1 = -(b.max_xyz[1] + pad);
      const rectPts = [
        new THREE.Vector3(x0, 0.04, z0),
        new THREE.Vector3(x1, 0.04, z0),
        new THREE.Vector3(x1, 0.04, z1),
        new THREE.Vector3(x0, 0.04, z1),
        new THREE.Vector3(x0, 0.04, z0),
      ];
      this.group.add(
        new THREE.Line(new THREE.BufferGeometry().setFromPoints(rectPts), footprintMat)
      );
    }

    // 5. Facility assets (grounded flanged gate valves with steel riser stands)
    const valveTemplate = this.models.valve;
    const riserMat = new THREE.MeshStandardMaterial({
      color: 0x3b4554,
      metalness: 0.78,
      roughness: 0.35,
    });
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x262c34,
      roughness: 0.9,
    });

    for (let i = 0; i < worldInit.assets.length; i++) {
      const asset = worldInit.assets[i];
      const valve = valveTemplate.clone(true);
      simToThree(asset.pos, valve.position);
      valve.scale.setScalar(1.3);
      valve.rotation.y = (i * Math.PI) / 4;
      this.group.add(valve);
      this._tagInteractive(valve, {
        kind: "asset",
        id: asset.id,
        assetType: asset.asset_type,
        ageYears: asset.age_years,
        posSim: asset.pos,
      });

      // Grounded vertical steel riser pipe & concrete footing under every valve
      const h = Math.max(0.2, valve.position.y);
      const riser = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.24, h, 10), riserMat);
      riser.position.set(valve.position.x, h * 0.5, valve.position.z);
      riser.castShadow = true;
      riser.receiveShadow = true;
      this.group.add(riser);

      const footing = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.22, 1.1), baseMat);
      footing.position.set(valve.position.x, 0.11, valve.position.z);
      footing.receiveShadow = true;
      this.group.add(footing);
    }

    // 6. Connected 3D industrial piping network linking tanks, buildings, rack, and valves
    this._buildPipingNetwork();

    // 7. Industrial steel perimeter fence along the pad boundary
    this._addPerimeterFence(minX, maxX, minY, maxY);
  }
}
