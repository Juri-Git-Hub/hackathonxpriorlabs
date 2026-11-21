import * as THREE from "three";
import { simToThree } from "./coords.js";
import {
  createContactAOTexture,
  createDronePadTexture,
  createOuterTerrainTexture,
  createRefineryConcreteTextures,
} from "./materials.js";

/**
 * Builds realistic industrial refinery terrain, concrete pad, access roads,
 * valve stands and equipment within the playable footprint.
 */
export class RefineryView {
  constructor(scene, models) {
    this.scene = scene;
    this.models = models;
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.groundMesh = null;
    this.interactiveObjects = [];
    this.sharedResources = new Set();
    for (const model of Object.values(models)) {
      model.traverse((child) => {
        if (child.geometry) this.sharedResources.add(child.geometry);
        for (const material of [child.material].flat()) {
          if (material) this.sharedResources.add(material);
        }
      });
    }
  }

  _tagInteractive(obj, meta) {
    obj.traverse((child) => {
      if (child.isMesh) {
        child.userData.inspect = meta;
        child.userData.inspectRoot = obj;
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

  _buildPipingNetwork(pipeRuns = []) {
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

    if (pipeRuns.length) {
      const materials = new Map();
      const matByKind = { gas: matYellowGas, steel: matSteel, coolant: matBlueCoolant };
      for (const run of pipeRuns) {
        let material = matByKind[run.material] ?? matYellowGas;
        if (run.color != null) {
          if (!materials.has(run.color)) {
            materials.set(run.color, matYellowGas.clone());
            materials.get(run.color).color.setHex(run.color);
          }
          material = materials.get(run.color);
        }
        this._addPipeRun(run.points, run.radius, material, matFlange, matSupport, {
          id: run.id,
          name: run.name,
          from: run.from_id,
          to: run.to_id,
          service: run.service ?? "Connected process gas network",
          pressureBar: run.pressureBar ?? run.pressure_bar,
        });
      }
      return;
    }

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

  build(worldInit, presentation = {}) {
    const resources = new Set();
    this.group.traverse((child) => {
      if (child.geometry) resources.add(child.geometry);
      for (const material of [child.material].flat()) {
        if (material && !this.sharedResources.has(material)) {
          resources.add(material);
          if (material.map) resources.add(material.map);
          if (material.bumpMap) resources.add(material.bumpMap);
        }
      }
    });
    this.group.clear();
    for (const resource of resources) {
      if (!this.sharedResources.has(resource)) resource.dispose();
    }
    this.interactiveObjects = [];

    const [minX, minY] = worldInit.bounds_min;
    const [maxX, maxY] = worldInit.bounds_max;
    const width = maxX - minX;
    const depth = maxY - minY;

    const surface = presentation.surface ?? "concrete";

    // 1. Surrounding textured terrain with rolling berms outside the 120x120m facility pad
    const outerGeo = new THREE.PlaneGeometry(580, 580, 52, 52);
    outerGeo.rotateX(-Math.PI / 2);
    const posAttr = outerGeo.attributes.position;
    const bermScale = surface === "desert" ? 1.55 : surface === "asphalt" ? 1.35 : 1.0;
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i);
      const vz = posAttr.getZ(i);
      const distOut = Math.max(Math.abs(vx) - 64, Math.abs(vz) - 64, 0);
      if (distOut > 0) {
        const berm =
          Math.min(1.0, distOut / 45) *
          bermScale *
          (2.8 * Math.sin(vx * 0.025) * Math.cos(vz * 0.025) +
            1.6 * Math.sin(vx * 0.055 + vz * 0.04));
        posAttr.setY(i, berm - 0.15);
      } else {
        posAttr.setY(i, -0.08);
      }
    }
    outerGeo.computeVertexNormals();
    const outerTex = createOuterTerrainTexture(surface);
    const outerMat = new THREE.MeshStandardMaterial({
      map: outerTex,
      roughness: surface === "wet_slate" ? 0.82 : 0.96,
      metalness: 0.02,
    });
    const outerGround = new THREE.Mesh(outerGeo, outerMat);
    outerGround.receiveShadow = true;
    this.group.add(outerGround);

    // 2. Main 120x120m facility concrete/gravel pad (raycast target)
    const { map: concreteMap, bumpMap: concreteBump } = createRefineryConcreteTextures({
      surface,
      centerLane: presentation.center_lane,
      zoneLabel: presentation.zone_label ?? "SECTOR A · TANK FARM",
    });
    const groundGeo = new THREE.PlaneGeometry(width, depth);
    const groundMat = new THREE.MeshStandardMaterial({
      map: concreteMap,
      bumpMap: concreteBump,
      bumpScale: surface === "desert" ? 0.12 : 0.08,
      roughness: surface === "wet_slate" ? 0.68 : surface === "desert" ? 0.94 : 0.88,
      metalness: surface === "wet_slate" ? 0.14 : 0.06,
    });
    this.groundMesh = new THREE.Mesh(groundGeo, groundMat);
    this.groundMesh.rotation.x = -Math.PI / 2;
    this.groundMesh.position.set(0, 0, 0);
    this.groundMesh.receiveShadow = true;
    this.group.add(this.groundMesh);

    // UAV Launch Pad ('H') at SW drone start zone (-48, -40)
    const padTex = createDronePadTexture();
    const launchPad = new THREE.Mesh(
      new THREE.PlaneGeometry(13, 13),
      new THREE.MeshStandardMaterial({
        map: padTex,
        roughness: 0.85,
        metalness: 0.08,
      })
    );
    launchPad.rotation.x = -Math.PI / 2;
    simToThree([-48, -40, 0.025], launchPad.position);
    launchPad.receiveShadow = true;
    this.group.add(launchPad);

    // 3. Industrial site details (concrete Jersey barriers, safety station, equipment skids)
    const barrierMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.9,
      metalness: 0.08,
    });
    const barrierGeo = new THREE.BoxGeometry(4.2, 0.95, 0.65);
    for (let i = 0; i < 6; i++) {
      const b = new THREE.Mesh(barrierGeo, barrierMat);
      b.position.set(-42.5, 0.48, -28 + i * 11);
      b.rotation.y = Math.PI / 2;
      b.castShadow = true;
      b.receiveShadow = true;
      this.group.add(b);
    }

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
        new THREE.LineBasicMaterial({ color: 0xeab308, opacity: 0.55, transparent: true })
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

    // 4. Blender GLB Structures (exact 1:1 alignment with BuildingBox AABBs + ground contact AO)
    const footprintMat = new THREE.LineBasicMaterial({
      color: 0xeab308,
      transparent: true,
      opacity: 0.45,
    });
    const circleAOTex = createContactAOTexture(true);
    const boxAOTex = createContactAOTexture(false);

    for (const b of worldInit.buildings) {
      if (b.kind === "pipe_segment") continue;
      const sx = b.max_xyz[0] - b.min_xyz[0];
      const sy = b.max_xyz[1] - b.min_xyz[1];
      const sz = b.max_xyz[2] - b.min_xyz[2];
      const cx = 0.5 * (b.min_xyz[0] + b.max_xyz[0]);
      const cy = 0.5 * (b.min_xyz[1] + b.max_xyz[1]);

      const modelKey = b.model ?? (b.id === "bldg_compressor" ? "compressor" : b.kind);
      const template = this.models[modelKey] ?? this.models.building;
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
        centerSim: [cx, cy, b.min_xyz[2] + 0.5 * sz],
      });
      if (b.min_xyz[2] > 0) continue;

      // Soft ground contact-shadow ambient occlusion decal (grounds every structure)
      const isTank = b.kind === "storage_tank";
      const aoPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(sx * 1.24, sy * 1.24),
        new THREE.MeshBasicMaterial({
          map: isTank ? circleAOTex : boxAOTex,
          transparent: true,
          depthWrite: false,
        })
      );
      aoPlane.rotation.x = -Math.PI / 2;
      simToThree([cx, cy, 0.022], aoPlane.position);
      this.group.add(aoPlane);

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
      const assetGroup = new THREE.Group();
      const valvePos = simToThree(asset.pos);

      const valve = valveTemplate.clone(true);
      valve.position.copy(valvePos);
      valve.scale.setScalar(1.3);
      valve.rotation.y = (i * Math.PI) / 4;
      assetGroup.add(valve);

      if (!worldInit.map_id || ["refinery", "generated"].includes(worldInit.map_id)) {
        // Grounded vertical steel riser pipe & concrete footing under every valve
        const h = Math.max(0.2, valvePos.y);
        const riser = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.24, h, 10), riserMat);
        riser.position.set(valvePos.x, h * 0.5, valvePos.z);
        riser.castShadow = true;
        riser.receiveShadow = true;
        assetGroup.add(riser);

        const footing = new THREE.Mesh(new THREE.BoxGeometry(1.1, 0.22, 1.1), baseMat);
        footing.position.set(valvePos.x, 0.11, valvePos.z);
        footing.receiveShadow = true;
        assetGroup.add(footing);
      }

      this.group.add(assetGroup);
      this._tagInteractive(assetGroup, {
        kind: "asset",
        id: asset.id,
        assetType: asset.asset_type,
        ageYears: asset.age_years,
        posSim: asset.pos,
      });
    }

    // 6. Connected 3D industrial piping network linking tanks, buildings, rack, and valves
    this._buildPipingNetwork(worldInit.pipe_runs?.length ? worldInit.pipe_runs : worldInit.pipes);

    // 7. Optional iconic flare stack landmark for refinery/petrochemical worlds
    if (presentation.flare_pos && this.models.flare_stack) {
      const flare = this.models.flare_stack.clone(true);
      simToThree(presentation.flare_pos, flare.position);
      this.group.add(flare);

      const [fx, fy] = presentation.flare_pos;
      const flareLight = new THREE.PointLight(0xff8c2b, 38, 48, 1.5);
      simToThree([fx, fy, 32], flareLight.position);
      this.group.add(flareLight);
    }

    this._addPerimeterFence(minX, maxX, minY, maxY);
  }
}
