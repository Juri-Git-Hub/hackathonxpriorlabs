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

  _buildPipingNetwork(pipesFromWorld = null) {
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

    if (Array.isArray(pipesFromWorld) && pipesFromWorld.length > 0) {
      const matByKind = {
        gas: matYellowGas,
        steel: matSteel,
        coolant: matBlueCoolant,
      };
      for (const p of pipesFromWorld) {
        this._addPipeRun(
          p.points,
          p.radius,
          matByKind[p.material] ?? matSteel,
          matFlange,
          matSupport,
          {
            id: p.id,
            name: p.name,
            service: p.service,
            pressureBar: p.pressure_bar,
          }
        );
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

  _addOuterRefineryContext() {
    // Surround the 120x120m active search zone with a realistic outer petrochemical complex
    // so the facility sits in a rich industrial horizon rather than an empty void.
    const placements = [
      // North-West secondary tank farm
      { key: "storage_tank", pos: [-88, 78, 0], scale: [18, 15, 18], rotY: 0.4 },
      { key: "storage_tank", pos: [-62, 84, 0], scale: [16, 13, 16], rotY: 1.2 },
      { key: "pipe_rack", pos: [-76, 66, 0], scale: [28, 7, 8], rotY: 0 },
      // Far North industrial process unit
      { key: "compressor", pos: [-12, 92, 0], scale: [22, 12, 18], rotY: 0 },
      { key: "storage_tank", pos: [24, 94, 0], scale: [20, 17, 20], rotY: 2.1 },
      { key: "building", pos: [62, 88, 0], scale: [20, 12, 16], rotY: Math.PI / 2 },
      // East refinery expansion & second distant flare
      { key: "storage_tank", pos: [92, 22, 0], scale: [18, 14, 18], rotY: 0.9 },
      { key: "compressor", pos: [96, -24, 0], scale: [20, 11, 16], rotY: Math.PI / 2 },
      { key: "flare_stack", pos: [-96, 98, 0], scale: [1.15, 1.15, 1.15], rotY: 0 },
      // West utility yard
      { key: "building", pos: [-94, -18, 0], scale: [18, 10, 14], rotY: 0 },
    ];

    for (const p of placements) {
      const tpl = this.models[p.key];
      if (!tpl) continue;
      const inst = tpl.clone(true);
      simToThree(p.pos, inst.position);
      inst.scale.set(p.scale[0], p.scale[1], p.scale[2]);
      inst.rotation.y = p.rotY;
      this.group.add(inst);
    }
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

    // 1. Surrounding textured industrial terrain with subtle rolling berms outside the 120x120m pad
    const outerGeo = new THREE.PlaneGeometry(580, 580, 48, 48);
    outerGeo.rotateX(-Math.PI / 2);
    const posAttr = outerGeo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const vx = posAttr.getX(i);
      const vz = posAttr.getZ(i);
      const distOut = Math.max(Math.abs(vx) - 64, Math.abs(vz) - 64, 0);
      if (distOut > 0) {
        const berm =
          Math.min(1.0, distOut / 45) *
          (2.4 * Math.sin(vx * 0.025) * Math.cos(vz * 0.025) +
            1.4 * Math.sin(vx * 0.06 + vz * 0.04));
        posAttr.setY(i, berm - 0.15);
      } else {
        posAttr.setY(i, -0.08);
      }
    }
    outerGeo.computeVertexNormals();
    const outerTex = createOuterTerrainTexture();
    const outerMat = new THREE.MeshStandardMaterial({
      map: outerTex,
      roughness: 0.96,
      metalness: 0.02,
    });
    const outerGround = new THREE.Mesh(outerGeo, outerMat);
    outerGround.receiveShadow = true;
    this.group.add(outerGround);

    // 2. Main 120x120m weathered concrete refinery pad with bump relief (raycast target)
    const { map: concreteMap, bumpMap: concreteBump } = createRefineryConcreteTextures();
    const groundGeo = new THREE.PlaneGeometry(width, depth);
    const groundMat = new THREE.MeshStandardMaterial({
      map: concreteMap,
      bumpMap: concreteBump,
      bumpScale: 0.08,
      roughness: 0.88,
      metalness: 0.06,
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

    // Perimeter floodlight poles + warm ground light-pool decals
    const poleGeo = new THREE.CylinderGeometry(0.14, 0.22, 13, 8);
    const poleMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.72, roughness: 0.38 });
    const lampGeo = new THREE.BoxGeometry(1.1, 0.25, 0.55);
    const lampMat = new THREE.MeshBasicMaterial({ color: 0xfff3d1 });

    // Warm radial floodlight pool texture
    const poolCanvas = document.createElement("canvas");
    poolCanvas.width = 128;
    poolCanvas.height = 128;
    const pctx = poolCanvas.getContext("2d");
    const pg = pctx.createRadialGradient(64, 64, 2, 64, 64, 62);
    pg.addColorStop(0.0, "rgba(255, 228, 175, 0.28)");
    pg.addColorStop(0.5, "rgba(255, 210, 140, 0.10)");
    pg.addColorStop(1.0, "rgba(255, 200, 120, 0.0)");
    pctx.fillStyle = pg;
    pctx.fillRect(0, 0, 128, 128);
    const poolTex = new THREE.CanvasTexture(poolCanvas);
    const poolMat = new THREE.MeshBasicMaterial({
      map: poolTex,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const poolGeo = new THREE.PlaneGeometry(22, 22);

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

      const pool = new THREE.Mesh(poolGeo, poolMat);
      pool.rotation.x = -Math.PI / 2;
      pool.position.set(px, 0.03, -py);
      this.group.add(pool);
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
      const sx = b.max_xyz[0] - b.min_xyz[0];
      const sy = b.max_xyz[1] - b.min_xyz[1];
      const sz = b.max_xyz[2] - b.min_xyz[2];
      const cx = 0.5 * (b.min_xyz[0] + b.max_xyz[0]);
      const cy = 0.5 * (b.min_xyz[1] + b.max_xyz[1]);

      // Use distinct Compressor Station asset for bldg_compressor so buildings aren't clones!
      const modelKey = b.id === "bldg_compressor" ? "compressor" : b.kind;
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
      const assetGroup = new THREE.Group();
      const valvePos = simToThree(asset.pos);

      const valve = valveTemplate.clone(true);
      valve.position.copy(valvePos);
      valve.scale.setScalar(1.3);
      valve.rotation.y = (i * Math.PI) / 4;
      assetGroup.add(valve);

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

      this.group.add(assetGroup);
      this._tagInteractive(assetGroup, {
        kind: "asset",
        id: asset.id,
        assetType: asset.asset_type,
        ageYears: asset.age_years,
        posSim: asset.pos,
      });
    }

    // 6. Connected 3D industrial piping network (driven by backend WorldInitPayload.pipes)
    this._buildPipingNetwork(worldInit.pipes);

    // 7. Iconic industrial flare stack tower at NE refinery corner with warm pilot light
    if (this.models.flare_stack) {
      const flare = this.models.flare_stack.clone(true);
      simToThree([48, 46, 0], flare.position);
      this.group.add(flare);

      const flareLight = new THREE.PointLight(0xff8c2b, 38, 45, 1.5);
      simToThree([48, 46, 32], flareLight.position);
      this.group.add(flareLight);
    }

    // 8. Industrial steel perimeter fence along the pad boundary
    this._addPerimeterFence(minX, maxX, minY, maxY);

    // 9. Surrounding outer refinery complex for rich industrial depth & horizon
    this._addOuterRefineryContext();
  }
}
