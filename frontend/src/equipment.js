import * as THREE from "three";

function addMesh(group, geometry, material, position) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
  return mesh;
}

function box(group, material, size, position) {
  return addMesh(group, new THREE.BoxGeometry(...size), material, position);
}

function cylinder(group, material, top, bottom, length, position, axis = "y") {
  const mesh = addMesh(
    group, new THREE.CylinderGeometry(top, bottom, length, 20), material, position,
  );
  if (axis === "x") mesh.rotation.z = Math.PI / 2;
  return mesh;
}

function frameBeam(group, material, start, end, width) {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = b.clone().sub(a);
  const beam = box(group, material, [width, direction.length(), width], a.lerp(b, 0.5).toArray());
  beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
}

function fractionator({ graphite, steel, copper, amber }) {
  const group = new THREE.Group();
  cylinder(group, graphite, 0.46, 0.50, 0.08, [0, 0.04, 0]);
  cylinder(group, copper, 0.29, 0.38, 0.76, [0, 0.46, 0]);
  cylinder(group, steel, 0.17, 0.29, 0.12, [0, 0.90, 0]);
  cylinder(group, graphite, 0.15, 0.16, 0.04, [0, 0.98, 0]);
  for (const y of [0.24, 0.49, 0.74]) {
    cylinder(group, graphite, 0.43, 0.43, 0.038, [0, y, 0]);
    cylinder(group, amber, 0.43, 0.43, 0.018, [0, y + 0.025, 0]);
    box(group, steel, [0.09, 0.14, 0.14], [0.43, y + 0.02, 0]);
  }
  return group;
}

function meteringSkid({ graphite, steel, teal, amber }) {
  const group = new THREE.Group();
  box(group, graphite, [1, 0.07, 1], [0, 0.035, 0]);
  for (const z of [-0.25, 0.25]) {
    cylinder(group, teal, 0.14, 0.14, 0.76, [0, 0.29, z], "x");
    for (const x of [-0.37, 0.37]) {
      cylinder(group, steel, 0.18, 0.18, 0.06, [x, 0.29, z], "x");
      box(group, graphite, [0.11, 0.12, 0.23], [x, 0.11, z]);
    }
    cylinder(group, amber, 0.065, 0.065, 0.24, [0.19, 0.53, z]);
    cylinder(group, steel, 0.11, 0.11, 0.045, [0.19, 0.67, z]);
  }
  box(group, teal, [0.22, 0.64, 0.24], [-0.35, 0.54, 0.07]);
  box(group, steel, [0.23, 0.075, 0.26], [-0.35, 0.93, 0.07]);
  return group;
}

function receiverSkid({ graphite, steel, teal }) {
  const group = new THREE.Group();
  box(group, graphite, [1, 0.07, 1], [0, 0.035, 0]);
  cylinder(group, steel, 0.28, 0.28, 0.74, [-0.02, 0.41, 0], "x");
  for (const x of [-0.35, 0.35]) {
    cylinder(group, teal, 0.31, 0.31, 0.12, [x, 0.41, 0], "x");
    box(group, graphite, [0.13, 0.19, 0.4], [x, 0.13, 0]);
  }
  cylinder(group, steel, 0.08, 0.08, 0.26, [0.1, 0.79, 0]);
  cylinder(group, teal, 0.12, 0.12, 0.05, [0.1, 0.96, 0]);
  return group;
}

function cryogenicVessel({ graphite, enamel, blue }) {
  const group = new THREE.Group();
  cylinder(group, graphite, 0.44, 0.50, 0.07, [0, 0.035, 0]);
  cylinder(group, enamel, 0.43, 0.43, 0.82, [0, 0.48, 0]);
  cylinder(group, enamel, 0.22, 0.43, 0.11, [0, 0.945, 0]);
  for (const y of [0.16, 0.55, 0.86]) {
    cylinder(group, blue, 0.448, 0.448, 0.032, [0, y, 0]);
  }
  cylinder(group, graphite, 0.055, 0.055, 0.70, [0.37, 0.52, 0.20]);
  return group;
}

function transferGantry({ graphite, steel, blue }) {
  const group = new THREE.Group();
  for (const z of [-0.39, 0.39]) {
    for (const y of [0.10, 0.90]) {
      box(group, blue, [1, 0.10, 0.10], [0, y, z]);
    }
    for (let i = 0; i < 6; i++) {
      const start = -0.47 + i * (0.94 / 6);
      const end = -0.47 + (i + 1) * (0.94 / 6);
      const slope = i % 2 === 0;
      frameBeam(group, steel, [start, slope ? 0.15 : 0.85, z], [end, slope ? 0.85 : 0.15, z], 0.05);
    }
  }
  box(group, graphite, [1, 0.06, 0.93], [0, 0.06, 0]);
  for (const z of [-0.22, 0, 0.22]) {
    cylinder(group, steel, 0.055, 0.055, 0.97, [0, 0.62, z], "x");
  }
  return group;
}

function compressorTrain({ graphite, steel, teal, amber }) {
  const group = new THREE.Group();
  box(group, graphite, [1, 0.10, 1], [0, 0.05, 0]);
  for (const z of [-0.22, 0.22]) {
    cylinder(group, teal, 0.19, 0.19, 0.59, [-0.12, 0.40, z], "x");
    cylinder(group, steel, 0.17, 0.17, 0.22, [0.29, 0.40, z], "x");
    cylinder(group, amber, 0.201, 0.201, 0.065, [0.16, 0.40, z], "x");
    for (let fin = 0; fin < 4; fin++) {
      cylinder(group, graphite, 0.202, 0.202, 0.019, [-0.31 + fin * 0.06, 0.40, z], "x");
    }
  }
  box(group, teal, [0.24, 0.75, 0.28], [0.30, 0.60, 0]);
  box(group, graphite, [0.18, 0.12, 0.04], [0.30, 0.79, 0.15]);
  return group;
}

function coolingCell({ graphite, enamel, blue }) {
  const group = new THREE.Group();
  box(group, graphite, [1, 0.13, 1], [0, 0.065, 0]);
  box(group, blue, [0.92, 0.33, 0.87], [0, 0.29, 0]);
  box(group, graphite, [0.89, 0.43, 0.81], [0, 0.64, 0]);
  for (const z of [-0.42, 0.42]) {
    for (let row = 0; row < 6; row++) {
      box(group, enamel, [0.87, 0.034, 0.055], [0, 0.44 + row * 0.066, z]);
    }
  }
  box(group, blue, [0.96, 0.06, 0.93], [0, 0.88, 0]);
  cylinder(group, enamel, 0.42, 0.42, 0.09, [0, 0.955, 0]);
  cylinder(group, graphite, 0.35, 0.35, 0.014, [0, 0.991, 0]);
  return group;
}

function exhaustStack({ graphite, steel }) {
  const group = new THREE.Group();
  cylinder(group, steel, 0.36, 0.47, 0.90, [0, 0.45, 0]);
  cylinder(group, steel, 0.40, 0.36, 0.10, [0, 0.95, 0]);
  cylinder(group, graphite, 0.50, 0.50, 0.04, [0, 0.02, 0]);
  for (const [y, radius] of [[0.31, 0.44], [0.61, 0.41]]) {
    cylinder(group, graphite, radius, radius, 0.018, [0, y, 0]);
  }
  return group;
}

// Templates occupy one simulation footprint and are shared by every instance.
export function createEquipmentModels() {
  const palette = {
    graphite: new THREE.MeshStandardMaterial({ color: 0x29353c, metalness: 0.42, roughness: 0.62 }),
    steel: new THREE.MeshStandardMaterial({ color: 0x99a9b2, metalness: 0.64, roughness: 0.48 }),
    enamel: new THREE.MeshStandardMaterial({ color: 0xb6c5c5, metalness: 0.28, roughness: 0.62 }),
    copper: new THREE.MeshStandardMaterial({ color: 0x8a5845, metalness: 0.46, roughness: 0.62 }),
    blue: new THREE.MeshStandardMaterial({ color: 0x30576d, metalness: 0.36, roughness: 0.57 }),
    teal: new THREE.MeshStandardMaterial({ color: 0x366759, metalness: 0.29, roughness: 0.61 }),
    amber: new THREE.MeshStandardMaterial({ color: 0xc99825, metalness: 0.32, roughness: 0.52 }),
  };
  return {
    fractionator: fractionator(palette),
    metering_skid: meteringSkid(palette),
    receiver_skid: receiverSkid(palette),
    cryogenic_vessel: cryogenicVessel(palette),
    transfer_gantry: transferGantry(palette),
    compressor_train: compressorTrain(palette),
    cooling_cell: coolingCell(palette),
    exhaust_stack: exhaustStack(palette),
  };
}
