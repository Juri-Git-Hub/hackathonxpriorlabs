import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  createCorrugatedCladdingTextures,
  createRoofGravelTexture,
  createTankSteelTextures,
} from "./materials.js";

const MODEL_PATHS = {
  storage_tank: "/models/storage_tank.glb",
  building: "/models/industrial_building.glb",
  compressor: "/models/compressor_station.glb",
  pipe_rack: "/models/pipe_rack.glb",
  valve: "/models/valve_assembly.glb",
  quadrotor: "/models/quadrotor.glb",
  flare_stack: "/models/flare_stack.glb",
  fractionator: "/models/fractionator.glb",
  horton_sphere: "/models/horton_sphere.glb",
  metering_skid: "/models/metering_skid.glb",
  receiver_skid: "/models/receiver_skid.glb",
  cryogenic_vessel: "/models/cryogenic_vessel.glb",
  transfer_gantry: "/models/transfer_gantry.glb",
  compressor_train: "/models/compressor_train.glb",
  cooling_cell: "/models/cooling_cell.glb",
  hyperbolic_tower: "/models/hyperbolic_tower.glb",
  exhaust_stack: "/models/exhaust_stack.glb",
  pump_house: "/models/pump_house.glb",
};

/**
 * Loads all 18 Blender-generated PBR GLB assets in parallel, enables shadows,
 * and enhances materials with procedural weathered steel / corrugated cladding maps.
 */
export async function loadModelAssets() {
  const loader = new GLTFLoader();
  const tankTex = createTankSteelTextures();
  const bldgTex = createCorrugatedCladdingTextures();
  const roofTex = createRoofGravelTexture();

  const entries = await Promise.all(
    Object.entries(MODEL_PATHS).map(async ([key, url]) => {
      const gltf = await loader.loadAsync(url);
      gltf.scene.traverse((obj) => {
        if (!obj.isMesh) return;
        obj.castShadow = true;
        obj.receiveShadow = true;

        const mat = obj.material;
        if (!mat || !mat.isMeshStandardMaterial) return;

        // Apply weathered painted steel seam texture to storage tanks & spherical shells
        if (
          (key === "storage_tank" && (obj.name.startsWith("TankBody") || obj.name.startsWith("TankDome"))) ||
          (key === "horton_sphere" && obj.name.startsWith("HortonSphereBody"))
        ) {
          mat.map = tankTex.map;
          mat.bumpMap = tankTex.bumpMap;
          mat.bumpScale = 0.04;
          mat.roughness = 0.50;
          mat.metalness = 0.42;
          mat.needsUpdate = true;
        }

        // Apply corrugated steel siding bump + texture to industrial building & pump house walls
        if (
          (key === "building" && obj.name.startsWith("MainShell")) ||
          (key === "compressor" && obj.name.startsWith("CompHall")) ||
          (key === "pump_house" && obj.name.startsWith("PumpWalls"))
        ) {
          mat.map = bldgTex.map;
          mat.bumpMap = bldgTex.bumpMap;
          mat.bumpScale = 0.06;
          mat.roughness = 0.56;
          mat.metalness = 0.44;
          mat.needsUpdate = true;
        }

        // Apply tar-and-gravel ballast texture to flat industrial roofs
        if (obj.name.startsWith("RoofDeck") || obj.name.startsWith("CompRoof")) {
          mat.map = roofTex;
          mat.bumpMap = roofTex;
          mat.bumpScale = 0.05;
          mat.roughness = 0.90;
          mat.metalness = 0.08;
          mat.needsUpdate = true;
        }
      });
      return [key, gltf.scene];
    })
  );
  return Object.fromEntries(entries);
}
