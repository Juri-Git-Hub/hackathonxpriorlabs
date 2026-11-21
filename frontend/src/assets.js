import { createEquipmentModels } from "./equipment.js";
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
};

/**
 * Loads all Blender-generated PBR GLB assets in parallel, enables shadows,
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

        // Apply weathered painted steel seam texture to storage tank shell & dome
        if (key === "storage_tank" && (obj.name.startsWith("TankBody") || obj.name.startsWith("TankDome"))) {
          mat.map = tankTex.map;
          mat.bumpMap = tankTex.bumpMap;
          mat.bumpScale = 0.04;
          mat.roughness = 0.52;
          mat.metalness = 0.42;
          mat.needsUpdate = true;
        }

        // Apply corrugated steel siding bump + texture to industrial building walls
        if ((key === "building" && obj.name.startsWith("MainShell")) || (key === "compressor" && obj.name.startsWith("CompHall"))) {
          mat.map = bldgTex.map;
          mat.bumpMap = bldgTex.bumpMap;
          mat.bumpScale = 0.06;
          mat.roughness = 0.58;
          mat.metalness = 0.45;
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
  return { ...Object.fromEntries(entries), ...createEquipmentModels() };
}
