import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

const MODEL_PATHS = {
  storage_tank: "/models/storage_tank.glb",
  building: "/models/industrial_building.glb",
  pipe_rack: "/models/pipe_rack.glb",
  valve: "/models/valve_assembly.glb",
  quadrotor: "/models/quadrotor.glb",
};

/**
 * Loads all Blender-generated PBR GLB assets in parallel and configures shadow flags.
 */
export async function loadModelAssets() {
  const loader = new GLTFLoader();
  const entries = await Promise.all(
    Object.entries(MODEL_PATHS).map(async ([key, url]) => {
      const gltf = await loader.loadAsync(url);
      gltf.scene.traverse((obj) => {
        if (obj.isMesh) {
          obj.castShadow = true;
          obj.receiveShadow = true;
        }
      });
      return [key, gltf.scene];
    })
  );
  return Object.fromEntries(entries);
}
