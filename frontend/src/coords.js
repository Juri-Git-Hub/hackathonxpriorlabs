import * as THREE from "three";

/**
 * Strictly convert Python Z-up simulation coordinates to Three.js Y-up:
 *   three_pos = (sim_x, sim_z, -sim_y)
 */
export function simToThree(pos, target = new THREE.Vector3()) {
  return target.set(pos[0], pos[2], -pos[1]);
}

/**
 * Inverse conversion for ground-click raycasts:
 *   sim_pos = (three_x, -three_z, three_y)
 */
export function threeToSim(vec) {
  return [vec.x, -vec.z, vec.y];
}
