import * as THREE from "three";
import { OutlinePass } from "three/examples/jsm/postprocessing/OutlinePass.js";

const DRONE_ACCENTS = {
  tabpfn: 0x22d3ee,
  gradient: 0xfb923c,
  lawnmower: 0xe2e8f0,
  manual: 0xa3e635,
};

const DEFAULT_SELECT_COLOR = 0x38bdf8;
const HOVER_COLOR = 0xe0f2fe;

/**
 * Whole-object 3D hover & selection highlighter.
 *
 * Combines:
 *  - Crisp silhouette + subtle occluded-edge outline via OutlinePass (hover & select)
 *  - Non-destructive whole-object surface emissive tint (clones materials on select,
 *    restores originals on deselect so shared PBR materials are never mutated)
 *  - Subtle precision click-point pin for pipelines (shows where "Trigger Leak Here" lands)
 */
export class SelectionHighlighter {
  constructor(scene, camera, width, height) {
    this.scene = scene;
    this.camera = camera;

    const res = new THREE.Vector2(width, height);

    // Subtle hover silhouette pass (disabled when idle so EffectComposer skips it)
    this.hoverPass = new OutlinePass(res, scene, camera);
    this.hoverPass.enabled = false;
    this.hoverPass.downSampleRatio = 1;
    this.hoverPass.edgeStrength = 2.6;
    this.hoverPass.edgeGlow = 0.25;
    this.hoverPass.edgeThickness = 1.0;
    this.hoverPass.pulsePeriod = 0;
    this.hoverPass.visibleEdgeColor.setHex(HOVER_COLOR);
    this.hoverPass.hiddenEdgeColor.setHex(0x0c4a6e);

    // Prominent selection silhouette pass (disabled when idle so EffectComposer skips it)
    this.selectPass = new OutlinePass(res, scene, camera);
    this.selectPass.enabled = false;
    this.selectPass.downSampleRatio = 1;
    this.selectPass.edgeStrength = 4.6;
    this.selectPass.edgeGlow = 0.6;
    this.selectPass.edgeThickness = 1.6;
    this.selectPass.pulsePeriod = 2.8;
    this.selectPass.visibleEdgeColor.setHex(DEFAULT_SELECT_COLOR);
    this.selectPass.hiddenEdgeColor.setHex(0x0c4a6e);

    this._selectedRoot = null;
    this._selectedMeta = null;
    this._hoveredRoot = null;
    this._accentColor = new THREE.Color(DEFAULT_SELECT_COLOR);

    // Track temporarily cloned materials for clean restoration on deselect
    // Array of { mesh, originalMaterial, clonedMaterial, baseEmissive, baseIntensity }
    this._tintedMeshes = [];

    // Precision pin marker for clicked point along a pipeline (for "Trigger Leak Here")
    this._pinGroup = new THREE.Group();
    this._pinGroup.visible = false;

    const pinCore = new THREE.Mesh(
      new THREE.SphereGeometry(0.32, 16, 16),
      new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        depthTest: false,
      })
    );
    const pinHalo = new THREE.Mesh(
      new THREE.RingGeometry(0.52, 0.78, 28),
      new THREE.MeshBasicMaterial({
        color: 0x7dd3fc,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.85,
        depthTest: false,
      })
    );
    pinHalo.rotation.x = -Math.PI / 2;
    this._pinHalo = pinHalo;
    this._pinGroup.add(pinCore, pinHalo);
    this.scene.add(this._pinGroup);
  }

  getPasses() {
    return [this.hoverPass, this.selectPass];
  }

  _accentHexForMeta(meta) {
    if (meta?.kind === "drone") {
      return DRONE_ACCENTS[meta.id] ?? DEFAULT_SELECT_COLOR;
    }
    return DEFAULT_SELECT_COLOR;
  }

  isSelected(rootObj) {
    return Boolean(rootObj) && this._selectedRoot === rootObj;
  }

  setHovered(rootObj) {
    const next = rootObj && rootObj !== this._selectedRoot ? rootObj : null;
    if (next === this._hoveredRoot) return;
    this._hoveredRoot = next;
    this.hoverPass.selectedObjects = next ? [next] : [];
    this.hoverPass.enabled = Boolean(next);
  }

  select(rootObj, meta = null, hitWorldPoint = null) {
    if (!rootObj) {
      this.clear();
      return;
    }

    const sameRoot = this._selectedRoot === rootObj;
    if (!sameRoot) {
      this._restoreTintedMaterials();
      this._selectedRoot = rootObj;
      this.selectPass.selectedObjects = [rootObj];
      this.selectPass.enabled = true;
      if (this._hoveredRoot === rootObj) {
        this._hoveredRoot = null;
        this.hoverPass.selectedObjects = [];
        this.hoverPass.enabled = false;
      }
    }

    this._selectedMeta = meta;
    const accentHex = this._accentHexForMeta(meta);
    this._accentColor.setHex(accentHex);

    this.selectPass.visibleEdgeColor.setHex(accentHex);
    this.selectPass.hiddenEdgeColor.setHex(accentHex).multiplyScalar(0.24);

    if (!sameRoot) {
      this._applySurfaceTint(rootObj, this._accentColor);
    }

    // Only show the precision point pin for pipelines (where user clicked along a long pipe run)
    if (meta?.kind === "pipe" && hitWorldPoint) {
      this._pinGroup.position.copy(hitWorldPoint);
      this._pinGroup.visible = true;
    } else {
      this._pinGroup.visible = false;
    }
  }

  clear() {
    this._restoreTintedMaterials();
    this._selectedRoot = null;
    this._selectedMeta = null;
    this.selectPass.selectedObjects = [];
    this.selectPass.enabled = false;
    this._pinGroup.visible = false;
  }

  clearAll() {
    this.clear();
    this._hoveredRoot = null;
    this.hoverPass.selectedObjects = [];
    this.hoverPass.enabled = false;
  }

  _applySurfaceTint(rootObj, accentColor) {
    rootObj.traverse((child) => {
      if (!child.isMesh || !child.visible) return;
      const mat = child.material;
      if (!mat || Array.isArray(mat)) return;
      // Only tint lit standard/physical/lambert/phong materials that support emissive
      if (!mat.emissive) return;
      // Skip already bright emissive accents (LEDs, glowing windows) so they don't wash out
      if ((mat.emissiveIntensity ?? 0) > 1.0) return;

      const cloned = mat.clone();
      const baseEmissive = cloned.emissive.clone();
      const baseIntensity = cloned.emissiveIntensity ?? 1.0;

      cloned.emissive.copy(baseEmissive).lerp(accentColor, 0.85);
      cloned.emissiveIntensity = Math.max(baseIntensity, 0.24);
      child.material = cloned;

      this._tintedMeshes.push({
        mesh: child,
        originalMaterial: mat,
        clonedMaterial: cloned,
        baseIntensity: Math.max(baseIntensity, 0.2),
      });
    });
  }

  _restoreTintedMaterials() {
    for (const entry of this._tintedMeshes) {
      entry.mesh.material = entry.originalMaterial;
      entry.clonedMaterial.dispose();
    }
    this._tintedMeshes.length = 0;
  }

  animate(elapsed) {
    if (this._tintedMeshes.length > 0) {
      const pulse = 0.22 + 0.09 * Math.sin(elapsed * 3.2);
      for (const entry of this._tintedMeshes) {
        entry.clonedMaterial.emissiveIntensity = Math.max(entry.baseIntensity, pulse);
      }
    }

    if (this._pinGroup.visible) {
      const s = 1.0 + 0.18 * Math.sin(elapsed * 4.5);
      this._pinHalo.scale.setScalar(s);
    }
  }
}
