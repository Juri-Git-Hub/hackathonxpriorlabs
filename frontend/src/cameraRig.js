import * as THREE from "three";

const WORLD_UP = new THREE.Vector3(0, 1, 0);

export const CAMERA_PRESETS = {
  overview: {
    label: "Overview",
    pos: [-64, 46, 68],
    target: [0, 6, 0],
  },
  top: {
    label: "Top-Down",
    pos: [0, 128, 0.2],
    target: [0, 0, 0],
  },
  low: {
    label: "Pipe Rack",
    pos: [-32, 13, 30],
    target: [-4, 4.5, 2],
  },
};

const MOVE_KEYS = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "KeyQ",
  "KeyE",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
]);

function isEditableTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return (
    tag === "INPUT" ||
    tag === "TEXTAREA" ||
    tag === "SELECT" ||
    el.isContentEditable
  );
}

/**
 * Ergonomic 3D camera rig layered cleanly on top of Three.js OrbitControls:
 * - Smooth damped WASD / QE (+ Arrow keys) flight with Shift (fast) & Alt (slow)
 * - Smooth animated camera presets (Overview, Top-Down, Pipe Rack)
 * - Cinematic Auto-Orbit turntable mode
 * - Smooth chase-orbit drone following (preserves user orbit offset)
 */
export class CameraRig {
  constructor(camera, controls, { onStateChange } = {}) {
    this.camera = camera;
    this.controls = controls;
    this.onStateChange = onStateChange;

    this.keys = new Set();
    this.shiftHeld = false;
    this.altHeld = false;

    this.autoOrbit = false;
    this.autoOrbitSpeed = 0.22; // rad/s
    this.followDroneId = null;
    this.activePreset = "overview";

    this._velocity = new THREE.Vector3();
    this._desiredVel = new THREE.Vector3();
    this._fwd = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._delta = new THREE.Vector3();
    this._tmpTarget = new THREE.Vector3();
    this._lastFollowPos = new THREE.Vector3();
    this._hasLastFollowPos = false;

    // Animated transition state
    this._fly = null;
    this._focusTarget = null;

    this._bindEvents();
  }

  _bindEvents() {
    this.controls.addEventListener("start", () => {
      // User grabbed the mouse to orbit/pan: stop automated camera moves
      this._fly = null;
      this._focusTarget = null;
      this._followPullIn = false;
      if (this.autoOrbit) {
        this.setAutoOrbit(false);
      }
      if (this.activePreset) {
        this.activePreset = null;
        this._emitState();
      }
    });

    window.addEventListener("keydown", (evt) => {
      if (evt.key === "Shift") this.shiftHeld = true;
      if (evt.key === "Alt") this.altHeld = true;

      if (evt.metaKey || evt.ctrlKey) return;
      if (isEditableTarget(document.activeElement)) return;

      if (MOVE_KEYS.has(evt.code)) {
        // Prevent arrow keys from scrolling anything
        if (evt.code.startsWith("Arrow")) evt.preventDefault();
        this.keys.add(evt.code);
        this._fly = null;
        this._focusTarget = null;
        if (this.followDroneId) {
          this.setFollowDrone(null);
        }
        if (this.autoOrbit) {
          this.setAutoOrbit(false);
        }
        if (this.activePreset) {
          this.activePreset = null;
          this._emitState();
        }
      }
    });

    window.addEventListener("keyup", (evt) => {
      if (evt.key === "Shift") this.shiftHeld = false;
      if (evt.key === "Alt") this.altHeld = false;
      this.keys.delete(evt.code);
    });

    const clearKeys = () => {
      this.keys.clear();
      this.shiftHeld = false;
      this.altHeld = false;
    };
    window.addEventListener("blur", clearKeys);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) clearKeys();
    });
  }

  _emitState() {
    this.onStateChange?.({
      autoOrbit: this.autoOrbit,
      followDroneId: this.followDroneId,
      activePreset: this.activePreset,
    });
  }

  isMoving() {
    return this.keys.size > 0;
  }

  flyTo(posArr, targetArr, duration = 0.65, presetKey = null) {
    this.followDroneId = null;
    this._hasLastFollowPos = false;
    this._focusTarget = null;
    this.autoOrbit = false;
    this.activePreset = presetKey;
    this._fly = {
      startPos: this.camera.position.clone(),
      endPos: new THREE.Vector3(posArr[0], posArr[1], posArr[2]),
      startTarget: this.controls.target.clone(),
      endTarget: new THREE.Vector3(targetArr[0], targetArr[1], targetArr[2]),
      elapsed: 0,
      duration,
    };
    this._emitState();
  }

  setPreset(key) {
    const preset = CAMERA_PRESETS[key];
    if (!preset) return false;
    this.flyTo(preset.pos, preset.target, 0.65, key);
    return true;
  }

  reset() {
    this.setPreset("overview");
  }

  setAutoOrbit(enabled) {
    const next = Boolean(enabled);
    if (this.autoOrbit === next) return;
    this.autoOrbit = next;
    if (next) {
      this._fly = null;
      this.activePreset = null;
    }
    this._emitState();
  }

  toggleAutoOrbit() {
    this.setAutoOrbit(!this.autoOrbit);
    return this.autoOrbit;
  }

  setFollowDrone(droneId) {
    const next = droneId || null;
    if (this.followDroneId === next) return;
    this.followDroneId = next;
    this._hasLastFollowPos = false;
    if (next) {
      this._fly = null;
      this._focusTarget = null;
      this.activePreset = null;
    }
    this._emitState();
  }

  focusPoint(worldVec) {
    this.followDroneId = null;
    this._hasLastFollowPos = false;
    this._fly = null;
    this.activePreset = null;
    this._focusTarget = worldVec.clone();
    this._emitState();
  }

  setImmediate(posArr, targetArr) {
    this._fly = null;
    this._focusTarget = null;
    this.followDroneId = null;
    this._hasLastFollowPos = false;
    this.autoOrbit = false;
    this.activePreset = null;
    this._velocity.set(0, 0, 0);
    this.camera.position.set(posArr[0], posArr[1], posArr[2]);
    this.controls.target.set(targetArr[0], targetArr[1], targetArr[2]);
    this.controls.update();
    this._emitState();
  }

  update(dt, getDroneWorldPos) {
    // 1. Smooth preset transition
    if (this._fly) {
      this._fly.elapsed += dt;
      const rawT = Math.min(1, this._fly.elapsed / this._fly.duration);
      // Cubic ease-in-out
      const t = rawT < 0.5 ? 4 * rawT * rawT * rawT : 1 - Math.pow(-2 * rawT + 2, 3) / 2;
      this.camera.position.lerpVectors(this._fly.startPos, this._fly.endPos, t);
      this.controls.target.lerpVectors(this._fly.startTarget, this._fly.endTarget, t);
      if (rawT >= 1) {
        this._fly = null;
      }
      return;
    }

    // 2. WASD / QE + Arrow key smooth flight
    let moveForward = 0;
    let moveRight = 0;
    let moveUp = 0;

    if (this.keys.has("KeyW") || this.keys.has("ArrowUp")) moveForward += 1;
    if (this.keys.has("KeyS") || this.keys.has("ArrowDown")) moveForward -= 1;
    if (this.keys.has("KeyD") || this.keys.has("ArrowRight")) moveRight += 1;
    if (this.keys.has("KeyA") || this.keys.has("ArrowLeft")) moveRight -= 1;
    if (this.keys.has("KeyE")) moveUp += 1;
    if (this.keys.has("KeyQ")) moveUp -= 1;

    this._desiredVel.set(0, 0, 0);
    if (moveForward !== 0 || moveRight !== 0 || moveUp !== 0) {
      this.camera.getWorldDirection(this._fwd);
      this._fwd.y = 0;
      if (this._fwd.lengthSq() < 1e-5) {
        // Looking straight down/up: fall back to screen-up projected onto XZ
        this._fwd.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
        this._fwd.y = 0;
        if (this._fwd.lengthSq() < 1e-5) {
          this._fwd.set(0, 0, -1);
        }
      }
      this._fwd.normalize();
      this._right.crossVectors(this._fwd, WORLD_UP).normalize();

      this._desiredVel
        .addScaledVector(this._fwd, moveForward)
        .addScaledVector(this._right, moveRight)
        .addScaledVector(WORLD_UP, moveUp);

      if (this._desiredVel.lengthSq() > 0) {
        this._desiredVel.normalize();
      }

      // Altitude-adaptive base speed for effortless macro + micro navigation
      const baseSpeed = Math.max(16, Math.min(82, this.camera.position.y * 0.85 + 12));
      const speedMult = this.shiftHeld ? 2.4 : this.altHeld ? 0.35 : 1.0;
      this._desiredVel.multiplyScalar(baseSpeed * speedMult);
    }

    const damp = 1 - Math.exp(-12 * dt);
    this._velocity.lerp(this._desiredVel, damp);

    if (this._velocity.lengthSq() > 1e-4) {
      this._delta.copy(this._velocity).multiplyScalar(dt);

      // Prevent clipping below ground plane
      const minCamY = 1.5;
      if (this.camera.position.y + this._delta.y < minCamY) {
        this._delta.y = minCamY - this.camera.position.y;
      }

      this.camera.position.add(this._delta);
      this.controls.target.add(this._delta);
      if (this.controls.target.y < 0) {
        this.controls.target.y = 0;
      }
    }

    // 3. Drone follow (chase-orbit: translates camera with drone while preserving user orbit)
    if (this.followDroneId && getDroneWorldPos) {
      const p = getDroneWorldPos(this.followDroneId, this._tmpTarget);
      if (p) {
        const alpha = Math.min(1.0, dt * 6.5);
        if (!this._hasLastFollowPos) {
          this._lastFollowPos.copy(p);
          this._hasLastFollowPos = true;
          this._followPullIn = true;
        }

        // Smoothly pull into a close chase-orbit distance if starting far away
        if (this._followPullIn) {
          const offset = this._delta.subVectors(this.camera.position, this.controls.target);
          const dist = offset.length();
          if (dist > 28) {
            offset.setLength(THREE.MathUtils.lerp(dist, 24, Math.min(1.0, dt * 4.5)));
            offset.y = THREE.MathUtils.lerp(offset.y, Math.min(offset.y, 13), Math.min(1.0, dt * 4.5));
            this.camera.position.copy(this.controls.target).add(offset);
          } else {
            this._followPullIn = false;
          }
        }

        const prevX = this.controls.target.x;
        const prevY = this.controls.target.y;
        const prevZ = this.controls.target.z;
        this.controls.target.lerp(p, alpha);
        this.camera.position.x += this.controls.target.x - prevX;
        this.camera.position.y = Math.max(2.0, this.camera.position.y + (this.controls.target.y - prevY));
        this.camera.position.z += this.controls.target.z - prevZ;
      }
    } else if (this._focusTarget) {
      this.controls.target.lerp(this._focusTarget, Math.min(1.0, dt * 6.0));
      if (this.controls.target.distanceTo(this._focusTarget) < 0.25) {
        this._focusTarget = null;
      }
    }

    // 4. Cinematic auto-orbit turntable around controls.target
    if (this.autoOrbit) {
      const offset = this._delta.subVectors(this.camera.position, this.controls.target);
      offset.applyAxisAngle(WORLD_UP, this.autoOrbitSpeed * dt);
      this.camera.position.copy(this.controls.target).add(offset);
    }
  }
}
