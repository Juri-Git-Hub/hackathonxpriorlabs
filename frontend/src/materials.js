import * as THREE from "three";

/**
 * Procedural PBR Canvas Textures (Albedo + Bump + Roughness + Contact AO)
 * Zero external image dependencies, loads in < 20ms, eliminates flat plastic look.
 */

export function createTankSteelTextures() {
  const w = 512;
  const h = 512;
  const albedoCanvas = document.createElement("canvas");
  albedoCanvas.width = w;
  albedoCanvas.height = h;
  const ctx = albedoCanvas.getContext("2d");

  const bumpCanvas = document.createElement("canvas");
  bumpCanvas.width = w;
  bumpCanvas.height = h;
  const bctx = bumpCanvas.getContext("2d");

  // Weathered industrial off-white / light steel base
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0.0, "#cbd5e1");
  grad.addColorStop(0.65, "#bfc9d4");
  grad.addColorStop(0.92, "#9ca6b2");
  grad.addColorStop(1.0, "#757e8a"); // Grime near ground plinth
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  bctx.fillStyle = "#808080";
  bctx.fillRect(0, 0, w, h);

  // Subtle vertical rain streaks & weathering patina
  for (let i = 0; i < 240; i++) {
    const x = Math.random() * w;
    const y = Math.random() * h * 0.7;
    const len = 40 + Math.random() * 140;
    const alpha = 0.03 + Math.random() * 0.06;
    ctx.fillStyle = `rgba(45, 55, 68, ${alpha})`;
    ctx.fillRect(x, y, 1.5 + Math.random() * 2.5, len);
  }

  // Welded steel horizontal & vertical plate seams
  ctx.strokeStyle = "rgba(35, 45, 58, 0.35)";
  ctx.lineWidth = 2;
  bctx.strokeStyle = "#303030";
  bctx.lineWidth = 3;
  for (let y = 64; y < h; y += 64) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    bctx.beginPath();
    bctx.moveTo(0, y);
    bctx.lineTo(w, y);
    bctx.stroke();
  }
  for (let x = 0; x < w; x += 64) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    bctx.beginPath();
    bctx.moveTo(x, 0);
    bctx.lineTo(x, h);
    bctx.stroke();
  }

  // Fine metallic grain noise
  for (let i = 0; i < 3500; i++) {
    const x = Math.random() * w;
    const y = Math.random() * h;
    const v = 100 + Math.floor(Math.random() * 80);
    ctx.fillStyle = `rgba(${v},${v},${v},0.06)`;
    ctx.fillRect(x, y, 2, 2);
  }

  const map = new THREE.CanvasTexture(albedoCanvas);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bumpCanvas);
  bumpMap.wrapS = THREE.RepeatWrapping;
  bumpMap.wrapT = THREE.RepeatWrapping;
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

export function createCorrugatedCladdingTextures() {
  const w = 256;
  const h = 256;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");

  const bc = document.createElement("canvas");
  bc.width = w;
  bc.height = h;
  const bctx = bc.getContext("2d");

  ctx.fillStyle = "#3b4756";
  ctx.fillRect(0, 0, w, h);

  // Vertical ribbed corrugated metal siding
  for (let x = 0; x < w; x += 8) {
    ctx.fillStyle = "rgba(255,255,255,0.06)";
    ctx.fillRect(x, 0, 3, h);
    ctx.fillStyle = "rgba(0,0,0,0.14)";
    ctx.fillRect(x + 4, 0, 3, h);

    bctx.fillStyle = "#b8b8b8";
    bctx.fillRect(x, 0, 4, h);
    bctx.fillStyle = "#484848";
    bctx.fillRect(x + 4, 0, 4, h);
  }

  // Bottom weathering dirt band
  const g = ctx.createLinearGradient(0, h * 0.72, 0, h);
  g.addColorStop(0, "rgba(18, 22, 28, 0)");
  g.addColorStop(1, "rgba(18, 22, 28, 0.45)");
  ctx.fillStyle = g;
  ctx.fillRect(0, h * 0.72, w, h * 0.28);

  const map = new THREE.CanvasTexture(c);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(3, 2);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bc);
  bumpMap.wrapS = THREE.RepeatWrapping;
  bumpMap.wrapT = THREE.RepeatWrapping;
  bumpMap.repeat.set(3, 2);
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

const SURFACE_STYLES = {
  concrete: {
    outer: "#293039",
    pad: "#313945",
    road: "#1e232a",
    laneLine: "rgba(234, 179, 8, 0.62)",
    grainBase: 28,
    stencil: "rgba(226, 232, 240, 0.22)",
  },
  desert: {
    outer: "#9a8262",
    pad: "#80725f",
    road: "#5e5346",
    laneLine: "rgba(249, 115, 22, 0.72)",
    grainBase: 112,
    stencil: "rgba(255, 244, 214, 0.34)",
  },
  frost: {
    outer: "#a1aab0",
    pad: "#929ca4",
    road: "#49525b",
    laneLine: "rgba(56, 189, 248, 0.68)",
    grainBase: 150,
    stencil: "rgba(224, 242, 254, 0.36)",
  },
  asphalt: {
    outer: "#1c2026",
    pad: "#2b323d",
    road: "#14181f",
    laneLine: "rgba(250, 204, 21, 0.74)",
    grainBase: 20,
    stencil: "rgba(226, 232, 240, 0.26)",
  },
  wet_slate: {
    outer: "#202a33",
    pad: "#293542",
    road: "#172029",
    laneLine: "rgba(96, 165, 250, 0.64)",
    grainBase: 26,
    stencil: "rgba(191, 219, 254, 0.28)",
  },
};

export function createRefineryConcreteTextures({
  frosted = false,
  surface = null,
  centerLane = false,
  zoneLabel = "SECTOR A · TANK FARM",
} = {}) {
  const styleKey = surface ?? (frosted ? "frost" : "concrete");
  const st = SURFACE_STYLES[styleKey] ?? SURFACE_STYLES.concrete;

  const size = 1024;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  const bc = document.createElement("canvas");
  bc.width = size;
  bc.height = size;
  const bctx = bc.getContext("2d");

  // Non-repeating industrial site base with varied concrete & gravel aprons
  ctx.fillStyle = st.outer;
  ctx.fillRect(0, 0, size, size);
  bctx.fillStyle = "#808080";
  bctx.fillRect(0, 0, size, size);

  // Subtle concrete pad zones (tank farm concrete vs central pipe yard)
  ctx.fillStyle = st.pad;
  ctx.fillRect(140, 120, 740, 660);

  // Darker asphalt service perimeter roads (west N-S road & south E-W road)
  ctx.fillStyle = st.road;
  ctx.fillRect(58, 42, 86, 940);
  ctx.fillRect(42, 830, 940, 88);
  if (centerLane) {
    ctx.fillRect(466, 42, 92, 940);
    ctx.fillStyle = "#667078";
    ctx.fillRect(488, 42, 5, 940);
    ctx.fillRect(531, 42, 5, 940);
  }

  // Dashed road centerlines
  ctx.strokeStyle = st.laneLine;
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 16]);
  ctx.beginPath();
  ctx.moveTo(101, 60);
  ctx.lineTo(101, 874);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(101, 874);
  ctx.lineTo(960, 874);
  if (centerLane) {
    ctx.moveTo(512, 60);
    ctx.lineTo(512, 964);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Subtle pedestrian safety crosswalk near control building
  ctx.fillStyle = "rgba(226, 232, 240, 0.28)";
  for (let i = 0; i < 6; i++) {
    ctx.fillRect(68 + i * 12, 620, 7, 26);
  }

  // Macro industrial weathering, oil stains, and surface variation (non-repetitive!)
  for (let i = 0; i < 95; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 22 + Math.random() * 75;
    const dark = Math.random() > 0.35;
    const g = ctx.createRadialGradient(x, y, 2, x, y, r);
    g.addColorStop(0, dark ? "rgba(12, 16, 22, 0.22)" : "rgba(95, 108, 124, 0.14)");
    g.addColorStop(1, "rgba(12, 16, 22, 0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Fine aggregate concrete grain
  for (let i = 0; i < 24000; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const v = st.grainBase + Math.floor(Math.random() * 38);
    ctx.fillStyle = `rgba(${v}, ${v + 4}, ${v + 8}, 0.24)`;
    ctx.fillRect(x, y, 2, 2);

    const bv = 95 + Math.floor(Math.random() * 65);
    bctx.fillStyle = `rgb(${bv},${bv},${bv})`;
    bctx.fillRect(x, y, 2, 2);
  }

  // Subtle concrete slab joints only on the inner concrete pad
  ctx.strokeStyle = "rgba(10, 14, 20, 0.34)";
  ctx.lineWidth = 1.5;
  bctx.strokeStyle = "#303030";
  bctx.lineWidth = 2;
  for (let p = 140; p <= 880; p += 74) {
    ctx.beginPath();
    ctx.moveTo(p, 120);
    ctx.lineTo(p, 780);
    ctx.stroke();
    bctx.beginPath();
    bctx.moveTo(p, 120);
    bctx.lineTo(p, 780);
    bctx.stroke();
  }
  for (let p = 120; p <= 780; p += 66) {
    ctx.beginPath();
    ctx.moveTo(140, p);
    ctx.lineTo(880, p);
    ctx.stroke();
    bctx.beginPath();
    bctx.moveTo(140, p);
    bctx.lineTo(880, p);
    bctx.stroke();
  }

  // Painted tarmac zone stencils
  ctx.fillStyle = st.stencil;
  ctx.font = "800 20px Inter, system-ui, sans-serif";
  ctx.fillText(zoneLabel, 470, 165);
  ctx.fillText("UAV INSPECTION ZONE", 365, 760);

  const map = new THREE.CanvasTexture(c);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;

  const bumpMap = new THREE.CanvasTexture(bc);
  bumpMap.anisotropy = 8;

  return { map, bumpMap };
}

export function createContactAOTexture(isCircle = false) {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  if (isCircle) {
    const g = ctx.createRadialGradient(128, 128, 62, 128, 128, 124);
    g.addColorStop(0.0, "rgba(4, 7, 12, 0.78)");
    g.addColorStop(0.68, "rgba(4, 7, 12, 0.55)");
    g.addColorStop(1.0, "rgba(4, 7, 12, 0.0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
  } else {
    // Soft rectangular ambient occlusion skirt
    ctx.fillStyle = "rgba(4, 7, 12, 0.0)";
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 24; i++) {
      const pad = i * 2.2;
      const alpha = 0.035 * (i / 24);
      ctx.fillStyle = `rgba(4, 7, 12, ${alpha})`;
      ctx.fillRect(pad, pad, size - pad * 2, size - pad * 2);
    }
  }

  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

export function createDronePadTexture() {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#1c222b";
  ctx.fillRect(0, 0, size, size);

  // Dashed yellow safety border
  ctx.strokeStyle = "#eab308";
  ctx.lineWidth = 8;
  ctx.strokeRect(10, 10, size - 20, size - 20);

  // Inner circle
  ctx.beginPath();
  ctx.arc(128, 128, 78, 0, Math.PI * 2);
  ctx.strokeStyle = "rgba(234, 179, 8, 0.85)";
  ctx.lineWidth = 6;
  ctx.stroke();

  // Bold 'H' helipad / UAV launch marking
  ctx.fillStyle = "#f8fafc";
  ctx.font = "900 104px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("H", 128, 132);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export function createRoofGravelTexture() {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#262b33";
  ctx.fillRect(0, 0, size, size);

  // Tar-and-gravel ballast speckles
  for (let i = 0; i < 6500; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const v = 26 + Math.floor(Math.random() * 44);
    ctx.fillStyle = `rgba(${v}, ${v + 2}, ${v + 5}, 0.42)`;
    ctx.fillRect(x, y, 2, 2);
  }
  // Subtle weathered tar seam lines & dark drainage patches
  ctx.strokeStyle = "rgba(12, 15, 20, 0.45)";
  ctx.lineWidth = 2;
  for (let y = 32; y < size; y += 64) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(size, y);
    ctx.stroke();
  }
  for (let i = 0; i < 10; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 18 + Math.random() * 34;
    const g = ctx.createRadialGradient(x, y, 2, x, y, r);
    g.addColorStop(0, "rgba(10, 14, 20, 0.35)");
    g.addColorStop(1, "rgba(10, 14, 20, 0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  const map = new THREE.CanvasTexture(c);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(2, 2);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  return map;
}

const OUTER_TERRAIN_PALETTES = {
  concrete: {
    base: "#1c2229",
    patchA: "rgba(44, 42, 38, 0.28)",
    patchB: "rgba(22, 30, 38, 0.30)",
    grain: [22, 26, 28],
  },
  desert: {
    base: "#8f714d",
    patchA: "rgba(176, 134, 86, 0.32)",
    patchB: "rgba(110, 82, 52, 0.30)",
    grain: [120, 96, 68],
  },
  frost: {
    base: "#9aaab6",
    patchA: "rgba(210, 224, 236, 0.35)",
    patchB: "rgba(118, 136, 152, 0.28)",
    grain: [148, 162, 174],
  },
  asphalt: {
    base: "#1d232c",
    patchA: "rgba(48, 58, 72, 0.28)",
    patchB: "rgba(18, 24, 32, 0.34)",
    grain: [24, 30, 38],
  },
  wet_slate: {
    base: "#1e2a30",
    patchA: "rgba(44, 62, 68, 0.30)",
    patchB: "rgba(20, 30, 38, 0.32)",
    grain: [26, 36, 42],
  },
};

export function createOuterTerrainTexture(surface = "concrete") {
  const pal = OUTER_TERRAIN_PALETTES[surface] ?? OUTER_TERRAIN_PALETTES.concrete;
  const size = 512;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  ctx.fillStyle = pal.base;
  ctx.fillRect(0, 0, size, size);

  // Tile-seamless soft radial variation (wraps across all 4 borders so zero grid seams appear)
  for (let i = 0; i < 42; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 36 + Math.random() * 68;
    const warm = Math.random() > 0.5;
    const col = warm ? pal.patchA : pal.patchB;
    for (const ox of (-size, 0, size)) {
      for (const oy of (-size, 0, size)) {
        const cx = x + ox;
        const cy = y + oy;
        if (cx + r < 0 || cx - r > size || cy + r < 0 || cy - r > size) continue;
        const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, r);
        g.addColorStop(0, col);
        g.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  for (let i = 0; i < 16000; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const d = Math.floor(Math.random() * 20) - 10;
    ctx.fillStyle = `rgba(${Math.max(0, Math.min(255, pal.grain[0] + d))}, ${Math.max(0, Math.min(255, pal.grain[1] + d))}, ${Math.max(0, Math.min(255, pal.grain[2] + d))}, 0.24)`;
    ctx.fillRect(x, y, 2, 2);
  }

  const map = new THREE.CanvasTexture(c);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(16, 16);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  return map;
}
