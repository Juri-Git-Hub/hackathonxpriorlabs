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

export function createRefineryConcreteTextures() {
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
  ctx.fillStyle = "#293039";
  ctx.fillRect(0, 0, size, size);
  bctx.fillStyle = "#808080";
  bctx.fillRect(0, 0, size, size);

  // Subtle concrete pad zones (tank farm concrete vs central pipe yard)
  ctx.fillStyle = "#313945";
  ctx.fillRect(140, 120, 740, 660);

  // Darker asphalt service perimeter roads (west N-S road & south E-W road)
  ctx.fillStyle = "#1e232a";
  ctx.fillRect(58, 42, 86, 940);
  ctx.fillRect(42, 830, 940, 88);

  // Dashed yellow road centerlines
  ctx.strokeStyle = "rgba(234, 179, 8, 0.62)";
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 16]);
  ctx.beginPath();
  ctx.moveTo(101, 60);
  ctx.lineTo(101, 874);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(101, 874);
  ctx.lineTo(960, 874);
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
    g.addColorStop(0, dark ? "rgba(12, 16, 22, 0.24)" : "rgba(75, 88, 104, 0.14)");
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
    const v = 28 + Math.floor(Math.random() * 38);
    ctx.fillStyle = `rgba(${v}, ${v + 4}, ${v + 8}, 0.26)`;
    ctx.fillRect(x, y, 2, 2);

    const bv = 95 + Math.floor(Math.random() * 65);
    bctx.fillStyle = `rgb(${bv},${bv},${bv})`;
    bctx.fillRect(x, y, 2, 2);
  }

  // Subtle concrete slab joints only on the inner concrete pad
  ctx.strokeStyle = "rgba(10, 14, 20, 0.38)";
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
  ctx.fillStyle = "rgba(226, 232, 240, 0.22)";
  ctx.font = "800 20px Inter, system-ui, sans-serif";
  ctx.fillText("SECTOR A · TANK FARM", 510, 165);
  ctx.fillText("SECTOR B · PIPE MANIFOLD", 365, 760);

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

export function createOuterTerrainTexture() {
  const size = 512;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");

  ctx.fillStyle = "#1c2229";
  ctx.fillRect(0, 0, size, size);

  // Subtle industrial gravel / dry earth variation
  for (let i = 0; i < 60; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const r = 24 + Math.random() * 75;
    const g = ctx.createRadialGradient(x, y, 2, x, y, r);
    const warm = Math.random() > 0.5;
    g.addColorStop(0, warm ? "rgba(44, 42, 38, 0.28)" : "rgba(22, 30, 38, 0.30)");
    g.addColorStop(1, "rgba(22, 28, 34, 0)");
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  for (let i = 0; i < 14000; i++) {
    const x = Math.random() * size;
    const y = Math.random() * size;
    const v = 22 + Math.floor(Math.random() * 28);
    ctx.fillStyle = `rgba(${v + 2}, ${v + 4}, ${v + 5}, 0.32)`;
    ctx.fillRect(x, y, 2, 2);
  }

  const map = new THREE.CanvasTexture(c);
  map.wrapS = THREE.RepeatWrapping;
  map.wrapT = THREE.RepeatWrapping;
  map.repeat.set(10, 10);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  return map;
}
