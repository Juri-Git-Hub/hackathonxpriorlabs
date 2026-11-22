const MODELS = {
  search: "Waiting for the first search prediction",
  source: "Waiting for gas contact",
  hazard: "Waiting for the first plume forecast",
};
const PANEL_WIDTH_KEY = "tabpfn-model-panel-width";
const RASTER_SIZE = 144;
const palette = [[64, 52, 160], [46, 116, 225], [48, 207, 210], [105, 222, 99], [255, 221, 76], [241, 75, 52]];
const heatColors = Array.from({ length: 256 }, (_, i) => {
  const value = i / 255;
  const scaled = value * (palette.length - 1);
  const lower = Math.min(Math.floor(scaled), palette.length - 2);
  const rgb = palette[lower].map((start, c) => Math.round(start + (palette[lower + 1][c] - start) * (scaled - lower)));
  return [...rgb, Math.round(218 * Math.min(1, value / 0.16))];
});

function sampleAxis(coords, low, high, reverse = false) {
  const first = coords[0];
  const last = coords.at(-1);
  const leftStep = coords.length > 1 ? coords[1] - first : 8;
  const rightStep = coords.length > 1 ? last - coords.at(-2) : 8;
  // Extra zero-valued cells fade the field beyond the sampled footprint.
  const padded = [Math.min(low, first - leftStep), ...coords, Math.max(high, last + rightStep)];
  let index = 0;
  const samples = Array.from({ length: RASTER_SIZE }, (_, i) => {
    const position = low + (high - low) * (i + 0.5) / RASTER_SIZE;
    while (position > padded[index + 1]) index++;
    const t = (position - padded[index]) / (padded[index + 1] - padded[index]);
    return { index, blend: t * t * (3 - 2 * t) };
  });
  return reverse ? samples.reverse() : samples;
}

export class ModelPanel {
  constructor(panel, toggle, close, divider) {
    this.panel = panel;
    this.toggle = toggle;
    this.divider = divider;
    this.status = panel.querySelector("#tabpfn-models-status");
    this.charts = Object.entries(MODELS).map(([id, empty]) => {
      const card = panel.querySelector(`[data-model="${id}"]`);
      return {
        id, empty, card, canvas: card.querySelector("canvas"),
        phase: card.querySelector(".model-phase"),
        time: card.querySelector(".model-time"), placeholder: card.querySelector(".model-empty"),
      };
    });
    this.snapshots = {};
    this.mode = null;
    this.heatLayer = document.createElement("canvas");
    this.heatLayer.width = this.heatLayer.height = RASTER_SIZE;
    this.heatContext = this.heatLayer.getContext("2d");
    this.heatImage = this.heatContext.createImageData(RASTER_SIZE, RASTER_SIZE);
    toggle.addEventListener("click", () => this.setOpen(panel.hidden));
    close.addEventListener("click", () => {
      this.setOpen(false);
      toggle.focus();
    });
    this.bindResize();
  }

  bindResize() {
    const { divider, panel } = this;
    const body = document.body;
    let drag = null;
    try {
      const saved = Number(localStorage.getItem(PANEL_WIDTH_KEY));
      if (saved > 0) body.style.setProperty("--model-panel-requested", `${saved}px`);
    } catch {
      // Resizing works in private or sandboxed sessions without storage.
    }
    const requestWidth = (width) => body.style.setProperty("--model-panel-requested", `${Math.max(0, width)}px`);
    const finish = (reset = false) => {
      body.classList.remove("model-panel-resizing");
      this.updateDivider();
      try {
        if (reset) localStorage.removeItem(PANEL_WIDTH_KEY);
        else localStorage.setItem(PANEL_WIDTH_KEY, String(panel.offsetWidth));
      } catch {
        // Storage is optional and does not affect the current width.
      }
      panel.dispatchEvent(new Event("resize"));
    };
    divider.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || drag) return;
      drag = { id: event.pointerId, x: event.clientX, width: panel.offsetWidth };
      divider.focus({ preventScroll: true });
      divider.setPointerCapture(event.pointerId);
      body.classList.add("model-panel-resizing");
    });
    window.addEventListener("pointermove", (event) => {
      if (drag?.id === event.pointerId) requestWidth(drag.width + drag.x - event.clientX);
    });
    window.addEventListener("pointerup", (event) => {
      if (drag?.id !== event.pointerId) return;
      requestWidth(drag.width + drag.x - event.clientX);
      drag = null;
      finish();
    });
    window.addEventListener("pointercancel", (event) => {
      if (drag?.id !== event.pointerId) return;
      drag = null;
      finish();
    });
    divider.addEventListener("keydown", (event) => {
      const width = panel.offsetWidth;
      const sizes = { ArrowLeft: width + 24, ArrowRight: width - 24, Home: 0, End: window.innerWidth };
      if (!Object.hasOwn(sizes, event.key)) return;
      event.preventDefault();
      event.stopPropagation();
      requestWidth(sizes[event.key]);
      finish();
    });
    divider.addEventListener("dblclick", () => {
      body.style.removeProperty("--model-panel-requested");
      finish(true);
    });
    window.addEventListener("resize", () => this.updateDivider());
  }

  updateDivider() {
    if (this.panel.hidden) return;
    const width = window.innerWidth;
    const max = width >= 1050 ? Math.min(width * 0.68, width - 430) : width;
    this.divider.setAttribute("aria-valuemin", String(Math.min(300, width)));
    this.divider.setAttribute("aria-valuemax", String(Math.round(max)));
    this.divider.setAttribute("aria-valuenow", String(this.panel.offsetWidth));
    this.divider.setAttribute("aria-valuetext", `${this.panel.offsetWidth} pixels wide`);
  }

  setOpen(open) {
    this.panel.hidden = !open;
    this.divider.hidden = !open;
    this.toggle.setAttribute("aria-expanded", String(open));
    document.body.classList.toggle("model-panel-open", open);
    if (open) {
      this.updateDivider();
      this.charts.forEach((chart) => this.draw(chart));
    }
  }

  updateBackend(status) {
    const previousMode = this.mode;
    this.mode = status.mode;
    const gpu = this.mode === "tabpfn-api";
    const label = gpu ? (status.error ? "Inference error · showing last predictions" : "GPU predictions")
      : "TabPFN is off. Launch with --tabpfn to see model predictions.";
    if (this.status.textContent !== label) this.status.textContent = label;
    const tracking = gpu && (status.search_phase === "track" || status.mission_state === "found");
    this.setModelPhase("search", tracking ? "Search paused · Tracking source" : "");
    this.setModelPhase("source", gpu && status.search_phase === "search" && !tracking && this.snapshots.source
      ? "Previous estimate · Searching again" : "");
    if (gpu && previousMode !== this.mode) this.setOpen(true);
    else if (previousMode !== this.mode) this.charts.forEach((chart) => this.draw(chart));
  }

  setModelPhase(id, label) {
    const chart = this.charts.find((item) => item.id === id);
    chart.card.classList.toggle("model-inactive", Boolean(label));
    chart.phase.hidden = !label;
    if (chart.phase.textContent !== label) chart.phase.textContent = label;
  }

  setWorld(world) {
    this.world = world;
    this.snapshots = {};
    const axis = (i) => [...new Set(world.grid_coords.map((point) => point[i]))].sort((a, b) => a - b);
    const xs = axis(0);
    const ys = axis(1);
    this.heatGrid = null;
    if (xs.length && ys.length) {
      const stride = xs.length + 2;
      const xIndex = new Map(xs.map((x, index) => [x, index + 1]));
      const yIndex = new Map(ys.map((y, index) => [y, (index + 1) * stride]));
      this.heatGrid = {
        stride, cells: new Float32Array(stride * (ys.length + 2)),
        cellIndices: world.grid_coords.map(([x, y]) => xIndex.get(x) + yIndex.get(y)),
        xSamples: sampleAxis(xs, world.bounds_min[0], world.bounds_max[0]),
        ySamples: sampleAxis(ys, world.bounds_min[1], world.bounds_max[1], true),
      };
    }
    this.charts.forEach((chart) => this.draw(chart));
  }

  updateBelief(belief) {
    if (belief.source_prediction_backend !== "tabpfn-4090-regressor") return;
    const incoming = {
      search: belief.tabpfn_search_map,
      source: belief.tabpfn_source_map,
      hazard: belief.hazard_observed_at >= 0 && belief.hazard_forecast_time != null
        ? { scores: belief.hazard_probabilities, sim_time: belief.hazard_observed_at, forecast: belief.hazard_forecast_time }
        : null,
    };
    for (const chart of this.charts) {
      const snapshot = incoming[chart.id];
      if (!snapshot || this.snapshots[chart.id]?.sim_time === snapshot.sim_time) continue;
      this.snapshots[chart.id] = snapshot;
      this.draw(chart);
    }
  }

  paintHeatmap(ctx, scores, left, top, width, height) {
    if (!scores?.length || !this.heatGrid) return;
    const { stride, cells, cellIndices, xSamples, ySamples } = this.heatGrid;
    cells.fill(0);
    for (let i = 0; i < Math.min(scores.length, cellIndices.length); i++) {
      const cell = cellIndices[i];
      if (Number.isFinite(scores[i])) cells[cell] = Math.max(cells[cell], scores[i]);
    }
    const data = this.heatImage.data;
    let pixel = 0;
    for (const { index: y, blend: fy } of ySamples) {
      const row = y * stride;
      for (const { index: x, blend: fx } of xSamples) {
        const a = row + x;
        const b = a + stride;
        const v0 = cells[a] + (cells[a + 1] - cells[a]) * fx;
        const v1 = cells[b] + (cells[b + 1] - cells[b]) * fx;
        const color = heatColors[Math.round(Math.min(1, v0 + (v1 - v0) * fy) * 255)];
        data.set(color, pixel);
        pixel += 4;
      }
    }
    this.heatContext.putImageData(this.heatImage, 0, 0);
    ctx.drawImage(this.heatLayer, left, top, width, height);
  }

  draw(chart) {
    if (this.panel.hidden) return;
    const { canvas, id } = chart;
    const ctx = canvas.getContext("2d");
    const snapshot = this.snapshots[id];
    chart.time.textContent = snapshot
      ? (id === "hazard" ? `For ${snapshot.forecast.toFixed(1)} s` : `At ${snapshot.sim_time.toFixed(1)} s`)
      : "";
    chart.placeholder.hidden = Boolean(snapshot);
    chart.placeholder.textContent = this.mode === "tabpfn-api" ? chart.empty : "No TabPFN prediction";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.world) return;

    const { bounds_min: low, bounds_max: high, buildings } = this.world;
    const scale = Math.min((canvas.width - 48) / (high[0] - low[0]), (canvas.height - 28) / (high[1] - low[1]));
    const width = (high[0] - low[0]) * scale;
    const height = (high[1] - low[1]) * scale;
    const left = (canvas.width - width) / 2;
    const top = (canvas.height - height) / 2;
    const project = ([x, y]) => [left + (x - low[0]) * scale, top + (high[1] - y) * scale];
    ctx.fillStyle = "#263b4f";
    ctx.fillRect(left, top, width, height);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.clip();

    const footprints = buildings.filter((building) => building.kind !== "pipe_segment").map((building) => {
      const [x, y] = project([building.min_xyz[0], building.max_xyz[1]]);
      return [x, y, (building.max_xyz[0] - building.min_xyz[0]) * scale, (building.max_xyz[1] - building.min_xyz[1]) * scale];
    });
    ctx.fillStyle = "#465d70";
    for (const bounds of footprints) ctx.fillRect(...bounds);
    this.paintHeatmap(ctx, snapshot?.scores, left, top, width, height);

    ctx.strokeStyle = "#7394ae23";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = Math.ceil(low[0] / 20) * 20; x < high[0]; x += 20) {
      const [px] = project([x, low[1]]);
      ctx.moveTo(px, top);
      ctx.lineTo(px, top + height);
    }
    for (let y = Math.ceil(low[1] / 20) * 20; y < high[1]; y += 20) {
      const [, py] = project([low[0], y]);
      ctx.moveTo(left, py);
      ctx.lineTo(left + width, py);
    }
    ctx.stroke();

    ctx.strokeStyle = "#95abc4af";
    ctx.lineWidth = 1.25;
    for (const bounds of footprints) ctx.strokeRect(...bounds);
    ctx.strokeStyle = "#b8c9dc55";
    for (const pipe of this.world.pipe_runs?.length ? this.world.pipe_runs : (this.world.pipes ?? [])) {
      ctx.beginPath();
      pipe.points.forEach((point, index) => {
        const [x, y] = project(point);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = "#62768e";
    ctx.strokeRect(left, top, width, height);
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = "#9cb3ca";
    ctx.fillText("↑ +Y", left - 37, top + 13);
    ctx.fillText("+X →", left + width + 9, top + height - 3);
    const rulerY = top + height + 7;
    ctx.fillRect(left, rulerY, 20 * scale, 1);
    ctx.fillText("20 m", left + 20 * scale + 5, rulerY + 4);
    if (snapshot?.estimate) {
      const [x, y] = project(snapshot.estimate);
      ctx.beginPath();
      ctx.moveTo(x, y - 9);
      ctx.lineTo(x + 9, y);
      ctx.lineTo(x, y + 9);
      ctx.lineTo(x - 9, y);
      ctx.closePath();
      ctx.fillStyle = "#21132f";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#eddcff";
      ctx.save();
      ctx.shadowColor = "#e7a3fa";
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.restore();
    }
  }
}
