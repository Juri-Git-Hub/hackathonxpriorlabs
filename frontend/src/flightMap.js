const TRAIL_COLORS = { tabpfn: "#22d3ee", "tabpfn-2": "#fb923c", "tabpfn-3": "#c4b5fd" };
const ESTIMATE_COLOR = "#fbbf24";
const LEAK_COLOR = "#fb7185";

function polyline(ctx, points, project) {
  ctx.beginPath();
  points.forEach((point, index) => {
    const [x, y] = project(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

export class FlightMap {
  constructor(canvas, estimateLabel) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.estimateLabel = estimateLabel;
    this.setWorld(null);
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  setWorld(world) {
    this.world = world;
    this.time = -Infinity;
    this.drones = [];
    this.estimateTime = -Infinity;
    this.confirmed = false;
    this.leaks = [];
    this.trails = new Map();
    this.updateBelief(null);
  }

  clearTrail() {
    this.trails = new Map();
    this.draw();
  }

  updateTick(tick) {
    if (tick.sim_time < this.time) this.setWorld(this.world);
    this.time = tick.sim_time;
    this.leaks = tick.leaks ?? [];
    this.drones = tick.drones;
    for (const drone of this.drones) {
      const trail = this.trails.get(drone.id) ?? [];
      const point = drone.pos;
      const last = trail.at(-1);
      if (!last || Math.hypot(point[0] - last[0], point[1] - last[1]) >= 0.25) {
        trail.push(point);
      }
      this.trails.set(drone.id, trail.length > 5000 ? trail.filter((_, index) => index % 2 === 0) : trail);
    }
    if (tick.mission_report && !this.confirmed) {
      this.updateBelief(tick.mission_report);
      this.confirmed = true;
    } else {
      this.draw();
    }
  }

  updateBelief(belief) {
    const estimateTime = belief?.sim_time ?? belief?.elapsed_sim_seconds ?? -Infinity;
    if (belief && (this.confirmed || belief.source_evidence === false ||
        estimateTime < this.estimateTime)) return;
    this.estimateTime = estimateTime;
    const point = belief?.estimated_source;
    const droneLabel = belief?.drone_id ? `${belief.drone_id.toUpperCase()} · ` : "";
    this.estimate = point?.length === 3 && point.every(Number.isFinite) ? point : null;
    this.estimateLabel.textContent = this.estimate
      ? `${droneLabel}X ${point[0].toFixed(1)} m · Y ${point[1].toFixed(1)} m · Alt ${point[2].toFixed(1)} m`
      : "Waiting for a leak estimate";
    this.draw();
  }

  draw() {
    const { canvas, ctx, world } = this;
    if (!ctx || !canvas.getClientRects().length) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!world || Math.min(width, height) < 60) return;

    const [minX, minY] = world.bounds_min ?? [-60, -60];
    const [maxX, maxY] = world.bounds_max ?? [60, 60];
    const scale = Math.min((width - 52) / (maxX - minX), (height - 52) / (maxY - minY));
    const left = (width - (maxX - minX) * scale) / 2;
    const top = (height - (maxY - minY) * scale) / 2;
    const project = ([x, y]) => [left + (x - minX) * scale, top + (maxY - y) * scale];
    const plotWidth = (maxX - minX) * scale;
    const plotHeight = (maxY - minY) * scale;

    ctx.fillStyle = "#162537";
    ctx.fillRect(left, top, plotWidth, plotHeight);
    ctx.strokeStyle = "#26394e";
    ctx.lineWidth = 1;
    for (let x = Math.ceil(minX / 20) * 20; x < maxX; x += 20) {
      polyline(ctx, [[x, minY], [x, maxY]], project);
    }
    for (let y = Math.ceil(minY / 20) * 20; y < maxY; y += 20) {
      polyline(ctx, [[minX, y], [maxX, y]], project);
    }
    ctx.strokeStyle = "#597086";
    ctx.strokeRect(left, top, plotWidth, plotHeight);

    ctx.fillStyle = "#35495e";
    ctx.strokeStyle = "#5b7085";
    for (const building of world.buildings ?? []) {
      if (building.kind === "pipe_segment") continue;
      const [x, y] = project([building.min_xyz[0], building.max_xyz[1]]);
      const w = (building.max_xyz[0] - building.min_xyz[0]) * scale;
      const h = (building.max_xyz[1] - building.min_xyz[1]) * scale;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }
    ctx.strokeStyle = "#8a9aac";
    for (const pipe of world.pipe_runs?.length ? world.pipe_runs : (world.pipes ?? [])) {
      polyline(ctx, pipe.points, project);
    }

    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = 2;
    for (const [id, trail] of this.trails) {
      ctx.strokeStyle = TRAIL_COLORS[id] ?? "#e2e8f0";
      polyline(ctx, trail, project);
    }
    ctx.strokeStyle = LEAK_COLOR;
    for (const leak of this.leaks) {
      const [x, y] = project(leak.pos);
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.stroke();
    }
    for (const drone of this.drones) {
      const [x, y] = project(drone.pos);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#e6fbff";
      ctx.fill();
      ctx.strokeStyle = TRAIL_COLORS[drone.id] ?? "#e2e8f0";
      ctx.stroke();
    }
    if (this.estimate) {
      const [x, y] = project(this.estimate);
      ctx.beginPath();
      ctx.moveTo(x, y - 9);
      ctx.lineTo(x + 9, y);
      ctx.lineTo(x, y + 9);
      ctx.lineTo(x - 9, y);
      ctx.closePath();
      ctx.fillStyle = "#251c10";
      ctx.fill();
      ctx.strokeStyle = ESTIMATE_COLOR;
      ctx.stroke();
    }

    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = "#a6b8cc";
    ctx.fillText("↑ +Y", left, top - 8);
    ctx.fillText("+X →", left + plotWidth - 29, top + plotHeight + 17);
    ctx.strokeStyle = "#a6b8cc";
    ctx.lineWidth = 1;
    const scaleY = top + plotHeight + 13;
    polyline(ctx, [[left, scaleY], [left + 10 * scale, scaleY]], (point) => point);
    ctx.fillText("10 m", left + 10 * scale + 6, scaleY + 4);
  }
}
