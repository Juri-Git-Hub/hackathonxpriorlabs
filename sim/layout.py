"""Seeded refinery layouts and a connected, obstacle-aware process pipe network.

Geometry is generated once in Python and shared with physics, planning and rendering.
Pipe routing is infrastructure geometry, independent of the drone search policy.
"""
from __future__ import annotations

import heapq

import numpy as np

from shared.schemas import BuildingBox, FacilityAsset, PipeRun

# Distinct procedural facility archetypes selected deterministically from seed.
# Each archetype assigns cohesive Blender GLB structures, proportions, pipe
# services, and atmosphere theme while keeping physics & MST connectivity exact.
_ARCHETYPES: tuple[dict, ...] = (
    {
        "name": "petrochemical",
        "theme": "refinery_dusk",
        "route_z": 4.5,
        "models": {
            "building": ["building", "compressor", "pump_house"],
            "storage_tank": ["fractionator", "horton_sphere", "storage_tank"],
            "pipe_rack": ["pipe_rack"],
        },
        "pipe_materials": ("gas", "steel", "coolant"),
    },
    {
        "name": "cryogenic_depot",
        "theme": "arctic_frost",
        "route_z": 4.2,
        "models": {
            "building": ["pump_house", "metering_skid", "building"],
            "storage_tank": ["cryogenic_vessel", "horton_sphere"],
            "pipe_rack": ["transfer_gantry", "pipe_rack"],
        },
        "pipe_materials": ("steel", "gas", "coolant"),
    },
    {
        "name": "desert_booster",
        "theme": "desert_sun",
        "route_z": 5.0,
        "models": {
            "building": ["metering_skid", "receiver_skid", "compressor_train", "compressor"],
            "storage_tank": ["horton_sphere", "exhaust_stack", "storage_tank"],
            "pipe_rack": ["pipe_rack", "transfer_gantry"],
        },
        "pipe_materials": ("gas", "gas", "steel"),
    },
    {
        "name": "thermal_cooling",
        "theme": "overcast_steam",
        "route_z": 4.8,
        "models": {
            "building": ["cooling_cell", "pump_house", "building"],
            "storage_tank": ["hyperbolic_tower", "exhaust_stack", "storage_tank"],
            "pipe_rack": ["pipe_rack"],
        },
        "pipe_materials": ("coolant", "steel", "coolant"),
    },
    {
        "name": "alpine_compressor",
        "theme": "alpine_steel",
        "route_z": 4.6,
        "models": {
            "building": ["compressor_train", "compressor", "receiver_skid"],
            "storage_tank": ["horton_sphere", "fractionator", "cryogenic_vessel"],
            "pipe_rack": ["transfer_gantry", "pipe_rack"],
        },
        "pipe_materials": ("gas", "steel", "gas"),
    },
)

# Realistic height / footprint multipliers by Blender model type
_MODEL_SIZE_HINT: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    # model -> ((xy_scale_min, xy_scale_max), (z_min, z_max))
    "fractionator": ((0.70, 0.92), (17.0, 23.0)),
    "horton_sphere": ((0.78, 0.96), (13.0, 17.5)),
    "cryogenic_vessel": ((0.62, 0.85), (16.5, 22.0)),
    "hyperbolic_tower": ((0.88, 1.08), (19.0, 25.0)),
    "exhaust_stack": ((0.18, 0.28), (19.0, 25.0)),
    "storage_tank": ((0.85, 1.05), (12.0, 16.5)),
    "metering_skid": ((0.75, 0.95), (5.5, 7.5)),
    "receiver_skid": ((0.78, 0.98), (6.0, 8.0)),
    "compressor_train": ((0.72, 0.92), (4.8, 6.8)),
    "cooling_cell": ((0.82, 1.02), (10.5, 14.0)),
    "pump_house": ((0.85, 1.08), (8.5, 11.5)),
    "compressor": ((0.88, 1.12), (9.0, 12.0)),
    "building": ((0.85, 1.10), (10.0, 13.5)),
    "pipe_rack": ((0.85, 1.10), (6.5, 9.0)),
    "transfer_gantry": ((0.85, 1.10), (7.5, 10.5)),
}

_SERVICE_BY_MAT: dict[str, tuple[str, float, float]] = {
    "gas": ("High-Pressure Gas", 18.0, 0.34),
    "steel": ("Process Feed", 12.5, 0.28),
    "coolant": ("Coolant Loop", 6.2, 0.30),
}


def layout_theme(seed: int) -> str:
    """Return deterministic visual atmosphere/ground theme for a generated seed."""
    return _ARCHETYPES[int(seed) % len(_ARCHETYPES)]["theme"]


def _free_xy(point, buildings, margin=0.0):
    x, y = point
    return all(not (b.min_xyz[0]-margin < x < b.max_xyz[0]+margin and
                        b.min_xyz[1]-margin < y < b.max_xyz[1]+margin) for b in buildings)


def _pipe_route(start, goal, buildings):
    """Orthogonal A* around expanded footprints; nodes come from obstacle edges."""
    boxes = [(b.min_xyz[0]-.9, b.min_xyz[1]-.9,
              b.max_xyz[0]+.9, b.max_xyz[1]+.9) for b in buildings]
    xs = sorted({-54., 54., start[0], goal[0], *(v for b in boxes for v in (b[0], b[2]))})
    ys = sorted({-54., 54., start[1], goal[1], *(v for b in boxes for v in (b[1], b[3]))})
    src = (xs.index(start[0]), ys.index(start[1]))
    dst = (xs.index(goal[0]), ys.index(goal[1]))

    def visible(a, b):
        ax, ay = xs[a[0]], ys[a[1]]
        bx, by = xs[b[0]], ys[b[1]]
        for x0, y0, x1, y1 in boxes:
            if a[0] == b[0]:
                if x0+1e-8 < ax < x1-1e-8 and max(ay, by) > y0+1e-8 and min(ay, by) < y1-1e-8:
                    return False
            elif y0+1e-8 < ay < y1-1e-8 and max(ax, bx) > x0+1e-8 and min(ax, bx) < x1-1e-8:
                return False
        return True

    def distance(a, b):
        return abs(xs[a[0]]-xs[b[0]])+abs(ys[a[1]]-ys[b[1]])

    queue = [(distance(src, dst), src)]
    cost, previous = {src: 0.}, {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == dst:
            path = [current]
            while path[-1] != src:
                path.append(previous[path[-1]])
            return [(xs[i], ys[j]) for i, j in reversed(path)]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (current[0]+di, current[1]+dj)
            if not (0 <= nxt[0] < len(xs) and 0 <= nxt[1] < len(ys)) or not visible(current, nxt):
                continue
            value = cost[current]+distance(current, nxt)
            if value < cost.get(nxt, float('inf')):
                cost[nxt], previous[nxt] = value, current
                heapq.heappush(queue, (value+distance(nxt, dst), nxt))
    raise ValueError('Could not connect the generated facility pipe network')


def generate_layout(building_templates, asset_templates, seed):
    """Relocate equipment with seed-driven archetypes, then connect all nodes with an MST.

    Buildings keep their identities and kinds while drawing varied Blender structures
    and proportions from the selected industrial archetype. Tanks retain square
    footprints so the renderer's cylinders/spheres and connection ports agree.
    """
    rng = np.random.default_rng(seed)
    archetype = _ARCHETYPES[int(seed) % len(_ARCHETYPES)]
    route_z = float(archetype["route_z"])
    used_models: set[str] = set()

    buildings = []
    for template in building_templates:
        pool = archetype["models"].get(template.kind, [template.kind])
        # Prefer unused models first so structures in a generated world look distinct
        unused = [m for m in pool if m not in used_models]
        choices = unused if unused else pool
        chosen = str(choices[int(rng.integers(len(choices)))])
        used_models.add(chosen)
        model_field = None if chosen in ("building", "storage_tank", "pipe_rack") else chosen

        base_size = np.subtract(template.max_xyz, template.min_xyz)
        (xy_lo, xy_hi), (z_lo, z_hi) = _MODEL_SIZE_HINT.get(
            chosen, ((0.85, 1.10), (float(base_size[2]) * 0.85, float(base_size[2]) * 1.10))
        )
        xy_scale = rng.uniform(xy_lo, xy_hi)
        size = np.array([
            base_size[0] * xy_scale,
            base_size[1] * xy_scale,
            rng.uniform(z_lo, z_hi),
        ], dtype=np.float64)
        if template.kind == "storage_tank":
            side = float(min(size[0], size[1]))
            size[0] = side
            size[1] = side
        elif rng.integers(2):
            size[:2] = size[1::-1]

        for _ in range(5000):
            center = rng.uniform(-44+size[:2]/2, 44-size[:2]/2)
            lo, hi = center-size[:2]/2, center+size[:2]/2
            separated = all(np.any(hi+7 < b.min_xyz[:2]) or np.any(lo-7 > b.max_xyz[:2]) for b in buildings)
            # Reserve all supported drone launch pads, including their safety margin.
            spawn_clear = all(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0)) >= 10
                              for p in (np.array([-48., y]) for y in (-45., -40., -35.)))
            if separated and spawn_clear:
                buildings.append(BuildingBox(id=template.id, kind=template.kind, model=model_field,
                    min_xyz=(float(lo[0]), float(lo[1]), 0.),
                    max_xyz=(float(hi[0]), float(hi[1]), float(size[2]))))
                break
        else:
            raise ValueError('Could not place buildings with safe separation')

    # Each structure has a port on its face pointing toward the facility center.
    centroid = np.mean([(np.array(b.min_xyz)+b.max_xyz)/2 for b in buildings], axis=0)
    nodes = []  # (id, physical attachment point, exterior routing point)
    for b in buildings:
        center = (np.array(b.min_xyz)+b.max_xyz)/2
        axis = int(np.argmax(np.abs(centroid[:2]-center[:2])))
        normal = np.zeros(3); normal[axis] = 1 if centroid[axis] >= center[axis] else -1
        port = center.copy()
        port[2] = min(route_z, max(2.5, b.max_xyz[2] - 1.5))
        port[axis] = (b.max_xyz if normal[axis] > 0 else b.min_xyz)[axis]
        nodes.append((b.id, port, port+normal*2.5))

    assets = []
    for i, template in enumerate(asset_templates):
        if i < len(buildings):
            point = nodes[i][2].copy()
        else:
            for _ in range(5000):
                point = np.r_[rng.uniform(-45, 45, 2), rng.uniform(3, 7)]
                if _free_xy(point[:2], buildings, margin=2.5) and all(np.linalg.norm(point-np.array(a.pos)) > 4 for a in assets):
                    break
            else:
                raise ValueError('Could not place connected facility assets')
        asset = template.model_copy(update={'pos': tuple(map(float, point))})
        assets.append(asset)
        nodes.append((asset.id, point, point))

    # Prim's MST connects the entire facility, with no isolated pipes or assets.
    connected = {0}
    edges = []
    while len(connected) < len(nodes):
        _, a, b = min((np.linalg.norm(nodes[a][2]-nodes[b][2]), a, b)
                     for a in sorted(connected) for b in range(len(nodes)) if b not in connected)
        connected.add(b); edges.append((a, b))

    mat_cycle = archetype["pipe_materials"]
    runs = []
    for i, (a, b) in enumerate(edges):
        aid, ap, outside_a = nodes[a]
        bid, bp, outside_b = nodes[b]
        path = [ap, outside_a, np.r_[outside_a[:2], route_z]]
        path += [np.r_[p, route_z] for p in _pipe_route(tuple(outside_a[:2]), tuple(outside_b[:2]), buildings)]
        path += [outside_b, bp]
        # Remove duplicate vertices and straight-line intermediate points.
        points = []
        for point in path:
            if points and np.allclose(point, points[-1]):
                continue
            while len(points) >= 2 and np.linalg.norm(np.cross(np.subtract(points[-1], points[-2]), np.subtract(point, points[-1]))) < 1e-8:
                points.pop()
            points.append(tuple(map(float, point)))
        p_mat = mat_cycle[i % len(mat_cycle)]
        service, pressure, base_r = _SERVICE_BY_MAT[p_mat]
        runs.append(PipeRun(
            id=f'pipe_{i+1:02d}',
            name=f'{archetype["name"].replace("_", " ").title()} Line {i+1:02d}',
            service=service,
            pressure_bar=round(pressure + float(rng.uniform(-1.5, 2.0)), 1),
            radius=round(base_r + float(rng.uniform(-0.03, 0.05)), 2),
            material=p_mat,
            from_id=aid,
            to_id=bid,
            points=points,
        ))
    return buildings, assets, runs
