"""Procedural industrial 3D GLB asset generator using Blender (headless).

Generates lightweight, PBR-material GLB assets into `frontend/public/models/`:
  - storage_tank.glb        (normalized: x,y in [-0.5, 0.5], z in [0, 1] -> Y-up in glTF)
  - industrial_building.glb (normalized: x,y in [-0.5, 0.5], z in [0, 1])
  - pipe_rack.glb           (normalized: x,y in [-0.5, 0.5], z in [0, 1])
  - valve_assembly.glb      (real-world ~1.2m industrial flanged gate valve)
  - quadrotor.glb           (realistic Matrice-style inspection drone w/ Rotor_0..3 + AccentLED)

Run:
    blender --background --python scripts/generate_assets.py
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy


OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "models"


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_mat(
    name: str,
    color: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    emission: tuple[float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None and emission_strength > 0.0:
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def assign_mat(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def set_smooth(obj: bpy.types.Object) -> None:
    if obj.type == "MESH" and obj.data:
        for poly in obj.data.polygons:
            poly.use_smooth = True


def add_bevel(obj: bpy.types.Object, width: float = 0.015, segments: int = 2) -> None:
    mod = obj.modifiers.new(name="Bevel", type="BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"


def add_box(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bevel: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    assign_mat(obj, mat)
    if bevel > 0.0:
        add_bevel(obj, width=bevel, segments=2)
    return obj


def add_cyl(
    name: str,
    loc: tuple[float, float, float],
    radius: float,
    depth: float,
    mat: bpy.types.Material,
    vertices: int = 32,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smooth: bool = True,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name
    assign_mat(obj, mat)
    if smooth:
        set_smooth(obj)
    return obj


def export_glb(filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = str(OUT_DIR / filename)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=False,
        export_yup=True,
        export_materials="EXPORT",
    )
    size_kb = os.path.getsize(path) / 1024.0
    print(f"Exported {filename} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 1. Storage Tank (normalized to [-0.5, 0.5] x [-0.5, 0.5] x [0, 1.0])
# ---------------------------------------------------------------------------
def build_storage_tank() -> None:
    clear_scene()
    mat_concrete = make_mat("Concrete", (0.28, 0.30, 0.32), metallic=0.0, roughness=0.92)
    mat_tank = make_mat("TankWhiteSteel", (0.72, 0.75, 0.78), metallic=0.35, roughness=0.42)
    mat_band = make_mat("TankBand", (0.36, 0.42, 0.48), metallic=0.55, roughness=0.38)
    mat_accent = make_mat("SafetyStripe", (0.85, 0.48, 0.08), metallic=0.1, roughness=0.45)
    mat_yellow = make_mat("SafetyYellow", (0.90, 0.68, 0.08), metallic=0.15, roughness=0.40)
    mat_dark_steel = make_mat("DarkSteel", (0.18, 0.21, 0.25), metallic=0.75, roughness=0.35)

    # Concrete plinth base (smooth 64-seg cylinder -> zero blocky facets!)
    add_cyl("Plinth", (0, 0, 0.03), radius=0.49, depth=0.06, mat=mat_concrete, vertices=64)

    # Main cylindrical tank shell
    add_cyl("TankBody", (0, 0, 0.46), radius=0.44, depth=0.80, mat=mat_tank, vertices=64)

    # Circumferential stiffening / weld rings
    for z in (0.18, 0.38, 0.58, 0.78):
        add_cyl(f"Ring_{z:.2f}", (0, 0, z), radius=0.450, depth=0.016, mat=mat_band, vertices=64)

    # Painted industrial hazard band near top
    add_cyl("HazardBand", (0, 0, 0.70), radius=0.444, depth=0.065, mat=mat_accent, vertices=64)

    # Ground-level bolted manway access hatch on +Y face
    add_cyl("ManwayNeck", (0, 0.445, 0.14), radius=0.055, depth=0.06, mat=mat_dark_steel, vertices=24, rot=(math.pi / 2, 0, 0))
    add_cyl("ManwayCover", (0, 0.478, 0.14), radius=0.068, depth=0.016, mat=mat_band, vertices=24, rot=(math.pi / 2, 0, 0))

    # Shallow smooth domed roof
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=16, radius=0.44, location=(0, 0, 0.86))
    dome = bpy.context.active_object
    dome.name = "TankDome"
    dome.scale = (1.0, 1.0, 0.24)
    assign_mat(dome, mat_tank)
    set_smooth(dome)

    # Top pressure-relief vent cap & hatch
    add_cyl("TopVent", (0, 0, 0.96), radius=0.07, depth=0.07, mat=mat_dark_steel, vertices=16)
    add_cyl("VentCap", (0, 0, 0.995), radius=0.095, depth=0.015, mat=mat_band, vertices=16)

    # Perimeter rooftop safety handrail (posts + 2 rings)
    rail_r = 0.415
    for i in range(16):
        ang = 2.0 * math.pi * i / 16.0
        px = rail_r * math.cos(ang)
        py = rail_r * math.sin(ang)
        add_cyl(f"RailPost_{i}", (px, py, 0.905), radius=0.006, depth=0.09, mat=mat_yellow, vertices=6)

    for rz in (0.92, 0.95):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=rail_r,
            minor_radius=0.006,
            major_segments=32,
            minor_segments=6,
            location=(0, 0, rz),
        )
        t = bpy.context.active_object
        t.name = f"RailRing_{rz:.2f}"
        assign_mat(t, mat_yellow)

    # Vertical caged access ladder on +X side
    for dy in (-0.032, 0.032):
        add_box(f"LadderRail_{dy:.3f}", (0.465, dy, 0.48), (0.012, 0.012, 0.86), mat_yellow)
    for k in range(14):
        rz = 0.08 + k * 0.06
        add_box(f"Rung_{k}", (0.465, 0.0, rz), (0.010, 0.065, 0.010), mat_yellow)

    # Side intake manifold pipe & elbow on -Y side
    add_cyl("IntakePipeV", (0.12, -0.465, 0.35), radius=0.032, depth=0.60, mat=mat_dark_steel, vertices=14)
    add_cyl("IntakePipeH", (0.12, -0.44, 0.64), radius=0.032, depth=0.08, mat=mat_dark_steel, vertices=14, rot=(math.pi / 2, 0, 0))
    add_cyl("IntakeFlange", (0.12, -0.465, 0.12), radius=0.052, depth=0.022, mat=mat_accent, vertices=14)

    export_glb("storage_tank.glb")


# ---------------------------------------------------------------------------
# 2. Industrial Building (normalized to [-0.5, 0.5] x [-0.5, 0.5] x [0, 1.0])
# ---------------------------------------------------------------------------
def build_industrial_building() -> None:
    clear_scene()
    mat_concrete = make_mat("ConcretePad", (0.22, 0.24, 0.27), metallic=0.0, roughness=0.94)
    mat_wall = make_mat("SteelCladding", (0.23, 0.28, 0.34), metallic=0.45, roughness=0.48)
    mat_trim = make_mat("DarkTrim", (0.11, 0.13, 0.16), metallic=0.72, roughness=0.36)
    mat_roof = make_mat("RoofMembrane", (0.16, 0.18, 0.21), metallic=0.15, roughness=0.82)
    mat_door = make_mat("RollupDoor", (0.36, 0.40, 0.44), metallic=0.60, roughness=0.42)
    mat_window = make_mat(
        "ControlWindow",
        (1.0, 0.80, 0.42),
        metallic=0.05,
        roughness=0.12,
        emission=(1.0, 0.72, 0.28),
        emission_strength=4.5,
    )
    mat_yellow = make_mat("BollardYellow", (0.92, 0.68, 0.06), metallic=0.15, roughness=0.38)
    mat_red_beacon = make_mat(
        "RoofBeacon",
        (1.0, 0.15, 0.10),
        metallic=0.1,
        roughness=0.2,
        emission=(1.0, 0.18, 0.12),
        emission_strength=5.0,
    )

    # Concrete foundation slab
    add_box("Foundation", (0, 0, 0.03), (1.0, 1.0, 0.06), mat_concrete)

    # Main structural shell
    add_box("MainShell", (0, 0, 0.45), (0.94, 0.94, 0.78), mat_wall)

    # Vertical structural steel pilasters along all 4 sides
    for x in (-0.47, -0.16, 0.16, 0.47):
        for y in (-0.475, 0.475):
            add_box(f"ColXY_{x:.2f}_{y:.2f}", (x, y, 0.45), (0.045, 0.035, 0.78), mat_trim)
    for y in (-0.16, 0.16):
        for x in (-0.475, 0.475):
            add_box(f"ColYX_{x:.2f}_{y:.2f}", (x, y, 0.45), (0.035, 0.045, 0.78), mat_trim)

    # Roof slab & parapet walls
    add_box("RoofDeck", (0, 0, 0.845), (0.98, 0.98, 0.035), mat_roof)
    add_box("ParapetN", (0, 0.48, 0.88), (0.98, 0.03, 0.05), mat_trim)
    add_box("ParapetS", (0, -0.48, 0.88), (0.98, 0.03, 0.05), mat_trim)
    add_box("ParapetE", (0.48, 0, 0.88), (0.03, 0.94, 0.05), mat_trim)
    add_box("ParapetW", (-0.48, 0, 0.88), (0.03, 0.94, 0.05), mat_trim)

    # Rooftop HVAC condenser units, fan shrouds, conduit pipes & exhaust stacks
    add_box("HVAC_1", (-0.20, 0.14, 0.91), (0.28, 0.22, 0.10), mat_door)
    add_cyl("HVACFan_1a", (-0.25, 0.14, 0.965), radius=0.055, depth=0.02, mat=mat_trim, vertices=14)
    add_cyl("HVACFan_1b", (-0.14, 0.14, 0.965), radius=0.055, depth=0.02, mat=mat_trim, vertices=14)

    add_box("HVAC_2", (0.18, -0.14, 0.905), (0.24, 0.20, 0.09), mat_door)
    add_cyl("HVACFan_2", (0.18, -0.14, 0.955), radius=0.06, depth=0.02, mat=mat_trim, vertices=14)

    # Rooftop pipe conduits connecting HVAC to exhaust stacks
    add_cyl("RoofConduit_1", (0.0, 0.14, 0.89), radius=0.018, depth=0.42, mat=mat_trim, vertices=10, rot=(0, math.pi / 2, 0))
    add_cyl("RoofConduit_2", (0.18, 0.05, 0.89), radius=0.016, depth=0.32, mat=mat_trim, vertices=10, rot=(math.pi / 2, 0, 0))
    add_box("RoofHatch", (-0.26, -0.22, 0.88), (0.14, 0.14, 0.05), mat_trim)

    add_cyl("ExhaustStack_1", (0.28, 0.24, 0.94), radius=0.042, depth=0.16, mat=mat_trim, vertices=14)
    add_cyl("ExhaustCap_1", (0.28, 0.24, 1.02), radius=0.055, depth=0.015, mat=mat_door, vertices=14)
    add_cyl("ExhaustStack_2", (0.16, 0.24, 0.925), radius=0.032, depth=0.13, mat=mat_trim, vertices=14)

    # Industrial rollup bay doors + rain canopy awnings on -Y and +Y faces
    for y_sign in (-1.0, 1.0):
        add_box(f"BayDoorFrame_{y_sign}", (0.0, y_sign * 0.476, 0.25), (0.30, 0.025, 0.40), mat_trim)
        add_box(f"BayDoor_{y_sign}", (0.0, y_sign * 0.480, 0.24), (0.25, 0.016, 0.36), mat_door)
        # Overhead steel rain canopy awning
        add_box(f"Awning_{y_sign}", (0.0, y_sign * 0.515, 0.47), (0.36, 0.09, 0.022), mat_trim)
        # Yellow safety bollards flanking the bay door
        for bx in (-0.17, 0.17):
            add_cyl(
                f"Bollard_{y_sign}_{bx}",
                (bx, y_sign * 0.51, 0.11),
                radius=0.016,
                depth=0.14,
                mat=mat_yellow,
                vertices=10,
            )

    # Glowing control-room window strip + lower intake louver grilles on +X and -X faces
    for x_sign in (-1.0, 1.0):
        add_box(f"WinFrame_{x_sign}", (x_sign * 0.476, 0.0, 0.62), (0.028, 0.60, 0.16), mat_trim)
        add_box(f"WinGlass_{x_sign}", (x_sign * 0.483, 0.0, 0.62), (0.016, 0.54, 0.11), mat_window)
        add_box(f"Louver_{x_sign}", (x_sign * 0.478, -0.24, 0.26), (0.024, 0.22, 0.18), mat_trim)
        add_box(f"ElecPanel_{x_sign}", (x_sign * 0.482, 0.26, 0.26), (0.032, 0.14, 0.20), mat_door)

    # Rooftop comms mast & red aviation beacon
    add_cyl("CommsMast", (-0.36, -0.34, 0.95), radius=0.012, depth=0.18, mat=mat_trim, vertices=8)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=8, radius=0.024, location=(-0.36, -0.34, 1.04))
    bcn = bpy.context.active_object
    bcn.name = "AviationBeacon"
    assign_mat(bcn, mat_red_beacon)

    export_glb("industrial_building.glb")


# ---------------------------------------------------------------------------
# 3. Pipe Rack Gantry (normalized to [-0.5, 0.5] x [-0.5, 0.5] x [0, 1.0])
# ---------------------------------------------------------------------------
def build_pipe_rack() -> None:
    clear_scene()
    mat_steel = make_mat("GantrySteel", (0.24, 0.28, 0.33), metallic=0.72, roughness=0.40)
    mat_pipe_silver = make_mat("PipeSilver", (0.70, 0.74, 0.78), metallic=0.82, roughness=0.28)
    mat_pipe_yellow = make_mat("PipeGasYellow", (0.88, 0.66, 0.08), metallic=0.25, roughness=0.38)
    mat_pipe_blue = make_mat("PipeCoolantBlue", (0.16, 0.45, 0.72), metallic=0.35, roughness=0.38)
    mat_flange = make_mat("PipeFlange", (0.42, 0.46, 0.50), metallic=0.80, roughness=0.32)
    mat_grate = make_mat("CatwalkGrate", (0.18, 0.20, 0.24), metallic=0.65, roughness=0.55)

    # 5 Steel portal frames spaced along X
    frame_xs = (-0.46, -0.23, 0.0, 0.23, 0.46)
    for i, fx in enumerate(frame_xs):
        # Vertical H-beam columns
        add_box(f"ColL_{i}", (fx, -0.44, 0.46), (0.024, 0.05, 0.92), mat_steel)
        add_box(f"ColR_{i}", (fx, 0.44, 0.46), (0.024, 0.05, 0.92), mat_steel)
        # Concrete footings
        add_box(f"FootL_{i}", (fx, -0.44, 0.03), (0.045, 0.09, 0.06), mat_grate)
        add_box(f"FootR_{i}", (fx, 0.44, 0.03), (0.045, 0.09, 0.06), mat_grate)
        # Horizontal cross-beams (lower pipe tier + upper pipe tier + top catwalk)
        add_box(f"Beam1_{i}", (fx, 0.0, 0.45), (0.022, 0.92, 0.045), mat_steel)
        add_box(f"Beam2_{i}", (fx, 0.0, 0.74), (0.022, 0.92, 0.045), mat_steel)
        add_box(f"BeamTop_{i}", (fx, 0.0, 0.92), (0.022, 0.92, 0.035), mat_steel)

    # Longitudinal stringers along X
    for y in (-0.44, 0.44):
        for z in (0.45, 0.74, 0.92):
            add_box(f"Stringer_{y:.2f}_{z:.2f}", (0.0, y, z), (0.96, 0.028, 0.032), mat_steel)

    # Top catwalk deck + safety yellow handrails
    add_box("CatwalkDeck", (0.0, 0.0, 0.945), (0.96, 0.36, 0.018), mat_grate)
    for y in (-0.18, 0.18):
        add_box(f"CatRail_{y:.2f}", (0.0, y, 0.99), (0.96, 0.012, 0.012), mat_pipe_yellow)

    # Multi-tier industrial pipelines running along X (rotated 90 deg around Y)
    pipes = [
        # (y, z, radius, material)
        (-0.30, 0.54, 0.055, mat_pipe_silver),
        (-0.10, 0.55, 0.065, mat_pipe_yellow),
        (0.12, 0.54, 0.050, mat_pipe_blue),
        (0.30, 0.53, 0.045, mat_pipe_silver),
        (-0.26, 0.82, 0.048, mat_pipe_yellow),
        (0.00, 0.82, 0.055, mat_pipe_silver),
        (0.26, 0.82, 0.048, mat_pipe_blue),
    ]
    for idx, (py, pz, rad, pmat) in enumerate(pipes):
        add_cyl(
            f"Pipe_{idx}",
            (0.0, py, pz),
            radius=rad,
            depth=1.0,
            mat=pmat,
            vertices=16,
            rot=(0.0, math.pi / 2.0, 0.0),
        )
        # Flanged couplings along each pipe
        for fx in (-0.34, 0.0, 0.34):
            add_cyl(
                f"Flange_{idx}_{fx:.2f}",
                (fx, py, pz),
                radius=rad * 1.38,
                depth=0.018,
                mat=mat_flange,
                vertices=14,
                rot=(0.0, math.pi / 2.0, 0.0),
            )

    export_glb("pipe_rack.glb")


# ---------------------------------------------------------------------------
# 4. Industrial Valve Assembly (~1.2m real-world scale)
# ---------------------------------------------------------------------------
def build_valve_assembly() -> None:
    clear_scene()
    mat_body = make_mat("ValveCastIron", (0.22, 0.27, 0.34), metallic=0.75, roughness=0.40)
    mat_flange = make_mat("ValveFlange", (0.52, 0.56, 0.60), metallic=0.85, roughness=0.30)
    mat_wheel = make_mat("ValveRedWheel", (0.85, 0.14, 0.12), metallic=0.25, roughness=0.35)
    mat_stem = make_mat("BrassStem", (0.82, 0.64, 0.22), metallic=0.85, roughness=0.25)

    # Horizontal valve body & twin flanges
    add_cyl("ValveBody", (0, 0, 0), radius=0.22, depth=0.90, mat=mat_body, vertices=16, rot=(0, math.pi / 2, 0))
    add_cyl("FlangeL", (-0.42, 0, 0), radius=0.36, depth=0.08, mat=mat_flange, vertices=16, rot=(0, math.pi / 2, 0))
    add_cyl("FlangeR", (0.42, 0, 0), radius=0.36, depth=0.08, mat=mat_flange, vertices=16, rot=(0, math.pi / 2, 0))

    # Bonnet & stem
    add_cyl("Bonnet", (0, 0, 0.30), radius=0.16, depth=0.45, mat=mat_body, vertices=14)
    add_cyl("Stem", (0, 0, 0.62), radius=0.05, depth=0.32, mat=mat_stem, vertices=12)

    # Handwheel torus + spokes
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.36, minor_radius=0.045, major_segments=20, minor_segments=8, location=(0, 0, 0.78)
    )
    wheel = bpy.context.active_object
    wheel.name = "Handwheel"
    assign_mat(wheel, mat_wheel)
    add_box("Spoke1", (0, 0, 0.78), (0.68, 0.05, 0.045), mat_wheel)
    add_box("Spoke2", (0, 0, 0.78), (0.05, 0.68, 0.045), mat_wheel)

    export_glb("valve_assembly.glb")


# ---------------------------------------------------------------------------
# 5. Industrial Inspection Quadrotor (Matrice 350 RTK-style, detailed PBR)
# ---------------------------------------------------------------------------
def build_quadrotor() -> None:
    clear_scene()
    mat_carbon = make_mat("CarbonFiber", (0.07, 0.08, 0.10), metallic=0.38, roughness=0.28)
    mat_shell = make_mat("DroneShell", (0.24, 0.28, 0.34), metallic=0.52, roughness=0.26)
    mat_alu = make_mat("Aluminum", (0.78, 0.82, 0.86), metallic=0.90, roughness=0.20)
    mat_battery = make_mat("BatteryPack", (0.14, 0.16, 0.20), metallic=0.25, roughness=0.45)
    mat_accent = make_mat(
        "AccentLED",
        (0.15, 0.88, 0.98),
        metallic=0.1,
        roughness=0.18,
        emission=(0.15, 0.88, 0.98),
        emission_strength=3.0,
    )
    mat_prop = make_mat("PropBlade", (0.10, 0.11, 0.13), metallic=0.25, roughness=0.35)
    mat_lens = make_mat("OpticLens", (0.04, 0.12, 0.22), metallic=0.92, roughness=0.06)

    # Layered central fuselage (nose faces +X)
    add_box("FuselageLower", (0, 0, 0.02), (0.86, 0.56, 0.18), mat_carbon)
    add_box("TopCanopy", (0.06, 0, 0.16), (0.68, 0.44, 0.13), mat_shell)
    add_box("NoseVisor", (0.43, 0, 0.12), (0.12, 0.34, 0.11), mat_carbon)

    # Dual rear slide-in LiPo battery packs
    add_box("BatteryL", (-0.34, -0.15, 0.14), (0.28, 0.16, 0.16), mat_battery)
    add_box("BatteryR", (-0.34, 0.15, 0.14), (0.28, 0.16, 0.16), mat_battery)

    # Glowing agent-color accent spine + dual RTK GPS antenna masts
    accent = add_box("AccentLED", (0.08, 0, 0.24), (0.52, 0.22, 0.045), mat_accent)
    accent.name = "AccentLED"
    for gy in (-0.16, 0.16):
        add_cyl(f"RTKMast_{gy}", (-0.18, gy, 0.34), radius=0.016, depth=0.20, mat=mat_alu, vertices=8)
        add_cyl(f"RTKPuck_{gy}", (-0.18, gy, 0.46), radius=0.075, depth=0.042, mat=mat_shell, vertices=14)

    # Under-slung dual optical + laser methane sniffer gimbal payload
    add_box("GimbalYoke", (0.26, 0, -0.14), (0.18, 0.22, 0.14), mat_alu)
    add_box("GimbalBody", (0.28, 0, -0.27), (0.22, 0.24, 0.18), mat_carbon)
    add_cyl("OpticalLens", (0.40, -0.05, -0.27), radius=0.06, depth=0.05, mat=mat_lens, vertices=14, rot=(0, math.pi / 2, 0))
    add_cyl("ThermalLens", (0.40, 0.06, -0.27), radius=0.045, depth=0.05, mat=mat_alu, vertices=14, rot=(0, math.pi / 2, 0))
    add_cyl("SnifferTube", (0.48, 0.0, -0.21), radius=0.028, depth=0.28, mat=mat_alu, vertices=10, rot=(0, math.pi / 2, 0))

    # T-bar carbon landing skids with cross-braces
    for sy in (-0.40, 0.40):
        add_cyl(f"SkidBar_{sy}", (0.02, sy, -0.44), radius=0.026, depth=0.96, mat=mat_carbon, vertices=10, rot=(0, math.pi / 2, 0))
        add_cyl(f"SkidLegF_{sy}", (0.22, sy * 0.80, -0.23), radius=0.022, depth=0.44, mat=mat_carbon, vertices=8, rot=(sy * 0.42, 0, 0))
        add_cyl(f"SkidLegR_{sy}", (-0.22, sy * 0.80, -0.23), radius=0.022, depth=0.44, mat=mat_carbon, vertices=8, rot=(sy * 0.42, 0, 0))

    # 4 diagonal motor booms, brushed aluminum motor bells, spinner hubs & twin-blade props
    arm_angles = [math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]
    arm_len = 1.05
    for i, ang in enumerate(arm_angles):
        cx = math.cos(ang) * (arm_len * 0.5)
        cy = math.sin(ang) * (arm_len * 0.5)
        mx = math.cos(ang) * arm_len
        my = math.sin(ang) * arm_len

        add_cyl(
            f"Arm_{i}",
            (cx, cy, 0.09),
            radius=0.042,
            depth=arm_len,
            mat=mat_carbon,
            vertices=10,
            rot=(0, math.pi / 2, ang),
        )
        # High-vis accent collar on each arm
        add_cyl(
            f"NavLED_Arm_{i}",
            (mx * 0.72, my * 0.72, 0.09),
            radius=0.050,
            depth=0.14,
            mat=mat_accent,
            vertices=10,
            rot=(0, math.pi / 2, ang),
        )
        # Motor bell + silver spinner cap + bright nav LED pod
        add_cyl(f"Motor_{i}", (mx, my, 0.14), radius=0.105, depth=0.13, mat=mat_alu, vertices=16)
        add_cyl(f"NavLED_{i}", (mx, my, 0.04), radius=0.065, depth=0.05, mat=mat_accent, vertices=12)

        # Twin-blade propeller (named Rotor_0..3 so Three.js spins them)
        rotor = add_box(f"Rotor_{i}", (mx, my, 0.22), (0.98, 0.095, 0.02), mat_prop)
        rotor.name = f"Rotor_{i}"

    export_glb("quadrotor.glb")


# ---------------------------------------------------------------------------
# 6. Compressor / Heat-Exchanger Station (distinct from Control Building!)
#    Normalized to [-0.5, 0.5] x [-0.5, 0.5] x [0, 1.0]
# ---------------------------------------------------------------------------
def build_compressor_station() -> None:
    clear_scene()
    mat_concrete = make_mat("CompConcrete", (0.24, 0.26, 0.29), metallic=0.0, roughness=0.92)
    mat_steel = make_mat("CompSteel", (0.20, 0.25, 0.31), metallic=0.58, roughness=0.44)
    mat_vessel = make_mat("PressureVessel", (0.64, 0.68, 0.72), metallic=0.72, roughness=0.30)
    mat_dark = make_mat("CompDark", (0.12, 0.14, 0.17), metallic=0.75, roughness=0.36)
    mat_yellow = make_mat("CompYellow", (0.90, 0.66, 0.08), metallic=0.20, roughness=0.38)
    mat_blue = make_mat("CompBlue", (0.18, 0.44, 0.72), metallic=0.40, roughness=0.36)

    # Concrete equipment pad
    add_box("Pad", (0, 0, 0.03), (1.0, 1.0, 0.06), mat_concrete)

    # Half-open steel compressor house on the -Y half
    add_box("CompHall", (0, -0.20, 0.34), (0.90, 0.52, 0.56), mat_steel)
    add_box("CompRoof", (0, -0.20, 0.64), (0.94, 0.56, 0.04), mat_dark)

    # Twin horizontal cylindrical pressure vessels / heat exchangers on the +Y half
    for dy in (0.16, 0.36):
        add_cyl(
            f"Vessel_{dy:.2f}",
            (0.0, dy, 0.32),
            radius=0.11,
            depth=0.78,
            mat=mat_vessel,
            vertices=20,
            rot=(0, math.pi / 2, 0),
        )
        # Steel saddle supports
        for sx in (-0.26, 0.26):
            add_box(f"Saddle_{dy:.2f}_{sx:.2f}", (sx, dy, 0.14), (0.06, 0.22, 0.20), mat_dark)
        # Flanged end caps
        for ex in (-0.40, 0.40):
            add_cyl(
                f"Cap_{dy:.2f}_{ex:.2f}",
                (ex, dy, 0.32),
                radius=0.118,
                depth=0.03,
                mat=mat_blue,
                vertices=16,
                rot=(0, math.pi / 2, 0),
            )

    # 3 Tall industrial exhaust silencer stacks rising to z = 0.98
    for i, sx in enumerate((-0.28, 0.0, 0.28)):
        add_cyl(f"SilencerBase_{i}", (sx, -0.22, 0.74), radius=0.055, depth=0.20, mat=mat_vessel, vertices=14)
        add_cyl(f"SilencerPipe_{i}", (sx, -0.22, 0.89), radius=0.032, depth=0.20, mat=mat_dark, vertices=12)

    # Safety yellow walkway handrail along vessel deck
    add_box("WalkRailF", (0, 0.47, 0.14), (0.92, 0.018, 0.018), mat_yellow)
    for px in (-0.44, -0.22, 0.0, 0.22, 0.44):
        add_box(f"WalkPost_{px:.2f}", (px, 0.47, 0.10), (0.018, 0.018, 0.12), mat_yellow)

    export_glb("compressor_station.glb")


# ---------------------------------------------------------------------------
# 7. Iconic Industrial Flare Stack Tower (~32m tall lattice derrick)
# ---------------------------------------------------------------------------
def build_flare_stack() -> None:
    clear_scene()
    mat_steel = make_mat("DerrickSteel", (0.28, 0.33, 0.39), metallic=0.78, roughness=0.38)
    mat_pipe = make_mat("FlarePipe", (0.55, 0.59, 0.64), metallic=0.82, roughness=0.30)
    mat_red = make_mat("FlareStripeRed", (0.85, 0.18, 0.14), metallic=0.2, roughness=0.4)
    mat_flame = make_mat(
        "FlareFlame",
        (1.0, 0.55, 0.10),
        metallic=0.0,
        roughness=0.2,
        emission=(1.0, 0.48, 0.08),
        emission_strength=6.0,
    )

    # Concrete footing
    add_box("FlareBase", (0, 0, 0.4), (4.2, 4.2, 0.8), mat_steel)

    # Central vertical flare riser pipe (30m)
    add_cyl("FlareRiser", (0, 0, 15.5), radius=0.42, depth=30.0, mat=mat_pipe, vertices=16)
    add_cyl("FlareTip", (0, 0, 30.8), radius=0.52, depth=1.2, mat=mat_red, vertices=16)

    # 4 tapered steel lattice legs + horizontal tie rings
    for i in range(4):
        ang = i * math.pi / 2 + math.pi / 4
        lx = math.cos(ang) * 1.45
        ly = math.sin(ang) * 1.45
        add_cyl(f"Leg_{i}", (lx * 0.72, ly * 0.72, 13.0), radius=0.12, depth=26.0, mat=mat_steel, vertices=8)

    for z in (4.0, 9.0, 14.0, 19.0, 24.0):
        s = 2.6 - 0.04 * z
        add_box(f"RingN_{z}", (0, s * 0.5, z), (s, 0.14, 0.14), mat_steel)
        add_box(f"RingS_{z}", (0, -s * 0.5, z), (s, 0.14, 0.14), mat_steel)
        add_box(f"RingE_{z}", (s * 0.5, 0, z), (0.14, s, 0.14), mat_steel)
        add_box(f"RingW_{z}", (-s * 0.5, 0, z), (0.14, s, 0.14), mat_steel)

    # Glowing pilot flame cone at top
    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=0.72, depth=2.6, location=(0, 0, 32.6))
    flame = bpy.context.active_object
    flame.name = "FlareFlame"
    assign_mat(flame, mat_flame)

    export_glb("flare_stack.glb")


# ---------------------------------------------------------------------------
# 8. Fractionator / Distillation Column (normalized [-0.5, 0.5]^2 x [0, 1])
# ---------------------------------------------------------------------------
def build_fractionator() -> None:
    clear_scene()
    mat_concrete = make_mat("FracConcrete", (0.25, 0.27, 0.30), metallic=0.0, roughness=0.92)
    mat_copper = make_mat("FracCopperClad", (0.58, 0.36, 0.26), metallic=0.58, roughness=0.42)
    mat_steel = make_mat("FracSteel", (0.64, 0.69, 0.74), metallic=0.74, roughness=0.32)
    mat_dark = make_mat("FracDarkSteel", (0.16, 0.19, 0.23), metallic=0.78, roughness=0.38)
    mat_yellow = make_mat("FracSafetyYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)
    mat_beacon = make_mat(
        "FracBeacon",
        (1.0, 0.16, 0.10),
        metallic=0.1,
        roughness=0.2,
        emission=(1.0, 0.18, 0.12),
        emission_strength=5.5,
    )

    # Heavy octagonal concrete skirt & plinth
    add_cyl("FracPlinth", (0, 0, 0.035), radius=0.47, depth=0.07, mat=mat_concrete, vertices=8, smooth=False)
    add_cyl("FracSkirt", (0, 0, 0.11), radius=0.35, depth=0.10, mat=mat_dark, vertices=32)

    # Lower wide stripping section & upper narrower rectifying section
    add_cyl("FracBodyLower", (0, 0, 0.35), radius=0.31, depth=0.40, mat=mat_copper, vertices=48)
    add_cyl("FracTransition", (0, 0, 0.58), radius=0.26, depth=0.08, mat=mat_steel, vertices=48)
    add_cyl("FracBodyUpper", (0, 0, 0.74), radius=0.23, depth=0.28, mat=mat_steel, vertices=48)

    # Domed top head + overhead vapor goose-neck pipe
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=12, radius=0.23, location=(0, 0, 0.88))
    dome = bpy.context.active_object
    dome.name = "FracDome"
    dome.scale = (1.0, 1.0, 0.34)
    assign_mat(dome, mat_steel)
    set_smooth(dome)

    add_cyl("OverheadRiser", (0, 0, 0.945), radius=0.048, depth=0.07, mat=mat_dark, vertices=16)
    add_cyl("OverheadCross", (-0.16, 0, 0.965), radius=0.042, depth=0.32, mat=mat_dark, vertices=14, rot=(0, math.pi / 2, 0))
    add_cyl("OverheadDowncomer", (-0.31, 0, 0.56), radius=0.042, depth=0.82, mat=mat_dark, vertices=14)

    # 3 wrap-around circular inspection catwalk decks with yellow safety handrails
    for idx, (z, r_in, r_out) in enumerate(((0.28, 0.32, 0.45), (0.54, 0.28, 0.42), (0.80, 0.24, 0.38))):
        add_cyl(f"Deck_{idx}", (0, 0, z), radius=r_out, depth=0.018, mat=mat_dark, vertices=32)
        bpy.ops.mesh.primitive_torus_add(
            major_radius=r_out - 0.01,
            minor_radius=0.008,
            major_segments=28,
            minor_segments=6,
            location=(0, 0, z + 0.055),
        )
        rail = bpy.context.active_object
        rail.name = f"DeckRail_{idx}"
        assign_mat(rail, mat_yellow)
        for p in range(10):
            ang = 2.0 * math.pi * p / 10.0
            add_cyl(
                f"DeckPost_{idx}_{p}",
                ((r_out - 0.01) * math.cos(ang), (r_out - 0.01) * math.sin(ang), z + 0.03),
                radius=0.006,
                depth=0.055,
                mat=mat_yellow,
                vertices=6,
            )
        # Side instrument / reboiler nozzle box
        add_box(f"NozzleBox_{idx}", (r_in + 0.05, 0.0, z + 0.04), (0.09, 0.11, 0.08), mat_steel)

    # Vertical caged ladder on +Y face
    for dx in (-0.025, 0.025):
        add_box(f"FracLadderRail_{dx:.3f}", (dx, 0.43, 0.48), (0.010, 0.010, 0.84), mat_yellow)
    for k in range(16):
        add_box(f"FracRung_{k}", (0.0, 0.43, 0.10 + k * 0.05), (0.055, 0.008, 0.008), mat_yellow)

    # Top aviation beacon
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=8, radius=0.025, location=(0.0, 0.0, 0.99))
    bcn = bpy.context.active_object
    bcn.name = "FracAviationBeacon"
    assign_mat(bcn, mat_beacon)

    export_glb("fractionator.glb")


# ---------------------------------------------------------------------------
# 9. Horton Sphere Pressurized LPG Tank (normalized [-0.5, 0.5]^2 x [0, 1])
# ---------------------------------------------------------------------------
def build_horton_sphere() -> None:
    clear_scene()
    mat_concrete = make_mat("SphereConcrete", (0.26, 0.28, 0.31), metallic=0.0, roughness=0.92)
    mat_shell = make_mat("SphereShell", (0.78, 0.81, 0.84), metallic=0.38, roughness=0.36)
    mat_band = make_mat("SphereBand", (0.86, 0.44, 0.10), metallic=0.20, roughness=0.40)
    mat_steel = make_mat("SphereLegs", (0.22, 0.26, 0.32), metallic=0.72, roughness=0.40)
    mat_yellow = make_mat("SphereYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)

    # Ring concrete foundation
    add_cyl("SphereFoundation", (0, 0, 0.03), radius=0.46, depth=0.06, mat=mat_concrete, vertices=32)

    # Main spherical pressure vessel centered at z = 0.56, radius = 0.36
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=28, radius=0.36, location=(0, 0, 0.56))
    sphere = bpy.context.active_object
    sphere.name = "HortonSphereBody"
    assign_mat(sphere, mat_shell)
    set_smooth(sphere)

    # Equatorial orange hazard weld band
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.358,
        minor_radius=0.016,
        major_segments=48,
        minor_segments=10,
        location=(0, 0, 0.56),
    )
    eq = bpy.context.active_object
    eq.name = "EquatorialBand"
    assign_mat(eq, mat_band)
    set_smooth(eq)

    # 6 tubular steel support legs around the equator + diagonal cross-bracing rods
    n_legs = 6
    leg_r = 0.355
    for i in range(n_legs):
        ang = 2.0 * math.pi * i / n_legs
        lx = leg_r * math.cos(ang)
        ly = leg_r * math.sin(ang)
        add_cyl(f"SphereLeg_{i}", (lx, ly, 0.30), radius=0.028, depth=0.56, mat=mat_steel, vertices=14)
        add_cyl(f"SphereFoot_{i}", (lx, ly, 0.065), radius=0.052, depth=0.04, mat=mat_concrete, vertices=12)

        # Horizontal tie beam to next leg at z = 0.24
        ang_next = 2.0 * math.pi * ((i + 1) % n_legs) / n_legs
        nx = leg_r * math.cos(ang_next)
        ny = leg_r * math.sin(ang_next)
        mx, my = 0.5 * (lx + nx), 0.5 * (ly + ny)
        span = math.hypot(nx - lx, ny - ly)
        yaw = math.atan2(ny - ly, nx - lx)
        add_cyl(
            f"TieBeam_{i}",
            (mx, my, 0.24),
            radius=0.012,
            depth=span,
            mat=mat_steel,
            vertices=8,
            rot=(0.0, math.pi / 2.0, yaw),
        )

    # Bottom central discharge riser & manifold
    add_cyl("BottomRiser", (0, 0, 0.14), radius=0.042, depth=0.22, mat=mat_steel, vertices=14)
    add_cyl("BottomHeader", (0.18, 0, 0.11), radius=0.036, depth=0.36, mat=mat_steel, vertices=14, rot=(0, math.pi / 2, 0))
    add_cyl("BottomFlange", (0.36, 0, 0.11), radius=0.055, depth=0.02, mat=mat_band, vertices=14, rot=(0, math.pi / 2, 0))

    # Top circular inspection platform, safety railing & relief valve stack
    add_cyl("TopPlatform", (0, 0, 0.925), radius=0.14, depth=0.016, mat=mat_steel, vertices=24)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.135, minor_radius=0.007, major_segments=24, minor_segments=6, location=(0, 0, 0.975)
    )
    tr = bpy.context.active_object
    tr.name = "TopRail"
    assign_mat(tr, mat_yellow)
    add_cyl("ReliefValveA", (-0.04, 0, 0.96), radius=0.018, depth=0.07, mat=mat_band, vertices=12)
    add_cyl("ReliefValveB", (0.04, 0, 0.96), radius=0.018, depth=0.07, mat=mat_band, vertices=12)

    export_glb("horton_sphere.glb")


# ---------------------------------------------------------------------------
# 10. Metering Skid (Pipe Loop inlet custody-transfer unit)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_metering_skid() -> None:
    clear_scene()
    mat_concrete = make_mat("MeterConcrete", (0.24, 0.26, 0.29), metallic=0.0, roughness=0.92)
    mat_skid = make_mat("MeterSkidDark", (0.15, 0.18, 0.22), metallic=0.76, roughness=0.38)
    mat_teal = make_mat("MeterTeal", (0.16, 0.44, 0.42), metallic=0.42, roughness=0.38)
    mat_steel = make_mat("MeterSteel", (0.72, 0.76, 0.80), metallic=0.84, roughness=0.26)
    mat_amber = make_mat("MeterAmber", (0.90, 0.62, 0.10), metallic=0.25, roughness=0.38)
    mat_screen = make_mat(
        "MeterScreen",
        (0.15, 0.92, 0.65),
        metallic=0.05,
        roughness=0.15,
        emission=(0.12, 0.88, 0.58),
        emission_strength=3.8,
    )

    # Heavy steel I-beam skid base on concrete pad
    add_box("SkidPad", (0, 0, 0.03), (1.0, 1.0, 0.06), mat_concrete)
    add_box("SkidFrame", (0, 0, 0.085), (0.94, 0.92, 0.05), mat_skid)

    # Twin parallel ultrasonic/orifice gas meter runs along X
    for dy in (-0.24, 0.24):
        add_cyl(f"MeterPipe_{dy:.2f}", (0.04, dy, 0.30), radius=0.105, depth=0.78, mat=mat_teal, vertices=24, rot=(0, math.pi / 2, 0))
        # Bolted ANSI flange collars & cradle saddles
        for fx in (-0.34, -0.06, 0.18, 0.38):
            add_cyl(f"MeterFlange_{dy:.2f}_{fx:.2f}", (fx, dy, 0.30), radius=0.145, depth=0.032, mat=mat_steel, vertices=20, rot=(0, math.pi / 2, 0))
        for sx in (-0.28, 0.30):
            add_box(f"MeterSaddle_{dy:.2f}_{sx:.2f}", (sx, dy, 0.16), (0.07, 0.18, 0.14), mat_skid)
        # Pressure/flow transmitter tree on top of each meter tube
        add_cyl(f"TransmitterStem_{dy:.2f}", (0.06, dy, 0.46), radius=0.025, depth=0.16, mat=mat_steel, vertices=12)
        add_cyl(f"TransmitterHead_{dy:.2f}", (0.06, dy, 0.56), radius=0.062, depth=0.07, mat=mat_amber, vertices=16)

    # Cross-connecting header manifold on +X end
    add_cyl("CrossHeader", (0.40, 0.0, 0.30), radius=0.095, depth=0.60, mat=mat_steel, vertices=20, rot=(math.pi / 2, 0, 0))

    # Telemetry & flow-computer cabinet on -X side with glowing HMI display
    add_box("TelemetryCab", (-0.34, 0.0, 0.46), (0.22, 0.34, 0.68), mat_teal)
    add_box("CabHood", (-0.34, 0.0, 0.82), (0.26, 0.38, 0.045), mat_skid)
    add_box("CabHMI", (-0.225, 0.0, 0.56), (0.015, 0.16, 0.12), mat_screen)

    # Protective open steel weather canopy roof on 4 corner posts (dark steel top + thin yellow perimeter trim)
    for cx in (-0.44, 0.44):
        for cy in (-0.43, 0.43):
            add_box(f"CanopyPost_{cx:.2f}_{cy:.2f}", (cx, cy, 0.52), (0.038, 0.038, 0.84), mat_skid)
    add_box("CanopyRoof", (0.0, 0.0, 0.955), (0.96, 0.94, 0.045), mat_skid)
    add_box("CanopyTrimN", (0.0, 0.475, 0.965), (0.98, 0.025, 0.035), mat_amber)
    add_box("CanopyTrimS", (0.0, -0.475, 0.965), (0.98, 0.025, 0.035), mat_amber)
    add_box("CanopyTrimE", (0.485, 0.0, 0.965), (0.025, 0.96, 0.035), mat_amber)
    add_box("CanopyTrimW", (-0.485, 0.0, 0.965), (0.025, 0.96, 0.035), mat_amber)

    export_glb("metering_skid.glb")


# ---------------------------------------------------------------------------
# 11. Receiver / Pig-Launcher Skid (Pipe Loop outlet unit)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_receiver_skid() -> None:
    clear_scene()
    mat_concrete = make_mat("RecvConcrete", (0.24, 0.26, 0.29), metallic=0.0, roughness=0.92)
    mat_dark = make_mat("RecvDark", (0.14, 0.17, 0.21), metallic=0.76, roughness=0.38)
    mat_steel = make_mat("RecvSteel", (0.68, 0.73, 0.78), metallic=0.82, roughness=0.28)
    mat_teal = make_mat("RecvTeal", (0.16, 0.44, 0.46), metallic=0.45, roughness=0.36)
    mat_amber = make_mat("RecvAmber", (0.92, 0.66, 0.08), metallic=0.22, roughness=0.38)

    # Concrete pad + heavy skid beam deck
    add_box("RecvPad", (0, 0, 0.03), (1.0, 1.0, 0.06), mat_concrete)
    add_box("RecvSkid", (0, 0, 0.085), (0.94, 0.92, 0.05), mat_dark)

    # Main horizontal slug-catcher / pig-receiver barrel
    add_cyl("RecvBarrel", (-0.02, 0.0, 0.40), radius=0.22, depth=0.76, mat=mat_steel, vertices=32, rot=(0, math.pi / 2, 0))
    # Quick-opening hinged closure door on +X end & reducer cap on -X end
    add_cyl("ClosureFlange", (0.37, 0.0, 0.40), radius=0.255, depth=0.06, mat=mat_teal, vertices=28, rot=(0, math.pi / 2, 0))
    add_cyl("ClosureDoor", (0.41, 0.0, 0.40), radius=0.22, depth=0.04, mat=mat_amber, vertices=24, rot=(0, math.pi / 2, 0))
    add_cyl("ReducerEnd", (-0.41, 0.0, 0.40), radius=0.19, depth=0.06, mat=mat_teal, vertices=24, rot=(0, math.pi / 2, 0))

    # Heavy steel saddle cradles
    for sx in (-0.28, 0.24):
        add_box(f"RecvSaddle_{sx:.2f}", (sx, 0.0, 0.19), (0.10, 0.42, 0.20), mat_dark)

    # Overhead blowdown / pressure-relief riser + bypass loop
    add_cyl("ReliefRiser", (0.06, 0.0, 0.76), radius=0.055, depth=0.34, mat=mat_steel, vertices=16)
    add_cyl("ReliefHead", (0.06, 0.0, 0.95), radius=0.095, depth=0.05, mat=mat_teal, vertices=16)
    add_cyl("BypassLine", (-0.02, -0.32, 0.34), radius=0.055, depth=0.68, mat=mat_teal, vertices=16, rot=(0, math.pi / 2, 0))
    add_cyl("BypassConnectL", (-0.30, -0.18, 0.34), radius=0.048, depth=0.28, mat=mat_dark, vertices=14, rot=(math.pi / 2, 0, 0))
    add_cyl("BypassConnectR", (0.26, -0.18, 0.34), radius=0.048, depth=0.28, mat=mat_dark, vertices=14, rot=(math.pi / 2, 0, 0))

    # Maintenance catwalk & yellow safety handrail along +Y side
    add_box("RecvWalkway", (0.0, 0.36, 0.14), (0.86, 0.18, 0.03), mat_dark)
    add_box("RecvRailTop", (0.0, 0.44, 0.34), (0.86, 0.018, 0.018), mat_amber)
    for px in (-0.40, -0.14, 0.14, 0.40):
        add_box(f"RecvPost_{px:.2f}", (px, 0.44, 0.24), (0.018, 0.018, 0.20), mat_amber)

    export_glb("receiver_skid.glb")


# ---------------------------------------------------------------------------
# 12. Cryogenic LNG/LOX Vessel (Tank Terminal signature)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_cryogenic_vessel() -> None:
    clear_scene()
    mat_concrete = make_mat("CryoConcrete", (0.30, 0.34, 0.38), metallic=0.0, roughness=0.90)
    mat_jacket = make_mat("CryoWhiteJacket", (0.86, 0.90, 0.94), metallic=0.28, roughness=0.30)
    mat_ice_blue = make_mat("CryoIceBlue", (0.16, 0.48, 0.68), metallic=0.35, roughness=0.34)
    mat_dark = make_mat("CryoDark", (0.18, 0.22, 0.27), metallic=0.72, roughness=0.38)
    mat_alu = make_mat("CryoFinAlu", (0.76, 0.82, 0.88), metallic=0.85, roughness=0.24)
    mat_yellow = make_mat("CryoYellow", (0.92, 0.70, 0.08), metallic=0.18, roughness=0.38)

    # Raised concrete ring pier & steel skirt
    add_cyl("CryoPlinth", (0, 0, 0.035), radius=0.47, depth=0.07, mat=mat_concrete, vertices=48)
    add_cyl("CryoSkirt", (0, 0, 0.095), radius=0.41, depth=0.06, mat=mat_dark, vertices=48)

    # Tall gleam-white insulated double-wall vacuum vessel
    add_cyl("CryoShell", (0, 0, 0.50), radius=0.39, depth=0.76, mat=mat_jacket, vertices=48)

    # Cryogenic blue accent bands & vacuum stiffening rings
    for z in (0.18, 0.52, 0.84):
        add_cyl(f"CryoBand_{z:.2f}", (0, 0, z), radius=0.402, depth=0.036, mat=mat_ice_blue, vertices=48)

    # Hemispherical dome top + dual pressure-relief valves
    bpy.ops.mesh.primitive_uv_sphere_add(segments=40, ring_count=16, radius=0.39, location=(0, 0, 0.88))
    dome = bpy.context.active_object
    dome.name = "CryoDome"
    dome.scale = (1.0, 1.0, 0.26)
    assign_mat(dome, mat_jacket)
    set_smooth(dome)

    add_cyl("CryoReliefA", (-0.08, 0, 0.96), radius=0.032, depth=0.08, mat=mat_dark, vertices=14)
    add_cyl("CryoReliefB", (0.08, 0, 0.96), radius=0.032, depth=0.08, mat=mat_dark, vertices=14)

    # Side ambient air-vaporizer aluminum fin-tube bank (+X side)
    for i, dy in enumerate((-0.14, -0.05, 0.05, 0.14)):
        add_cyl(f"VapTube_{i}", (0.43, dy, 0.46), radius=0.022, depth=0.66, mat=mat_alu, vertices=10)
    add_box("VapHeaderBot", (0.43, 0.0, 0.14), (0.05, 0.34, 0.04), mat_ice_blue)
    add_box("VapHeaderTop", (0.43, 0.0, 0.78), (0.05, 0.34, 0.04), mat_ice_blue)

    # Caged ladder on -X side
    for dy in (-0.025, 0.025):
        add_box(f"CryoLadder_{dy:.3f}", (-0.42, dy, 0.48), (0.010, 0.010, 0.80), mat_yellow)
    for k in range(14):
        add_box(f"CryoRung_{k}", (-0.42, 0.0, 0.12 + k * 0.055), (0.008, 0.055, 0.008), mat_yellow)

    export_glb("cryogenic_vessel.glb")


# ---------------------------------------------------------------------------
# 13. Transfer Gantry Truss Bridge (Tank Terminal elevated bridge)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_transfer_gantry() -> None:
    clear_scene()
    mat_blue = make_mat("GantryMarineBlue", (0.16, 0.36, 0.54), metallic=0.58, roughness=0.40)
    mat_steel = make_mat("GantryLightSteel", (0.70, 0.75, 0.80), metallic=0.80, roughness=0.30)
    mat_deck = make_mat("GantryDeck", (0.18, 0.21, 0.25), metallic=0.68, roughness=0.52)
    mat_yellow = make_mat("GantryYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)

    # Lower & upper longitudinal chord beams along X
    for y in (-0.42, 0.42):
        add_box(f"ChordBot_{y:.2f}", (0.0, y, 0.10), (1.0, 0.075, 0.085), mat_blue)
        add_box(f"ChordTop_{y:.2f}", (0.0, y, 0.90), (1.0, 0.075, 0.085), mat_blue)
        # Vertical truss posts & diagonal Warren braces
        n_bays = 6
        for i in range(n_bays + 1):
            fx = -0.48 + i * (0.96 / n_bays)
            add_box(f"Post_{y:.2f}_{i}", (fx, y, 0.50), (0.045, 0.06, 0.80), mat_blue)
        for i in range(n_bays):
            x0 = -0.48 + i * (0.96 / n_bays)
            x1 = -0.48 + (i + 1) * (0.96 / n_bays)
            xm = 0.5 * (x0 + x1)
            span = x1 - x0
            diag_len = math.hypot(span, 0.76)
            pitch = math.atan2(span, 0.76) * (1.0 if i % 2 == 0 else -1.0)
            add_box(f"Diag_{y:.2f}_{i}", (xm, y, 0.50), (0.036, 0.045, diag_len), mat_steel, rot=(0.0, pitch, 0.0))

    # Cross portal beams & grated catwalk floor
    for i in range(7):
        fx = -0.48 + i * (0.96 / 6.0)
        add_box(f"CrossBot_{i}", (fx, 0.0, 0.08), (0.045, 0.88, 0.05), mat_blue)
        add_box(f"CrossTop_{i}", (fx, 0.0, 0.92), (0.045, 0.88, 0.05), mat_blue)
    add_box("BridgeDeck", (0.0, 0.0, 0.13), (0.98, 0.82, 0.035), mat_deck)

    # Safety handrails along catwalk
    for y in (-0.36, 0.36):
        add_box(f"BridgeRail_{y:.2f}", (0.0, y, 0.32), (0.98, 0.016, 0.016), mat_yellow)

    # 3 Insulated cryogenic transfer pipelines suspended inside the truss
    for idx, dy in enumerate((-0.20, 0.0, 0.20)):
        add_cyl(f"BridgePipe_{idx}", (0.0, dy, 0.64), radius=0.058, depth=0.98, mat=mat_steel, vertices=16, rot=(0, math.pi / 2, 0))
        for fx in (-0.30, 0.0, 0.30):
            add_cyl(f"BridgeFlange_{idx}_{fx:.2f}", (fx, dy, 0.64), radius=0.082, depth=0.022, mat=mat_blue, vertices=14, rot=(0, math.pi / 2, 0))

    export_glb("transfer_gantry.glb")


# ---------------------------------------------------------------------------
# 14. Compressor Train Skid (Compressor Yard bay unit)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_compressor_train() -> None:
    clear_scene()
    mat_concrete = make_mat("TrainConcrete", (0.22, 0.24, 0.27), metallic=0.0, roughness=0.92)
    mat_dark = make_mat("TrainDark", (0.14, 0.17, 0.21), metallic=0.76, roughness=0.38)
    mat_teal = make_mat("TrainTeal", (0.16, 0.42, 0.38), metallic=0.45, roughness=0.38)
    mat_steel = make_mat("TrainSteel", (0.68, 0.72, 0.76), metallic=0.82, roughness=0.28)
    mat_amber = make_mat("TrainAmber", (0.90, 0.64, 0.08), metallic=0.24, roughness=0.38)
    mat_led = make_mat(
        "TrainLED",
        (0.20, 0.90, 1.0),
        metallic=0.1,
        roughness=0.15,
        emission=(0.18, 0.85, 1.0),
        emission_strength=3.5,
    )

    # Concrete inertia block & steel baseplate
    add_box("InertiaPad", (0, 0, 0.05), (1.0, 1.0, 0.10), mat_concrete)
    add_box("SkidBase", (0, 0, 0.12), (0.94, 0.92, 0.04), mat_dark)

    # Twin heavy reciprocating compressor cylinders + finned motor housings along X
    for dy in (-0.23, 0.23):
        add_cyl(f"MotorBody_{dy:.2f}", (-0.12, dy, 0.38), radius=0.17, depth=0.56, mat=mat_teal, vertices=24, rot=(0, math.pi / 2, 0))
        add_cyl(f"CompHead_{dy:.2f}", (0.28, dy, 0.38), radius=0.145, depth=0.22, mat=mat_steel, vertices=22, rot=(0, math.pi / 2, 0))
        add_cyl(f"CouplingGuard_{dy:.2f}", (0.16, dy, 0.38), radius=0.185, depth=0.065, mat=mat_amber, vertices=22, rot=(0, math.pi / 2, 0))
        # Cooling fins on motor body
        for f in range(5):
            add_cyl(
                f"Fin_{dy:.2f}_{f}",
                (-0.34 + f * 0.075, dy, 0.38),
                radius=0.186,
                depth=0.018,
                mat=mat_dark,
                vertices=22,
                rot=(0, math.pi / 2, 0),
            )
        # Overhead pulsation dampener bottle
        add_cyl(f"PulsationBottle_{dy:.2f}", (-0.04, dy, 0.72), radius=0.085, depth=0.58, mat=mat_steel, vertices=18, rot=(0, math.pi / 2, 0))
        add_cyl(f"BottleLegA_{dy:.2f}", (-0.22, dy, 0.56), radius=0.032, depth=0.18, mat=mat_dark, vertices=12)
        add_cyl(f"BottleLegB_{dy:.2f}", (0.16, dy, 0.56), radius=0.032, depth=0.18, mat=mat_dark, vertices=12)

    # Local acoustic intake hood & PLC kiosk on +X center
    add_box("PLCKiosk", (0.32, 0.0, 0.54), (0.20, 0.28, 0.76), mat_teal)
    add_box("PLCScreen", (0.425, 0.0, 0.68), (0.016, 0.14, 0.10), mat_led)
    add_box("IntakeHood", (-0.36, 0.0, 0.78), (0.18, 0.58, 0.24), mat_dark)

    export_glb("compressor_train.glb")


# ---------------------------------------------------------------------------
# 15. Induced-Draft Cooling Tower Cell (Cooling Works)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_cooling_cell() -> None:
    clear_scene()
    mat_concrete = make_mat("CoolConcrete", (0.24, 0.27, 0.30), metallic=0.0, roughness=0.92)
    mat_slate = make_mat("CoolSlateBlue", (0.18, 0.34, 0.44), metallic=0.35, roughness=0.46)
    mat_dark = make_mat("CoolDark", (0.12, 0.15, 0.19), metallic=0.68, roughness=0.45)
    mat_enamel = make_mat("CoolEnamel", (0.74, 0.79, 0.82), metallic=0.28, roughness=0.42)
    mat_yellow = make_mat("CoolYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)

    # Cold-water concrete basin
    add_box("WaterBasin", (0, 0, 0.07), (1.0, 1.0, 0.14), mat_concrete)

    # Lower louver plenum & upper fiberglass casing
    add_box("LowerPlenum", (0, 0, 0.30), (0.92, 0.88, 0.32), mat_slate)
    add_box("UpperCasing", (0, 0, 0.62), (0.90, 0.84, 0.34), mat_dark)

    # Horizontal intake louver slats on +Y and -Y faces
    for y_sign in (-1.0, 1.0):
        for row in range(6):
            add_box(
                f"Louver_{y_sign}_{row}",
                (0.0, y_sign * 0.445, 0.20 + row * 0.052),
                (0.84, 0.035, 0.022),
                mat_enamel,
                rot=(y_sign * 0.35, 0.0, 0.0),
            )

    # Corner structural steel pilasters
    for cx in (-0.45, 0.45):
        for cy in (-0.43, 0.43):
            add_box(f"CoolCol_{cx:.2f}_{cy:.2f}", (cx, cy, 0.48), (0.05, 0.05, 0.72), mat_enamel)

    # Fan deck roof + yellow safety perimeter guardrail
    add_box("FanDeck", (0, 0, 0.81), (0.96, 0.92, 0.045), mat_slate)
    add_box("DeckRailN", (0, 0.45, 0.88), (0.94, 0.016, 0.016), mat_yellow)
    add_box("DeckRailS", (0, -0.45, 0.88), (0.94, 0.016, 0.016), mat_yellow)
    add_box("DeckRailE", (0.47, 0, 0.88), (0.016, 0.90, 0.016), mat_yellow)
    add_box("DeckRailW", (-0.47, 0, 0.88), (0.016, 0.90, 0.016), mat_yellow)

    # Large circular fiberglass fan stack shroud + axial fan hub & 4 pitched blades
    add_cyl("FanShroudOuter", (0, 0, 0.915), radius=0.38, depth=0.17, mat=mat_enamel, vertices=36)
    add_cyl("FanThroat", (0, 0, 0.94), radius=0.33, depth=0.13, mat=mat_dark, vertices=32)
    add_cyl("FanHub", (0, 0, 0.945), radius=0.08, depth=0.05, mat=mat_slate, vertices=16)
    for b in range(4):
        ang = b * math.pi / 2.0
        add_box(
            f"FanBlade_{b}",
            (0.18 * math.cos(ang), 0.18 * math.sin(ang), 0.945),
            (0.24, 0.065, 0.012),
            mat_enamel,
            rot=(0.25, 0.0, ang),
        )

    export_glb("cooling_cell.glb")


# ---------------------------------------------------------------------------
# 16. Natural-Draft Hyperbolic Cooling Tower (Cooling Works signature!)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_hyperbolic_tower() -> None:
    clear_scene()
    mat_concrete = make_mat("HypConcrete", (0.56, 0.59, 0.62), metallic=0.0, roughness=0.86)
    mat_base = make_mat("HypBaseDark", (0.24, 0.27, 0.30), metallic=0.05, roughness=0.92)
    mat_band_red = make_mat("HypWarnRed", (0.82, 0.18, 0.14), metallic=0.10, roughness=0.48)
    mat_band_white = make_mat("HypWarnWhite", (0.90, 0.92, 0.94), metallic=0.08, roughness=0.48)
    mat_throat = make_mat("HypInterior", (0.06, 0.08, 0.11), metallic=0.0, roughness=0.95)

    # Circular water collection basin at base
    add_cyl("HypBasin", (0, 0, 0.03), radius=0.49, depth=0.06, mat=mat_base, vertices=48)

    # 16 angled V-shaped concrete diagonal support stilts around the air intake
    n_stilts = 16
    for i in range(n_stilts):
        ang = 2.0 * math.pi * i / n_stilts
        r_mid = 0.455
        sx = r_mid * math.cos(ang)
        sy = r_mid * math.sin(ang)
        tilt = 0.32 if i % 2 == 0 else -0.32
        add_cyl(
            f"HypStilt_{i}",
            (sx, sy, 0.095),
            radius=0.016,
            depth=0.14,
            mat=mat_concrete,
            vertices=8,
            rot=(tilt * math.cos(ang), tilt * math.sin(ang), 0.0),
        )

    # Exact smooth hyperboloid concrete shell from z = 0.13 to z = 1.00
    n_seg = 48
    n_rings = 24
    z_min, z_max = 0.13, 1.00
    z_throat = 0.74
    r_throat = 0.285
    c_param = 0.48

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for r_idx in range(n_rings + 1):
        t = r_idx / n_rings
        z = z_min + t * (z_max - z_min)
        rad = r_throat * math.sqrt(1.0 + ((z - z_throat) / c_param) ** 2)
        for s_idx in range(n_seg):
            ang = 2.0 * math.pi * s_idx / n_seg
            verts.append((rad * math.cos(ang), rad * math.sin(ang), z))

    for r_idx in range(n_rings):
        for s_idx in range(n_seg):
            s_next = (s_idx + 1) % n_seg
            a = r_idx * n_seg + s_idx
            b = r_idx * n_seg + s_next
            c = (r_idx + 1) * n_seg + s_next
            d = (r_idx + 1) * n_seg + s_idx
            faces.append((a, b, c, d))

    mesh = bpy.data.meshes.new("HyperboloidShellMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    shell_obj = bpy.data.objects.new("HyperboloidShell", mesh)
    bpy.context.collection.objects.link(shell_obj)
    assign_mat(shell_obj, mat_concrete)
    set_smooth(shell_obj)

    # Top stiffening cornice rim & recessed dark interior throat
    r_top = r_throat * math.sqrt(1.0 + ((z_max - z_throat) / c_param) ** 2)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=r_top, minor_radius=0.015, major_segments=48, minor_segments=8, location=(0, 0, z_max - 0.008)
    )
    rim = bpy.context.active_object
    rim.name = "HypTopRim"
    assign_mat(rim, mat_concrete)
    set_smooth(rim)

    add_cyl("HypDarkThroat", (0, 0, 0.94), radius=r_top * 0.93, depth=0.02, mat=mat_throat, vertices=36)

    # Red & white aviation warning bands near top
    for idx, (bz, bmat) in enumerate(((0.91, mat_band_red), (0.86, mat_band_white), (0.81, mat_band_red))):
        brad = r_throat * math.sqrt(1.0 + ((bz - z_throat) / c_param) ** 2) + 0.006
        add_cyl(f"HypBand_{idx}", (0, 0, bz), radius=brad, depth=0.045, mat=bmat, vertices=48)

    export_glb("hyperbolic_tower.glb")


# ---------------------------------------------------------------------------
# 17. Industrial Flue / Exhaust Chimney Stack (normalized [-0.5, 0.5]^2 x [0, 1])
# ---------------------------------------------------------------------------
def build_exhaust_stack() -> None:
    clear_scene()
    mat_concrete = make_mat("StackConcrete", (0.26, 0.29, 0.32), metallic=0.0, roughness=0.92)
    mat_steel = make_mat("StackSteel", (0.66, 0.71, 0.76), metallic=0.72, roughness=0.34)
    mat_dark = make_mat("StackDark", (0.16, 0.19, 0.23), metallic=0.78, roughness=0.38)
    mat_red = make_mat("StackRed", (0.84, 0.18, 0.14), metallic=0.18, roughness=0.42)
    mat_white = make_mat("StackWhite", (0.90, 0.92, 0.94), metallic=0.14, roughness=0.42)
    mat_yellow = make_mat("StackYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)
    mat_beacon = make_mat(
        "StackBeacon",
        (1.0, 0.16, 0.10),
        metallic=0.1,
        roughness=0.2,
        emission=(1.0, 0.18, 0.12),
        emission_strength=5.5,
    )

    # Concrete base pedestal
    add_cyl("StackPedestal", (0, 0, 0.04), radius=0.49, depth=0.08, mat=mat_concrete, vertices=24)

    # Tapered chimney body (smooth cone frustum)
    bpy.ops.mesh.primitive_cone_add(
        vertices=32, radius1=0.44, radius2=0.30, depth=0.84, location=(0, 0, 0.50)
    )
    body = bpy.context.active_object
    body.name = "StackBody"
    assign_mat(body, mat_steel)
    set_smooth(body)

    # Red/white aviation warning stripes on upper third
    for idx, (z, r1, r2, smat) in enumerate((
        (0.74, 0.355, 0.340, mat_red),
        (0.81, 0.342, 0.328, mat_white),
        (0.88, 0.330, 0.316, mat_red),
    )):
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=r1, radius2=r2, depth=0.068, location=(0, 0, z))
        stripe = bpy.context.active_object
        stripe.name = f"StackStripe_{idx}"
        assign_mat(stripe, smat)
        set_smooth(stripe)

    # Flared dark exhaust nozzle crown + twin aviation warning beacons
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=0.30, radius2=0.36, depth=0.08, location=(0, 0, 0.96))
    crown = bpy.context.active_object
    crown.name = "StackCrown"
    assign_mat(crown, mat_dark)
    set_smooth(crown)

    for bx in (-0.36, 0.36):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=8, radius=0.032, location=(bx, 0, 0.98))
        bcn = bpy.context.active_object
        bcn.name = f"StackBeacon_{bx:.2f}"
        assign_mat(bcn, mat_beacon)

    # Mid-height emissions sampling catwalk ring with yellow handrail
    add_cyl("CEMSDeck", (0, 0, 0.48), radius=0.46, depth=0.018, mat=mat_dark, vertices=24)
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.45, minor_radius=0.010, major_segments=24, minor_segments=6, location=(0, 0, 0.525)
    )
    crail = bpy.context.active_object
    crail.name = "CEMSRail"
    assign_mat(crail, mat_yellow)

    export_glb("exhaust_stack.glb")


# ---------------------------------------------------------------------------
# 18. Industrial Pump & Substation House (pitched roof, distinct from control bldg)
#     Normalized [-0.5, 0.5]^2 x [0, 1]
# ---------------------------------------------------------------------------
def build_pump_house() -> None:
    clear_scene()
    mat_concrete = make_mat("PumpConcrete", (0.24, 0.26, 0.29), metallic=0.0, roughness=0.92)
    mat_wall = make_mat("PumpWall", (0.30, 0.36, 0.42), metallic=0.40, roughness=0.48)
    mat_roof = make_mat("PumpRoof", (0.18, 0.28, 0.38), metallic=0.52, roughness=0.44)
    mat_dark = make_mat("PumpDark", (0.13, 0.16, 0.20), metallic=0.74, roughness=0.38)
    mat_door = make_mat("PumpDoor", (0.46, 0.50, 0.55), metallic=0.62, roughness=0.38)
    mat_yellow = make_mat("PumpYellow", (0.92, 0.68, 0.08), metallic=0.18, roughness=0.38)
    mat_light = make_mat(
        "PumpLight",
        (0.95, 0.96, 1.0),
        metallic=0.05,
        roughness=0.12,
        emission=(0.85, 0.92, 1.0),
        emission_strength=4.2,
    )

    # Concrete plinth
    add_box("PumpFoundation", (0, 0, 0.03), (1.0, 1.0, 0.06), mat_concrete)

    # Main hall walls
    add_box("PumpWalls", (0, 0, 0.36), (0.92, 0.88, 0.60), mat_wall)

    # Pitched gable roof slabs + ridge ventilator monitor
    add_box("RoofSlopeL", (0, -0.23, 0.76), (0.96, 0.50, 0.04), mat_roof, rot=(0.34, 0.0, 0.0))
    add_box("RoofSlopeR", (0, 0.23, 0.76), (0.96, 0.50, 0.04), mat_roof, rot=(-0.34, 0.0, 0.0))
    add_box("RidgeMonitor", (0, 0, 0.88), (0.76, 0.16, 0.12), mat_dark)
    add_box("RidgeCap", (0, 0, 0.95), (0.80, 0.22, 0.03), mat_roof)

    # Dual rollup service doors + bright LED entrance strips on -Y face
    for dx in (-0.22, 0.22):
        add_box(f"PumpBayFrame_{dx:.2f}", (dx, -0.445, 0.24), (0.24, 0.025, 0.36), mat_dark)
        add_box(f"PumpBayDoor_{dx:.2f}", (dx, -0.452, 0.23), (0.20, 0.016, 0.33), mat_door)
        add_box(f"PumpLamp_{dx:.2f}", (dx, -0.455, 0.45), (0.16, 0.02, 0.028), mat_light)

    # Exterior suction/discharge pump manifold stubs on +Y face
    for dx in (-0.28, -0.09, 0.09, 0.28):
        add_cyl(f"PumpStub_{dx:.2f}", (dx, 0.47, 0.28), radius=0.042, depth=0.10, mat=mat_dark, vertices=14, rot=(math.pi / 2, 0, 0))
        add_cyl(f"PumpFlange_{dx:.2f}", (dx, 0.51, 0.28), radius=0.062, depth=0.02, mat=mat_yellow, vertices=14, rot=(math.pi / 2, 0, 0))

    # Side transformer / electrical switchgear bay on +X face
    add_box("TransformerBox", (0.48, 0.0, 0.24), (0.08, 0.40, 0.34), mat_dark)
    for dy in (-0.12, 0.0, 0.12):
        add_cyl(f"Insulator_{dy:.2f}", (0.48, dy, 0.46), radius=0.018, depth=0.10, mat=mat_door, vertices=10)

    export_glb("pump_house.glb")


if __name__ == "__main__":
    build_storage_tank()
    build_industrial_building()
    build_pipe_rack()
    build_valve_assembly()
    build_quadrotor()
    build_compressor_station()
    build_flare_stack()
    build_fractionator()
    build_horton_sphere()
    build_metering_skid()
    build_receiver_skid()
    build_cryogenic_vessel()
    build_transfer_gantry()
    build_compressor_train()
    build_cooling_cell()
    build_hyperbolic_tower()
    build_exhaust_stack()
    build_pump_house()
    print("All 18 GLB assets generated successfully.")
