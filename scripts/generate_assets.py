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


def add_box(
    name: str,
    loc: tuple[float, float, float],
    scale: tuple[float, float, float],
    mat: bpy.types.Material,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    assign_mat(obj, mat)
    return obj


def add_cyl(
    name: str,
    loc: tuple[float, float, float],
    radius: float,
    depth: float,
    mat: bpy.types.Material,
    vertices: int = 24,
    rot: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name
    assign_mat(obj, mat)
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

    # Concrete plinth base
    add_cyl("Plinth", (0, 0, 0.03), radius=0.49, depth=0.06, mat=mat_concrete, vertices=32)

    # Main cylindrical tank shell
    add_cyl("TankBody", (0, 0, 0.46), radius=0.44, depth=0.80, mat=mat_tank, vertices=36)

    # Circumferential stiffening / weld rings
    for z in (0.18, 0.38, 0.58, 0.78):
        add_cyl(f"Ring_{z:.2f}", (0, 0, z), radius=0.452, depth=0.018, mat=mat_band, vertices=36)

    # Painted industrial hazard band near top
    add_cyl("HazardBand", (0, 0, 0.70), radius=0.445, depth=0.065, mat=mat_accent, vertices=36)

    # Shallow domed roof (squashed UV sphere)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=12, radius=0.44, location=(0, 0, 0.86))
    dome = bpy.context.active_object
    dome.name = "TankDome"
    dome.scale = (1.0, 1.0, 0.24)
    assign_mat(dome, mat_tank)

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

    # Rooftop HVAC condenser units & exhaust stacks
    add_box("HVAC_1", (-0.20, 0.12, 0.915), (0.26, 0.22, 0.11), mat_door)
    add_box("HVAC_2", (0.18, -0.14, 0.91), (0.22, 0.20, 0.10), mat_door)
    add_cyl("ExhaustStack_1", (0.28, 0.24, 0.93), radius=0.04, depth=0.14, mat=mat_trim, vertices=14)
    add_cyl("ExhaustStack_2", (0.16, 0.24, 0.92), radius=0.032, depth=0.12, mat=mat_trim, vertices=14)

    # Industrial rollup bay doors on -Y and +Y faces
    for y_sign in (-1.0, 1.0):
        add_box(f"BayDoorFrame_{y_sign}", (0.0, y_sign * 0.476, 0.25), (0.28, 0.02, 0.38), mat_trim)
        add_box(f"BayDoor_{y_sign}", (0.0, y_sign * 0.478, 0.24), (0.24, 0.015, 0.35), mat_door)
        # Yellow safety bollards flanking the bay door
        for bx in (-0.16, 0.16):
            add_cyl(
                f"Bollard_{y_sign}_{bx}",
                (bx, y_sign * 0.505, 0.11),
                radius=0.016,
                depth=0.14,
                mat=mat_yellow,
                vertices=10,
            )

    # Glowing control-room window strip on +X and -X faces
    for x_sign in (-1.0, 1.0):
        add_box(f"WinFrame_{x_sign}", (x_sign * 0.476, 0.0, 0.62), (0.025, 0.60, 0.16), mat_trim)
        add_box(f"WinGlass_{x_sign}", (x_sign * 0.482, 0.0, 0.62), (0.015, 0.54, 0.11), mat_window)

    # Rooftop comms mast & red aviation beacon
    add_cyl("CommsMast", (-0.36, -0.34, 0.94), radius=0.012, depth=0.16, mat=mat_trim, vertices=8)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=8, radius=0.022, location=(-0.36, -0.34, 1.02))
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
# 5. Industrial Inspection Quadrotor (Matrice-style, ~2.2m visual span)
# ---------------------------------------------------------------------------
def build_quadrotor() -> None:
    clear_scene()
    mat_carbon = make_mat("CarbonFiber", (0.08, 0.09, 0.11), metallic=0.35, roughness=0.32)
    mat_shell = make_mat("DroneShell", (0.20, 0.23, 0.28), metallic=0.45, roughness=0.30)
    mat_alu = make_mat("Aluminum", (0.72, 0.76, 0.80), metallic=0.88, roughness=0.24)
    mat_accent = make_mat(
        "AccentLED",
        (0.15, 0.85, 0.95),
        metallic=0.1,
        roughness=0.2,
        emission=(0.15, 0.85, 0.95),
        emission_strength=2.2,
    )
    mat_prop = make_mat("PropBlade", (0.12, 0.13, 0.15), metallic=0.2, roughness=0.42)

    # Central fuselage (nose faces +X in Blender -> +X in Three.js)
    add_box("FuselageCore", (0, 0, 0.04), (0.78, 0.50, 0.24), mat_carbon)
    add_box("TopCanopy", (0.04, 0, 0.18), (0.62, 0.38, 0.12), mat_shell)

    # Glowing agent-color accent strip + GPS mast
    accent = add_box("AccentLED", (0.04, 0, 0.255), (0.50, 0.24, 0.04), mat_accent)
    accent.name = "AccentLED"
    add_cyl("GPSMast", (-0.18, 0, 0.34), radius=0.018, depth=0.18, mat=mat_alu, vertices=8)
    add_cyl("GPSPuck", (-0.18, 0, 0.44), radius=0.085, depth=0.04, mat=mat_shell, vertices=14)

    # Under-slung optical + gas sniffer gimbal payload
    add_box("GimbalMount", (0.24, 0, -0.14), (0.16, 0.16, 0.12), mat_alu)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=14, ring_count=10, radius=0.13, location=(0.28, 0, -0.25))
    cam = bpy.context.active_object
    cam.name = "SensorGimbal"
    assign_mat(cam, mat_carbon)
    add_cyl("SnifferTube", (0.44, 0, -0.25), radius=0.032, depth=0.22, mat=mat_alu, vertices=10, rot=(0, math.pi / 2, 0))

    # Landing skids
    for sy in (-0.34, 0.34):
        add_cyl(f"SkidBar_{sy}", (0.0, sy, -0.40), radius=0.024, depth=0.88, mat=mat_carbon, vertices=10, rot=(0, math.pi / 2, 0))
        for sx in (-0.22, 0.22):
            add_cyl(f"SkidLeg_{sy}_{sx}", (sx, sy * 0.82, -0.22), radius=0.02, depth=0.38, mat=mat_carbon, vertices=8, rot=(sy * 0.45, 0, 0))

    # 4 diagonal motor booms, aluminum motor bells, and separately named rotors
    arm_angles = [math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]
    arm_len = 0.95
    for i, ang in enumerate(arm_angles):
        cx = math.cos(ang) * (arm_len * 0.5)
        cy = math.sin(ang) * (arm_len * 0.5)
        mx = math.cos(ang) * arm_len
        my = math.sin(ang) * arm_len

        # Tubular carbon arm
        add_cyl(
            f"Arm_{i}",
            (cx, cy, 0.08),
            radius=0.036,
            depth=arm_len,
            mat=mat_carbon,
            vertices=10,
            rot=(0, math.pi / 2, ang),
        )
        # Motor bell & nav LED pod underneath
        add_cyl(f"Motor_{i}", (mx, my, 0.13), radius=0.09, depth=0.12, mat=mat_alu, vertices=14)
        add_cyl(f"NavLED_{i}", (mx, my, 0.04), radius=0.05, depth=0.04, mat=mat_accent, vertices=10)

        # Twin-blade propeller (named Rotor_0..3 so Three.js can spin each one)
        rotor = add_box(f"Rotor_{i}", (mx, my, 0.21), (0.86, 0.085, 0.018), mat_prop)
        rotor.name = f"Rotor_{i}"

    export_glb("quadrotor.glb")


if __name__ == "__main__":
    build_storage_tank()
    build_industrial_building()
    build_pipe_rack()
    build_valve_assembly()
    build_quadrotor()
    print("All GLB assets generated successfully.")
