#!/usr/bin/env python3
"""Generate a sim-specific MuJoCo MJCF model for simple_quad_v0.

The Onshape URDF remains the source of truth. This generator adds only the
simulation details that the export currently lacks: a floating base, primitive
contact geometry, conservative joint ranges, and ST3215-HS-like position
actuators.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPO_ROOT
    / "sim/robots/simple_quad_v0/onshape_export/assembly_1/urdf/assembly_1.urdf"
)
DEFAULT_OUTPUT = REPO_ROOT / "sim/robots/simple_quad_v0/mjcf/simple_quad_v0.xml"
DEFAULT_ACTUATOR_CONFIG = REPO_ROOT / "sim/configs/actuators/st3215_hs.yaml"

JOINT_RANGES = {
    "hip": (-0.9, 0.9),
    "knee": (-1.25, 1.25),
}


@dataclass(frozen=True)
class Inertial:
    mass: float
    pos: tuple[float, float, float]
    euler: tuple[float, float, float]
    diaginertia: tuple[float, float, float]


@dataclass(frozen=True)
class VisualBox:
    pos: tuple[float, float, float]
    euler: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    pos: tuple[float, float, float]
    euler: tuple[float, float, float]
    axis: tuple[float, float, float]


def parse_vec(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    parts = value.split()
    if len(parts) != 3:
        return default
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def rotate_rpy(vector: tuple[float, float, float], rpy: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a vector by fixed-axis URDF roll, pitch, yaw."""

    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    x, y, z = vector

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    x1 = x
    y1 = cx * y - sx * z
    z1 = sx * y + cx * z
    x2 = cy * x1 + sy * z1
    y2 = y1
    z2 = -sy * x1 + cy * z1
    return (cz * x2 - sz * y2, sz * x2 + cz * y2, z2)


def read_simple_yaml(path: Path) -> dict[str, dict[str, float | int | str]]:
    """Read the small YAML shape used by the actuator config without PyYAML."""

    data: dict[str, dict[str, float | int | str]] = {}
    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            data.setdefault(section, {})
            continue
        if section and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip()
            try:
                parsed: float | int | str
                parsed_float = float(value)
                parsed = int(parsed_float) if parsed_float.is_integer() else parsed_float
            except ValueError:
                parsed = value
            data[section][key.strip()] = parsed
    return data


def load_actuator_defaults(path: Path) -> dict[str, float]:
    defaults = {
        "torque_limit_nm": 1.37,
        "velocity_limit_rad_s": 8.0,
        "position_kp": 22.0,
        "position_kd": 0.8,
    }
    if not path.exists():
        return defaults
    raw = read_simple_yaml(path).get("simulation_defaults", {})
    for key in defaults:
        value = raw.get(key)
        if isinstance(value, (float, int)):
            defaults[key] = float(value)
    return defaults


def mesh_bbox(mesh_path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return min/max bounds from glTF POSITION accessors."""

    data = json.loads(mesh_path.read_text(encoding="utf-8"))
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    for accessor in data.get("accessors", []):
        if accessor.get("type") == "VEC3" and "coord_accessor" in accessor.get("name", ""):
            if "min" in accessor and "max" in accessor:
                mins.append([float(value) for value in accessor["min"]])
                maxs.append([float(value) for value in accessor["max"]])
    if not mins or not maxs:
        return ((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))
    lower = tuple(min(values[index] for values in mins) for index in range(3))
    upper = tuple(max(values[index] for values in maxs) for index in range(3))
    return lower, upper  # type: ignore[return-value]


def clamp_half_extents(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(max(0.004, abs(value)) for value in values)  # type: ignore[return-value]


def parse_inertial(link: ET.Element) -> Inertial | None:
    inertial = link.find("inertial")
    if inertial is None:
        return None
    mass_elem = inertial.find("mass")
    inertia_elem = inertial.find("inertia")
    origin = inertial.find("origin")
    if mass_elem is None or inertia_elem is None:
        return None
    mass = float(mass_elem.attrib["value"])
    pos = parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    euler = parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    diaginertia = (
        float(inertia_elem.attrib["ixx"]),
        float(inertia_elem.attrib["iyy"]),
        float(inertia_elem.attrib["izz"]),
    )
    return Inertial(mass=mass, pos=pos, euler=euler, diaginertia=diaginertia)


def parse_visual_box(link: ET.Element, package_dir: Path) -> VisualBox | None:
    visual = link.find("visual")
    if visual is None:
        return None
    mesh = visual.find(".//mesh")
    if mesh is None:
        return None
    filename = mesh.attrib.get("filename", "")
    if filename.startswith(f"package://{package_dir.name}/"):
        mesh_path = package_dir / filename.removeprefix(f"package://{package_dir.name}/")
    else:
        mesh_path = package_dir / filename
    if not mesh_path.exists():
        return None

    origin = visual.find("origin")
    origin_pos = parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    origin_euler = parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    lower, upper = mesh_bbox(mesh_path)
    center = tuple((lower[index] + upper[index]) * 0.5 for index in range(3))
    half_extents = clamp_half_extents(tuple((upper[index] - lower[index]) * 0.5 for index in range(3)))
    rotated_center = rotate_rpy(center, origin_euler)
    pos = tuple(origin_pos[index] + rotated_center[index] for index in range(3))

    color = visual.find(".//color")
    rgba = parse_vec4(color.attrib.get("rgba") if color is not None else None, (0.65, 0.65, 0.65, 1.0))
    return VisualBox(pos=pos, euler=origin_euler, size=half_extents, rgba=rgba)


def parse_vec4(value: str | None, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is None:
        return default
    parts = value.split()
    if len(parts) != 4:
        return default
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def parse_joint(joint: ET.Element) -> UrdfJoint:
    origin = joint.find("origin")
    parent = joint.find("parent")
    child = joint.find("child")
    axis = joint.find("axis")
    return UrdfJoint(
        name=joint.attrib["name"],
        joint_type=joint.attrib.get("type", "fixed"),
        parent=parent.attrib["link"] if parent is not None else "",
        child=child.attrib["link"] if child is not None else "",
        pos=parse_vec(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
        euler=parse_vec(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
        axis=parse_vec(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
    )


def joint_range(name: str) -> tuple[float, float]:
    if "knee" in name:
        return JOINT_RANGES["knee"]
    return JOINT_RANGES["hip"]


def add_inertial(body: ET.Element, inertial: Inertial | None) -> None:
    if inertial is None:
        return
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": format_vec(inertial.pos),
            "euler": format_vec(inertial.euler),
            "mass": f"{inertial.mass:.9g}",
            "diaginertia": format_vec(inertial.diaginertia),
        },
    )


def add_visual_collision(body: ET.Element, link_name: str, visual: VisualBox | None) -> None:
    if visual is None:
        return

    geom_attrs = {
        "type": "box",
        "pos": format_vec(visual.pos),
        "euler": format_vec(visual.euler),
        "size": format_vec(visual.size),
        "rgba": format_vec(visual.rgba),
    }
    ET.SubElement(
        body,
        "geom",
        {
            **geom_attrs,
            "name": f"{link_name}_visual",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
            "mass": "0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            **geom_attrs,
            "name": f"{link_name}_collision",
            "rgba": "0.35 0.35 0.35 0.18",
            "group": "3",
            "friction": "0.9 0.02 0.001",
            "mass": "0",
        },
    )


def build_mjcf(
    urdf_path: Path = DEFAULT_URDF,
    output_path: Path = DEFAULT_OUTPUT,
    actuator_config_path: Path = DEFAULT_ACTUATOR_CONFIG,
) -> Path:
    package_dir = urdf_path.parent.parent if urdf_path.parent.name == "urdf" else urdf_path.parent
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = [parse_joint(joint) for joint in root.findall("joint")]
    children: dict[str, list[UrdfJoint]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)

    fixed_root = next((joint for joint in joints if joint.parent == "root" and joint.child == "body"), None)
    base_link = fixed_root.child if fixed_root is not None else "body"
    if base_link not in links:
        raise SystemExit(f"Unable to find base link {base_link!r} in {urdf_path}")

    actuator_defaults = load_actuator_defaults(actuator_config_path)

    mjcf = ET.Element("mujoco", {"model": "simple_quad_v0"})
    ET.SubElement(
        mjcf,
        "compiler",
        {
            "angle": "radian",
            "coordinate": "local",
            "autolimits": "false",
        },
    )
    ET.SubElement(
        mjcf,
        "option",
        {
            "timestep": "0.002",
            "integrator": "RK4",
            "gravity": "0 0 -9.81",
        },
    )
    ET.SubElement(mjcf, "size", {"njmax": "100", "nconmax": "100"})

    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(asset, "material", {"name": "floor", "rgba": "0.28 0.30 0.31 1"})

    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key", "pos": "0 -1.5 2.5", "dir": "0 1 -1"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.02",
            "material": "floor",
            "friction": "0.9 0.02 0.001",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "track",
            "mode": "trackcom",
            "pos": "0.65 -0.9 0.45",
            "xyaxes": "0.82 0.57 0 -0.20 0.29 0.94",
        },
    )
    target_body = ET.SubElement(worldbody, "body", {"name": "target_marker", "mocap": "true", "pos": "0.45 0 0.025"})
    ET.SubElement(
        target_body,
        "geom",
        {
            "name": "target_marker_geom",
            "type": "sphere",
            "size": "0.035",
            "rgba": "1 0.12 0.08 0.75",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    actuator = ET.SubElement(mjcf, "actuator")

    actuated_joint_names: list[str] = []

    def add_link(parent_xml: ET.Element, link_name: str, incoming_joint: UrdfJoint | None) -> ET.Element:
        attrs = {"name": link_name}
        if incoming_joint is None:
            attrs["pos"] = "0 0 0.17"
        else:
            attrs["pos"] = format_vec(incoming_joint.pos)
            attrs["euler"] = format_vec(incoming_joint.euler)
        body = ET.SubElement(parent_xml, "body", attrs)

        if incoming_joint is None:
            ET.SubElement(body, "freejoint", {"name": "root_free"})
        elif incoming_joint.joint_type in {"continuous", "revolute"}:
            lower, upper = joint_range(incoming_joint.name)
            ET.SubElement(
                body,
                "joint",
                {
                    "name": incoming_joint.name,
                    "type": "hinge",
                    "axis": format_vec(incoming_joint.axis),
                    "range": f"{lower:.6g} {upper:.6g}",
                    "limited": "true",
                    "damping": f"{actuator_defaults['position_kd']:.6g}",
                    "armature": "0.002",
                    "frictionloss": "0.01",
                },
            )
            actuated_joint_names.append(incoming_joint.name)

        link = links[link_name]
        add_inertial(body, parse_inertial(link))
        add_visual_collision(body, link_name, parse_visual_box(link, package_dir))

        for child_joint in children.get(link_name, []):
            if child_joint.joint_type == "fixed":
                for grandchild_joint in children.get(child_joint.child, []):
                    add_link(body, grandchild_joint.child, grandchild_joint)
            else:
                add_link(body, child_joint.child, child_joint)
        return body

    add_link(worldbody, base_link, None)

    for name in actuated_joint_names:
        lower, upper = joint_range(name)
        torque = actuator_defaults["torque_limit_nm"]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{name}_servo",
                "joint": name,
                "kp": f"{actuator_defaults['position_kp']:.6g}",
                "ctrllimited": "true",
                "ctrlrange": f"{lower:.6g} {upper:.6g}",
                "forcelimited": "true",
                "forcerange": f"{-torque:.6g} {torque:.6g}",
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(mjcf, space="  ")
    ET.ElementTree(mjcf).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simple_quad_v0 MuJoCo MJCF from the Onshape URDF.")
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Source URDF path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output MJCF XML path.")
    parser.add_argument(
        "--actuator-config",
        default=str(DEFAULT_ACTUATOR_CONFIG),
        help="ST3215-HS actuator config YAML path.",
    )
    args = parser.parse_args()

    output = build_mjcf(
        urdf_path=Path(args.urdf).resolve(),
        output_path=Path(args.output).resolve(),
        actuator_config_path=Path(args.actuator_config).resolve(),
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
