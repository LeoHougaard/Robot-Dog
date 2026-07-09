#!/usr/bin/env python3
"""Validate an Onshape URDF export package.

This script intentionally uses only the Python standard library so it can run
before the MuJoCo/RL dependencies are installed.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_PACKAGE_DIR = "sim/robots/simple_quad_v0/onshape_export/assembly_1"
EXPECTED_SIMPLE_QUAD_JOINTS = {
    "revolute_back_left_hip",
    "revolute_back_left_knee",
    "revolute_back_right_hip",
    "revolute_back_right_knee",
    "revolute_front_left_hip",
    "revolute_front_left_knee",
    "revolute_front_right_hip",
    "revolute_front_right_knee",
}
KNOWN_JOINT_TYPES = {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}
AXIS_JOINT_TYPES = {"revolute", "continuous", "prismatic"}
LIMIT_JOINT_TYPES = {"revolute", "prismatic"}
INERTIA_COMPONENTS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
INERTIA_EPSILON = 1e-18


@dataclass(frozen=True)
class UrdfPackage:
    package_dir: Path
    urdf_path: Path


def find_urdf(package_dir: Path) -> Path:
    """Find a single URDF in a flat or common nested Onshape export layout."""

    if package_dir.is_file():
        if package_dir.suffix.lower() == ".urdf":
            return package_dir.resolve()
        raise SystemExit(f"Not a URDF file: {package_dir}")

    search_groups = [
        sorted(package_dir.glob("*.urdf")),
        sorted((package_dir / "urdf").glob("*.urdf")),
        sorted(package_dir.glob("*/urdf/*.urdf")),
        sorted(package_dir.glob("**/*.urdf")),
    ]
    for urdfs in search_groups:
        unique = sorted(set(urdfs))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            names = ", ".join(str(path.relative_to(package_dir)) for path in unique)
            raise SystemExit(f"More than one .urdf file found in {package_dir}: {names}")
    raise SystemExit(f"No .urdf file found in {package_dir}")


def infer_package_dir(path: Path, urdf_path: Path | None = None) -> UrdfPackage:
    """Return the package root and URDF path for an Onshape export.

    Onshape often exports package roots like assembly_1/ with the URDF in
    assembly_1/urdf/assembly_1.urdf and meshes in assembly_1/meshes/.
    """

    package_dir = path.resolve()
    if urdf_path is None:
        urdf = find_urdf(package_dir)
    else:
        urdf = urdf_path.resolve()
        if not urdf.exists():
            raise SystemExit(f"URDF does not exist: {urdf}")

    if package_dir.is_file():
        package_dir = package_dir.parent

    if urdf.parent.name == "urdf" and (urdf.parent.parent / "meshes").exists():
        package_dir = urdf.parent.parent

    return UrdfPackage(package_dir=package_dir, urdf_path=urdf)


def append_candidate(candidates: list[Path], candidate: Path) -> None:
    resolved = candidate.resolve()
    if resolved not in candidates:
        candidates.append(resolved)


def file_path_from_uri(filename: str) -> Path:
    parsed = urlparse(filename)
    path_text = unquote(parsed.path)
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        path_text = f"//{parsed.netloc}{path_text}"
    if len(path_text) >= 3 and path_text[0] == "/" and path_text[2] == ":":
        path_text = path_text[1:]
    return Path(path_text)


def mesh_path_from_filename(filename: str, package_dir: Path, urdf_dir: Path) -> Path | None:
    raw_filename = unquote(filename)
    raw_path = Path(raw_filename)
    if raw_path.is_absolute():
        return raw_path.resolve()

    parsed = urlparse(filename)
    if parsed.scheme in {"http", "https"}:
        return None

    if parsed.scheme == "package":
        package_name = unquote(parsed.netloc)
        raw_path = unquote(parsed.path.lstrip("/"))
        parts = Path(raw_path).parts
        candidates: list[Path] = []
        if parts:
            append_candidate(candidates, package_dir.joinpath(*parts))
            if parts[0] == package_dir.name:
                append_candidate(candidates, package_dir.joinpath(*parts[1:]))
        if package_name:
            append_candidate(candidates, package_dir.parent.joinpath(package_name, *parts))
            append_candidate(candidates, package_dir.joinpath(package_name, *parts))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else package_dir

    if parsed.scheme == "file":
        return file_path_from_uri(filename).resolve()

    if parsed.scheme:
        return package_dir / raw_filename

    candidates = []
    append_candidate(candidates, urdf_dir / raw_filename)
    append_candidate(candidates, package_dir / raw_filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_xyz(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 3:
        return None
    values = [parse_float(part) for part in parts]
    if any(parsed is None for parsed in values):
        return None
    return tuple(parsed for parsed in values if parsed is not None)  # type: ignore[return-value]


def fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6g}"


def inertia_determinant(values: dict[str, float]) -> float:
    ixx = values["ixx"]
    ixy = values["ixy"]
    ixz = values["ixz"]
    iyy = values["iyy"]
    iyz = values["iyz"]
    izz = values["izz"]
    return (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )


def is_positive_definite_inertia(values: dict[str, float]) -> bool:
    minor_1 = values["ixx"]
    minor_2 = values["ixx"] * values["iyy"] - values["ixy"] * values["ixy"]
    determinant = inertia_determinant(values)
    return (
        minor_1 > INERTIA_EPSILON
        and minor_2 > INERTIA_EPSILON
        and determinant > INERTIA_EPSILON
    )


def check_inertia_matrix(link_name: str, inertial: ET.Element, warnings: list[str]) -> None:
    mass_elem = inertial.find("mass")
    mass = parse_float(mass_elem.attrib.get("value")) if mass_elem is not None else None
    if mass is None or mass <= 0:
        warnings.append(f"Link {link_name} has missing or non-positive mass.")

    inertia = inertial.find("inertia")
    if inertia is None:
        warnings.append(f"Link {link_name} has no inertia tensor.")
        return

    inertia_values = {
        component: parse_float(inertia.attrib.get(component))
        for component in INERTIA_COMPONENTS
    }
    missing = [
        component
        for component, value in inertia_values.items()
        if value is None
    ]
    if missing:
        warnings.append(
            f"Link {link_name} has missing or invalid inertia components: {', '.join(missing)}."
        )
        return

    values = {
        component: value
        for component, value in inertia_values.items()
        if value is not None
    }
    if any(values[diagonal] <= 0 for diagonal in ("ixx", "iyy", "izz")):
        warnings.append(f"Link {link_name} has non-positive principal inertia.")
        return

    if not is_positive_definite_inertia(values):
        warnings.append(f"Link {link_name} inertia tensor is not positive definite.")
        return

    ixx = values["ixx"]
    iyy = values["iyy"]
    izz = values["izz"]
    if ixx + iyy < izz or ixx + izz < iyy or iyy + izz < ixx:
        warnings.append(
            f"Link {link_name} inertia may violate triangle inequality "
            f"(ixx={ixx:g}, iyy={iyy:g}, izz={izz:g})."
        )


def parent_child_maps(joints: list[ET.Element]) -> tuple[dict[str, str], Counter[str]]:
    child_to_parent: dict[str, str] = {}
    parent_counts: Counter[str] = Counter()
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link") if parent is not None else None
        child_name = child.attrib.get("link") if child is not None else None
        if parent_name and child_name:
            child_to_parent[child_name] = parent_name
            parent_counts[parent_name] += 1
    return child_to_parent, parent_counts


def check_urdf(urdf_path: Path, package_dir: Path, expected_simple_quad: bool) -> int:
    root = ET.parse(urdf_path).getroot()

    links = root.findall("link")
    joints = root.findall("joint")
    revolute_joints = [j for j in joints if j.attrib.get("type") in {"revolute", "continuous"}]
    link_name_values = [link.attrib.get("name", "") for link in links]
    link_names = {name for name in link_name_values if name}
    child_to_parent, _parent_counts = parent_child_maps(joints)
    root_links = sorted(link for link in link_names if link and link not in child_to_parent)

    warnings: list[str] = []
    errors: list[str] = []
    sim_notes: list[str] = []

    if not links:
        errors.append("URDF has no links.")
    if not joints:
        warnings.append("URDF has no joints.")
    if not revolute_joints:
        warnings.append("URDF has no revolute or continuous joints.")

    if "" in link_name_values:
        errors.append("A link is missing its name attribute.")
    for name, count in Counter(link_name_values).items():
        if name and count > 1:
            errors.append(f"Duplicate link name: {name}")

    for link in links:
        name = link.attrib.get("name", "<unnamed>")
        inertial = link.find("inertial")
        if inertial is None:
            warnings.append(f"Link {name} has no inertial block.")
        else:
            check_inertia_matrix(name, inertial, warnings)

        if link.find("collision") is None:
            sim_notes.append(f"Link {name} has no collision geometry.")

    collision_links = sum(1 for link in links if link.find("collision") is not None)
    if links and collision_links == 0:
        sim_notes.append("URDF has no collision elements; add simple collision geoms for MuJoCo contact.")

    joint_names = {joint.attrib.get("name", "") for joint in revolute_joints}
    joint_name_values = [joint.attrib.get("name", "") for joint in joints]
    if "" in joint_name_values:
        errors.append("A joint is missing its name attribute.")
    for name, count in Counter(joint_name_values).items():
        if name and count > 1:
            errors.append(f"Duplicate joint name: {name}")

    if expected_simple_quad:
        missing = sorted(EXPECTED_SIMPLE_QUAD_JOINTS - joint_names)
        extra = sorted(name for name in joint_names - EXPECTED_SIMPLE_QUAD_JOINTS if name)
        if missing:
            errors.append("Missing expected simple_quad_v0 joints: " + ", ".join(missing))
        if extra:
            warnings.append("Unexpected actuated joint names: " + ", ".join(extra))

    child_to_joint: dict[str, str] = {}
    for joint in joints:
        name = joint.attrib.get("name", "<unnamed>")
        joint_type = joint.attrib.get("type", "<missing>")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link") if parent is not None else None
        child_name = child.attrib.get("link") if child is not None else None
        if joint_type not in KNOWN_JOINT_TYPES:
            errors.append(f"Joint {name} has unsupported type: {joint_type}")
        if parent_name not in link_names:
            errors.append(f"Joint {name} parent link does not exist: {parent_name}")
        if child_name not in link_names:
            errors.append(f"Joint {name} child link does not exist: {child_name}")
        elif child_name:
            previous_joint = child_to_joint.get(child_name)
            if previous_joint:
                errors.append(
                    f"Link {child_name} is child of multiple joints: {previous_joint}, {name}"
                )
            else:
                child_to_joint[child_name] = name

        if joint_type in AXIS_JOINT_TYPES:
            axis = parse_xyz(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else None
            if axis is None:
                warnings.append(f"Joint {name} has no valid axis.")
            else:
                norm = math.sqrt(sum(component * component for component in axis))
                if not 0.99 <= norm <= 1.01:
                    warnings.append(f"Joint {name} axis is not normalized: {axis}.")

            limit = joint.find("limit")
            if joint_type in LIMIT_JOINT_TYPES and limit is None:
                warnings.append(f"{joint_type.capitalize()} joint {name} has no limit.")
            if joint_type == "continuous":
                sim_notes.append(f"Continuous joint {name} needs sim-only range/position target limits.")
            if limit is not None:
                lower = parse_float(limit.attrib.get("lower"))
                upper = parse_float(limit.attrib.get("upper"))
                effort = parse_float(limit.attrib.get("effort"))
                velocity = parse_float(limit.attrib.get("velocity"))
                if joint_type in LIMIT_JOINT_TYPES and (lower is None or upper is None):
                    warnings.append(f"Joint {name} has missing lower/upper position limits.")
                elif lower is not None and upper is not None and lower >= upper:
                    warnings.append(f"Joint {name} has invalid limit range: {lower:g} >= {upper:g}.")
                if effort is not None and effort <= 0:
                    warnings.append(f"Joint {name} has non-positive effort limit.")
                if velocity is not None and velocity <= 0:
                    warnings.append(f"Joint {name} has non-positive velocity limit.")

    mesh_filenames = [
        mesh.attrib["filename"]
        for mesh in root.findall(".//mesh")
        if "filename" in mesh.attrib
    ]
    missing_meshes = []
    remote_meshes = 0
    for filename in mesh_filenames:
        parsed = urlparse(filename)
        raw_path = Path(unquote(filename))
        if (
            parsed.scheme
            and parsed.scheme not in {"package", "file", "http", "https"}
            and not raw_path.is_absolute()
        ):
            errors.append(f"Unsupported mesh URI scheme '{parsed.scheme}' in {filename}")
            continue
        mesh_path = mesh_path_from_filename(filename, package_dir, urdf_path.parent)
        if mesh_path is None:
            remote_meshes += 1
        elif not mesh_path.exists():
            missing_meshes.append(filename)

    for filename in sorted(set(missing_meshes)):
        errors.append(f"Missing mesh referenced by URDF: {filename}")

    if not root_links:
        errors.append("URDF has no parentless root link; the joint graph may contain a cycle.")
    elif len(root_links) > 1:
        errors.append("URDF has multiple root links: " + ", ".join(root_links))
    elif root_links == ["root"] and "body" in child_to_parent and child_to_parent["body"] == "root":
        sim_notes.append("URDF uses fixed root -> body; generated MuJoCo should add a freejoint to body.")
    else:
        sim_notes.append(f"URDF root link is {root_links[0]}; no floating/free base joint is declared.")

    print(f"Package root: {package_dir}")
    print(f"URDF: {urdf_path}")
    print(f"Robot name: {root.attrib.get('name', '<unnamed>')}")
    print(f"Links: {len(links)}")
    print(f"Joints: {len(joints)}")
    print(f"Revolute/continuous joints: {len(revolute_joints)}")
    print(f"Mesh references: {len(mesh_filenames)} ({len(set(mesh_filenames))} unique)")
    print(f"Collision links: {collision_links} / {len(links)}")
    print(f"Root links: {', '.join(root_links) if root_links else '<none>'}")
    if remote_meshes:
        print(f"Remote mesh references skipped: {remote_meshes}")

    if revolute_joints:
        print("\nActuated joint summary:")
        for joint in revolute_joints:
            name = joint.attrib.get("name", "<unnamed>")
            joint_type = joint.attrib.get("type", "<missing>")
            axis = joint.find("axis").attrib.get("xyz") if joint.find("axis") is not None else "missing"
            limit = joint.find("limit")
            if limit is None:
                limit_text = "no URDF limit"
            else:
                limit_text = (
                    f"lower={fmt(parse_float(limit.attrib.get('lower')))} "
                    f"upper={fmt(parse_float(limit.attrib.get('upper')))} "
                    f"effort={fmt(parse_float(limit.attrib.get('effort')))} "
                    f"velocity={fmt(parse_float(limit.attrib.get('velocity')))}"
                )
            print(f"  - {name}: type={joint_type}, axis={axis}, {limit_text}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if sim_notes:
        print("\nSimulation notes:")
        for note in sim_notes:
            print(f"  - {note}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nPackage check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check an Onshape URDF export package.")
    parser.add_argument(
        "package_dir",
        nargs="?",
        default=DEFAULT_PACKAGE_DIR,
        help="Onshape package root, e.g. sim/robots/simple_quad_v0/onshape_export/assembly_1.",
    )
    parser.add_argument("--urdf", help="Specific URDF file to check.")
    parser.add_argument(
        "--no-simple-quad-expectations",
        action="store_true",
        help="Disable simple_quad_v0 expected joint-name checks.",
    )
    args = parser.parse_args()

    package_path = Path(args.package_dir).resolve()
    if not package_path.exists():
        raise SystemExit(f"Package directory does not exist: {package_path}")

    package = infer_package_dir(package_path, Path(args.urdf) if args.urdf else None)
    return check_urdf(
        package.urdf_path,
        package.package_dir,
        expected_simple_quad=not args.no_simple_quad_expectations,
    )


if __name__ == "__main__":
    sys.exit(main())
