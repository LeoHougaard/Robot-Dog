"""Procedural sim-only terrain helpers for the simple quadruped environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree as ET


HFIELD_NAME = "procedural_terrain"
FLOOR_GEOM_NAME = "floor"


@dataclass(frozen=True)
class TerrainConfig:
    name: str
    label: str
    amplitude_m: float
    slope_m: float
    roughness_m: float
    bump_height_m: float
    bump_count: int
    friction_min: float
    friction_max: float
    nrow: int = 65
    ncol: int = 65
    size_x: float = 2.0
    size_y: float = 2.0
    hfield_base_m: float = 0.025

    @property
    def uses_hfield(self) -> bool:
        return self.name != "flat"

    @property
    def nominal_friction(self) -> float:
        return 0.5 * (self.friction_min + self.friction_max)

    @property
    def friction_string(self) -> str:
        return f"{self.nominal_friction:.3f} 0.02 0.001"

    @property
    def hfield_z_scale(self) -> float:
        return max(0.002, 2.0 * self.amplitude_m)


TERRAIN_CONFIGS = {
    "flat": TerrainConfig(
        name="flat",
        label="flat",
        amplitude_m=0.0,
        slope_m=0.0,
        roughness_m=0.0,
        bump_height_m=0.0,
        bump_count=0,
        friction_min=0.9,
        friction_max=0.9,
    ),
    "mild": TerrainConfig(
        name="mild",
        label="mild rough",
        amplitude_m=0.010,
        slope_m=0.004,
        roughness_m=0.004,
        bump_height_m=0.006,
        bump_count=4,
        friction_min=0.80,
        friction_max=1.05,
    ),
    "rough": TerrainConfig(
        name="rough",
        label="rough",
        amplitude_m=0.020,
        slope_m=0.010,
        roughness_m=0.008,
        bump_height_m=0.014,
        bump_count=8,
        friction_min=0.65,
        friction_max=1.15,
    ),
    "hard": TerrainConfig(
        name="hard",
        label="hard rough",
        amplitude_m=0.032,
        slope_m=0.018,
        roughness_m=0.014,
        bump_height_m=0.022,
        bump_count=12,
        friction_min=0.55,
        friction_max=1.25,
    ),
}

TERRAIN_ALIASES = {
    "none": "flat",
    "plane": "flat",
    "level": "flat",
    "low": "mild",
    "mild_rough": "mild",
    "medium": "rough",
    "uneven": "rough",
    "high": "hard",
    "curriculum": "curriculum",
}


def normalize_terrain_name(name: str | None) -> str:
    value = (name or "flat").strip().lower().replace("-", "_")
    value = TERRAIN_ALIASES.get(value, value)
    if value == "curriculum":
        return value
    if value not in TERRAIN_CONFIGS:
        allowed = ", ".join([*TERRAIN_CONFIGS, "curriculum"])
        raise ValueError(f"Unknown terrain '{name}'. Expected one of: {allowed}.")
    return value


def get_terrain_config(name: str | None) -> TerrainConfig:
    normalized = normalize_terrain_name(name)
    if normalized == "curriculum":
        return TERRAIN_CONFIGS["rough"]
    return TERRAIN_CONFIGS[normalized]


def curriculum_terrain_name(
    terrain: str | None,
    curriculum: str | tuple[str, ...] | list[str] | bool | None,
    episode_index: int,
) -> str:
    normalized = normalize_terrain_name(terrain)
    if isinstance(curriculum, (tuple, list)):
        levels = [normalize_terrain_name(str(level)) for level in curriculum if str(level).strip()]
        if not levels:
            return normalized
        if len(levels) == 1:
            return levels[0]
        span = 20
        index = min(len(levels) - 1, episode_index // span)
        return levels[index]
    if isinstance(curriculum, bool):
        curriculum_value = "auto" if curriculum else ""
    else:
        curriculum_value = (curriculum or "").strip().lower().replace("_", "-")

    if normalized != "curriculum" and curriculum_value in ("", "none", "off", "false", "0"):
        return normalized

    if curriculum_value in ("", "auto", "flat-mild-rough", "flat-to-rough", "on", "true", "1"):
        if episode_index < 30:
            return "flat"
        if episode_index < 90:
            return "mild"
        return "rough"
    if curriculum_value in ("flat-mild", "flat-to-mild"):
        return "flat" if episode_index < 30 else "mild"
    if curriculum_value in ("mild-rough", "mild-to-rough"):
        return "mild" if episode_index < 45 else "rough"
    return normalize_terrain_name(curriculum_value)


def generate_heightfield(config: TerrainConfig, seed: int):
    import numpy as np

    if not config.uses_hfield:
        return np.zeros((config.nrow, config.ncol), dtype=np.float32)

    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, config.ncol, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, config.nrow, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    slope_angle = float(rng.uniform(-3.141592653589793, 3.141592653589793))
    heights = config.slope_m * (np.cos(slope_angle) * xx + np.sin(slope_angle) * yy)

    noise = rng.normal(0.0, 1.0, size=(config.nrow, config.ncol)).astype(np.float32)
    for _ in range(5):
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, -1, axis=0)
            + np.roll(noise, 1, axis=1)
            + np.roll(noise, -1, axis=1)
        ) / 5.0
    max_abs_noise = float(np.max(np.abs(noise)))
    if max_abs_noise > 1e-6:
        heights += config.roughness_m * noise / max_abs_noise

    for _ in range(config.bump_count):
        cx = float(rng.uniform(-0.85, 0.85))
        cy = float(rng.uniform(-0.85, 0.85))
        radius = float(rng.uniform(0.08, 0.24))
        sign = float(rng.choice([-1.0, 1.0]))
        height = sign * float(rng.uniform(0.35, 1.0)) * config.bump_height_m
        heights += height * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius * radius)))

    center_height = float(heights[config.nrow // 2, config.ncol // 2])
    heights = heights - center_height
    heights = np.clip(heights, -config.amplitude_m, config.amplitude_m)
    return heights.astype(np.float32)


def heightfield_to_mujoco_data(heights, config: TerrainConfig):
    import numpy as np

    if not config.uses_hfield:
        return np.zeros((config.nrow, config.ncol), dtype=np.float32)
    return np.clip((heights + config.amplitude_m) / config.hfield_z_scale, 0.0, 1.0).astype(np.float32)


def terrain_xml(base_xml: str, config: TerrainConfig) -> str:
    if not config.uses_hfield:
        return base_xml

    root = ET.fromstring(base_xml)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    for existing in list(asset.findall("hfield")):
        if existing.get("name") == HFIELD_NAME:
            asset.remove(existing)

    ET.SubElement(
        asset,
        "hfield",
        {
            "name": HFIELD_NAME,
            "nrow": str(config.nrow),
            "ncol": str(config.ncol),
            "size": (
                f"{config.size_x:.3f} {config.size_y:.3f} "
                f"{config.hfield_z_scale:.4f} {config.hfield_base_m:.4f}"
            ),
        },
    )

    floor = root.find(f".//geom[@name='{FLOOR_GEOM_NAME}']")
    if floor is None:
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")
        floor = ET.SubElement(worldbody, "geom", {"name": FLOOR_GEOM_NAME})

    floor.set("type", "hfield")
    floor.set("hfield", HFIELD_NAME)
    floor.set("pos", f"0 0 {-config.amplitude_m:.5f}")
    floor.set("friction", config.friction_string)
    floor.attrib.pop("size", None)
    return ET.tostring(root, encoding="unicode")


def apply_terrain_to_model(
    model: Any,
    mujoco_module: Any,
    config: TerrainConfig,
    seed: int,
    randomize_friction: bool = False,
    hfield_config: TerrainConfig | None = None,
) -> dict[str, float | str]:
    import numpy as np

    scale_config = hfield_config or config
    hfield_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_HFIELD, HFIELD_NAME)
    if hfield_id >= 0:
        adr = int(model.hfield_adr[hfield_id])
        count = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
        if config.uses_hfield:
            heights = generate_heightfield(config, seed)
        else:
            heights = np.zeros((scale_config.nrow, scale_config.ncol), dtype=np.float32)
        model.hfield_data[adr : adr + count] = heightfield_to_mujoco_data(heights, scale_config).reshape(-1)

    friction = config.nominal_friction
    if randomize_friction and config.friction_min < config.friction_max:
        rng = np.random.default_rng(seed + 7919)
        friction = float(rng.uniform(config.friction_min, config.friction_max))

    floor_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    if floor_id >= 0:
        model.geom_friction[floor_id, 0] = friction
        model.geom_friction[floor_id, 1] = 0.02
        model.geom_friction[floor_id, 2] = 0.001

    return {
        "terrain": config.name,
        "terrain_seed": float(seed),
        "floor_friction": float(friction),
    }
