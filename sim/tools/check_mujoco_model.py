#!/usr/bin/env python3
"""Load and optionally render the generated simple_quad_v0 MuJoCo model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "sim/robots/simple_quad_v0/mjcf/simple_quad_v0.xml"


def write_ppm(path: Path, rgb) -> None:
    height, width = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(rgb.astype("uint8").tobytes())


def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from sim.tools.build_simple_quad_mjcf import build_mjcf

    return build_mjcf(output_path=model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that MuJoCo can load the simple quadruped MJCF.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Generated MJCF model path.")
    parser.add_argument("--steps", type=int, default=100, help="Number of zero-control simulation steps.")
    parser.add_argument("--render", help="Optional PPM output path for one rendered frame.")
    args = parser.parse_args()

    model_path = ensure_model(Path(args.model).resolve())

    try:
        import mujoco
    except ImportError as exc:
        print(f"MuJoCo import failed: {exc}", file=sys.stderr)
        print("Install sim dependencies, then rerun this command.", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    for _ in range(max(0, args.steps)):
        mujoco.mj_step(model, data)

    print(f"model_path={model_path}")
    print(f"nbody={model.nbody}")
    print(f"njnt={model.njnt}")
    print(f"nq={model.nq}")
    print(f"nv={model.nv}")
    print(f"nu={model.nu}")
    print(f"sim_time={data.time:.3f}")
    print(f"base_z={float(data.qpos[2]):.4f}")

    if args.render:
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera="track")
        rgb = renderer.render()
        write_ppm(Path(args.render).resolve(), rgb)
        renderer.close()
        print(f"rendered={Path(args.render).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
