#!/usr/bin/env python3
"""Replay a recorded simple_quad_v0 action trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_ppm(path: Path, rgb) -> None:
    height, width = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(rgb.astype("uint8").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay recorded simple_quad_v0 actions.")
    parser.add_argument("recording", help=".npz recording from sim/evaluate.py --record")
    parser.add_argument("--max-steps", type=int, help="Optional cap on replayed actions.")
    parser.add_argument("--render-rgb", help="Optional final-frame PPM output path.")
    args = parser.parse_args()

    try:
        import numpy as np
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    recording = np.load(Path(args.recording).resolve(), allow_pickle=False)
    actions = recording["action"]
    if args.max_steps is not None:
        actions = actions[: args.max_steps]

    env = SimpleQuadStandEnv(render_mode="rgb_array" if args.render_rgb else None, randomize_actuators=False)
    _, _ = env.reset(seed=1)
    total_reward = 0.0
    steps = 0
    info = {}
    terminated = truncated = False
    for action in actions:
        if terminated or truncated:
            break
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    if args.render_rgb:
        frame = env.render()
        if frame is not None:
            write_ppm(Path(args.render_rgb).resolve(), frame)
            print(f"rendered={Path(args.render_rgb).resolve()}")

    env.close()
    print(
        f"replayed_steps={steps} return={total_reward:.3f} "
        f"base_height={info.get('base_height', float('nan')):.3f} "
        f"upright={info.get('upright', float('nan')):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
