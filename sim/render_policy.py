#!/usr/bin/env python3
"""Render rollout snapshots for a simple_quad_v0 policy."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TERRAIN = "flat"


def supported_env_kwargs(env_cls, kwargs: dict):
    try:
        parameters = inspect.signature(env_cls).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in parameters and value is not None}


def terrain_label(env, requested: str) -> str:
    for attr_name in ("terrain", "terrain_name", "terrain_level"):
        value = getattr(env, attr_name, None)
        if value:
            return str(value)
    return requested


def load_policy(checkpoint: str | None):
    if checkpoint is None:
        return None
    from stable_baselines3 import PPO

    return PPO.load(checkpoint)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render policy rollout snapshots.")
    parser.add_argument("--task", default="walk", choices=["stand", "walk", "target"])
    parser.add_argument("--policy", default="ppo", choices=["random", "reference", "ppo"])
    parser.add_argument("--checkpoint", help="Stable-Baselines3 PPO checkpoint for --policy ppo.")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--output", default="sim/runs/walk_rollout_contact_sheet.png")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--target-velocity", type=float, help="Target-directed velocity for target task.")
    parser.add_argument("--episode-seconds", type=float, help="Override episode duration for supported envs.")
    parser.add_argument("--target-radius-min", type=float, help="Minimum random target radius for target task.")
    parser.add_argument("--target-radius-max", type=float, help="Maximum random target radius for target task.")
    parser.add_argument("--success-radius", type=float, help="Success radius for target task.")
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN, help="Terrain preset to request from supported envs.")
    parser.add_argument("--terrain-seed", type=int, help="Optional terrain generation seed for supported envs.")
    parser.add_argument(
        "--terrain-curriculum",
        nargs="?",
        const="flat,mild,rough,hard",
        help="Comma-separated terrain curriculum for supported envs.",
    )
    args = parser.parse_args()
    terrain_curriculum = None
    if args.terrain_curriculum:
        terrain_curriculum = tuple(
            value.strip() for value in args.terrain_curriculum.split(",") if value.strip()
        )

    try:
        import numpy as np
        from PIL import Image, ImageDraw
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.policy == "ppo" and not args.checkpoint:
        print("--policy ppo requires --checkpoint PATH.", file=sys.stderr)
        return 1

    if args.task == "target":
        env_cls = SimpleQuadTargetEnv
    elif args.task == "walk":
        env_cls = SimpleQuadWalkEnv
    else:
        env_cls = SimpleQuadStandEnv
    env = env_cls(
        **supported_env_kwargs(
            env_cls,
            {
                "seed": args.seed,
                "render_mode": "rgb_array",
                "randomize_actuators": False,
                "terrain": args.terrain,
                "terrain_seed": args.terrain_seed,
                "terrain_curriculum": terrain_curriculum,
                "target_velocity": args.target_velocity,
                "episode_seconds": args.episode_seconds,
                "target_radius_min": args.target_radius_min,
                "target_radius_max": args.target_radius_max,
                "success_radius": args.success_radius,
            },
        )
    )
    policy = load_policy(args.checkpoint) if args.policy == "ppo" else None
    terrain_text = terrain_label(env, args.terrain)

    obs, reset_info = env.reset(seed=args.seed)
    initial_target_distance = float(reset_info.get("target_distance", float("nan")))
    capture_steps = set(np.linspace(0, args.steps - 1, args.frames, dtype=int).tolist())
    images: list[Image.Image] = []
    total_reward = 0.0
    info = {}
    terminated = truncated = False

    def capture_frame(step_index: int) -> None:
        frame = env.render()
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 420, 78), fill=(255, 255, 255))
        draw.text(
            (16, 16),
            (
                f"task={args.task} policy={args.policy} terrain={terrain_text}\n"
                f"step={step_index} dist={info.get('target_distance', info.get('forward_distance', 0.0)):.3f}m "
                f"z={info.get('base_height', 0.0):.3f}m "
                f"upright={info.get('upright', 0.0):.3f} success={info.get('success', False)}"
            ),
            fill=(0, 0, 0),
        )
        images.append(image)

    try:
        for step in range(args.steps):
            if args.policy == "reference" and hasattr(env, "reference_action"):
                action = env.reference_action()
            elif args.policy == "random":
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if step in capture_steps:
                capture_frame(step)
            if terminated or truncated:
                if step not in capture_steps:
                    capture_frame(step)
                break
    finally:
        env.close()

    if not images:
        print("No frames captured.", file=sys.stderr)
        return 1

    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    width, height = images[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), color=(245, 245, 245))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        sheet.paste(image, (x, y))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"rendered={output}")
    final_target_distance = float(info.get("target_distance", float("nan")))
    target_distance_reduction = initial_target_distance - final_target_distance
    status = "terminated" if terminated else "truncated" if truncated else "max_steps"
    print(
        f"steps={step + 1} return={total_reward:.3f} status={status} "
        f"forward_distance={info.get('forward_distance', float('nan')):.3f} "
        f"initial_target_distance={initial_target_distance:.3f} "
        f"target_distance={final_target_distance:.3f} "
        f"target_distance_reduction={target_distance_reduction:.3f} "
        f"base_height={info.get('base_height', float('nan')):.3f} "
        f"upright={info.get('upright', float('nan')):.3f} "
        f"success={info.get('success', False)} terrain={terrain_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
