#!/usr/bin/env python3
"""Evaluate a random or trained simple_quad_v0 standing policy."""

from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_policy(checkpoint: str | None):
    if checkpoint is None:
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(f"Stable-Baselines3 import failed: {exc}") from exc
    return PPO.load(checkpoint)


def write_ppm(path: Path, rgb) -> None:
    height, width = rgb.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(rgb.astype("uint8").tobytes())


TARGET_SECTORS = (
    "front",
    "front_left",
    "left",
    "back_left",
    "back",
    "back_right",
    "right",
    "front_right",
)


def supported_kwargs(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def make_env(env_cls, **kwargs):
    return env_cls(**supported_kwargs(env_cls, kwargs))


def mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def target_sector(target_x: float, target_y: float, base_x: float = 0.0, base_y: float = 0.0) -> str:
    dx = target_x - base_x
    dy = target_y - base_y
    if not (math.isfinite(dx) and math.isfinite(dy)) or (abs(dx) < 1e-9 and abs(dy) < 1e-9):
        return "unknown"
    angle = math.atan2(dy, dx)
    sector_index = int(((angle + math.pi / 8.0) % (2.0 * math.pi)) // (math.pi / 4.0))
    return TARGET_SECTORS[sector_index]


def format_sector_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    ordered = [sector for sector in TARGET_SECTORS if counts.get(sector, 0) > 0]
    ordered.extend(sector for sector in sorted(counts) if sector not in TARGET_SECTORS)
    return ",".join(f"{sector}:{counts[sector]}" for sector in ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate simple_quad_v0 in MuJoCo.")
    parser.add_argument("--robot", default="simple_quad_v0", choices=["simple_quad_v0"])
    parser.add_argument("--task", default="stand", choices=["stand", "walk", "target"])
    parser.add_argument("--policy", choices=["random", "reference", "ppo"], default="random")
    parser.add_argument("--checkpoint", help="Optional Stable-Baselines3 PPO .zip checkpoint.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--record", help="Optional .npz path for qpos/qvel/action/reward trajectory.")
    parser.add_argument("--render-rgb", help="Optional final-frame PPM path.")
    parser.add_argument("--target-velocity", type=float, help="Walking or target-directed velocity in m/s.")
    parser.add_argument("--episode-seconds", type=float, help="Override episode duration for supported envs.")
    parser.add_argument("--target-radius-min", type=float, help="Minimum random target radius for target task.")
    parser.add_argument("--target-radius-max", type=float, help="Maximum random target radius for target task.")
    parser.add_argument("--success-radius", type=float, help="Success radius for target task.")
    parser.add_argument("--terrain", default="flat", help="Terrain preset to request from envs that support terrain.")
    parser.add_argument("--terrain-seed", type=int, help="Terrain seed. Defaults to --seed.")
    parser.add_argument(
        "--terrain-curriculum",
        action="store_true",
        help="Request terrain curriculum mode from envs that support it.",
    )
    args = parser.parse_args()
    if args.terrain_seed is None:
        args.terrain_seed = args.seed

    use_ppo = args.policy == "ppo" or args.checkpoint is not None
    if use_ppo and args.checkpoint is None:
        print("--policy ppo requires --checkpoint PATH.", file=sys.stderr)
        return 1

    try:
        import numpy as np
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        policy = load_policy(args.checkpoint) if use_ppo else None
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    render_mode = "rgb_array" if args.render_rgb else None
    env_kwargs = {
        "seed": args.seed,
        "render_mode": render_mode,
        "randomize_actuators": False,
        "terrain": args.terrain,
        "terrain_seed": args.terrain_seed,
        "terrain_curriculum": args.terrain_curriculum,
        "deterministic": True,
        "episode_seconds": args.episode_seconds,
    }
    try:
        if args.task == "target":
            env = make_env(
                SimpleQuadTargetEnv,
                **env_kwargs,
                target_velocity=args.target_velocity,
                target_radius_min=args.target_radius_min,
                target_radius_max=args.target_radius_max,
                success_radius=args.success_radius,
            )
        elif args.task == "walk":
            env = make_env(SimpleQuadWalkEnv, **env_kwargs, target_velocity=args.target_velocity or 0.12)
        else:
            env = make_env(SimpleQuadStandEnv, **env_kwargs)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    records: dict[str, list] = {"qpos": [], "qvel": [], "action": [], "reward": []}
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    time_to_successes: list[float] = []
    target_distance_reductions: list[float] = []
    failure_sectors: dict[str, int] = {}
    last_frame = None

    try:
        for episode in range(args.episodes):
            obs, reset_info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            info = {}
            success_step: int | None = None
            initial_target_distance = float(reset_info.get("target_distance", float("nan")))
            initial_target_x = float(reset_info.get("target_x", float("nan")))
            initial_target_y = float(reset_info.get("target_y", float("nan")))
            base_x = base_y = 0.0
            if hasattr(env, "data"):
                base_x = float(env.data.qpos[0])
                base_y = float(env.data.qpos[1])
            initial_target_sector = target_sector(initial_target_x, initial_target_y, base_x, base_y)
            while not (terminated or truncated) and steps < args.max_steps:
                if args.policy == "reference" and hasattr(env, "reference_action"):
                    action = env.reference_action()
                elif policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if args.task == "target" and success_step is None and bool(info.get("success", False)):
                    success_step = steps
                if args.record:
                    records["qpos"].append(env.data.qpos.copy())
                    records["qvel"].append(env.data.qvel.copy())
                    records["action"].append(np.asarray(action, dtype=np.float32).copy())
                    records["reward"].append(float(reward))
            if render_mode:
                last_frame = env.render()
            returns.append(total_reward)
            lengths.append(steps)
            success = bool(info.get("success", False))
            successes.append(success)
            if args.task == "target":
                final_target_distance = float(info.get("target_distance", initial_target_distance))
                if math.isfinite(initial_target_distance) and math.isfinite(final_target_distance):
                    target_distance_reductions.append(initial_target_distance - final_target_distance)
                if success and success_step is not None:
                    time_to_successes.append(float(success_step) * float(getattr(env, "control_dt", float("nan"))))
                elif not success:
                    failure_sectors[initial_target_sector] = failure_sectors.get(initial_target_sector, 0) + 1
            status = "terminated" if terminated else "truncated"
            print(
                f"episode={episode} return={total_reward:.3f} steps={steps} status={status} "
                f"base_height={info.get('base_height', float('nan')):.3f} "
                f"upright={info.get('upright', float('nan')):.3f} "
                f"forward_distance={info.get('forward_distance', float('nan')):.3f} "
                f"forward_velocity={info.get('forward_velocity', float('nan')):.3f} "
                f"target_distance={info.get('target_distance', float('nan')):.3f} "
                f"success={info.get('success', False)}"
            )

        if args.record:
            record_path = Path(args.record).resolve()
            record_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                record_path,
                qpos=np.asarray(records["qpos"]),
                qvel=np.asarray(records["qvel"]),
                action=np.asarray(records["action"]),
                reward=np.asarray(records["reward"]),
                model_path=str(env.model_path),
            )
            print(f"recorded={record_path}")

        if args.render_rgb and last_frame is not None:
            render_path = Path(args.render_rgb).resolve()
            write_ppm(render_path, last_frame)
            print(f"rendered={render_path}")
    finally:
        env.close()

    print(f"mean_return={float(np.mean(returns)):.3f}")
    print(f"mean_length={float(np.mean(lengths)):.1f}")
    if args.task == "target":
        print(f"success_rate={float(np.mean(successes)):.3f} successes={sum(successes)}/{len(successes)}")
        print(f"mean_time_to_success={mean_or_nan(time_to_successes):.3f}")
        print(f"target_distance_reduction={mean_or_nan(target_distance_reductions):.3f}")
        print(f"failure_sectors={format_sector_counts(failure_sectors)}")
    print(
        f"terrain={args.terrain} terrain_seed={args.terrain_seed} "
        f"terrain_curriculum={args.terrain_curriculum}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
