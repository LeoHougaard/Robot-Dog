#!/usr/bin/env python3
"""Open a MuJoCo viewer for a trained or scripted simple_quad_v0 policy."""

from __future__ import annotations

import argparse
import inspect
import sys
import time
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


def latest_checkpoint(task: str) -> Path:
    if task == "target":
        roots = [
            REPO_ROOT / "sim/runs/target_far_overnight",
            REPO_ROOT / "sim/runs/target_far_run_ppo",
            REPO_ROOT / "sim/runs/target_far_dagger",
            REPO_ROOT / "sim/runs/target_random_dagger_slew",
            REPO_ROOT / "sim/runs/target_random_dagger",
            REPO_ROOT / "sim/runs/target_random_pretrained",
            REPO_ROOT / "sim/runs/target_pretrained",
            REPO_ROOT / "sim/runs/target_ppo",
        ]
    elif task == "walk":
        roots = [
            REPO_ROOT / "sim/runs/walk_pretrained",
            REPO_ROOT / "sim/runs",
        ]
    else:
        roots = [REPO_ROOT / "sim/runs"]

    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.rglob("policy.zip") if path.is_file())
        candidates.extend(path for path in root.rglob("policy_*_steps.zip") if path.is_file())
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No trained {task} checkpoint found under {', '.join(str(root) for root in roots)}."
        )
    return candidates[0]


def resolve_checkpoint(value: str | None, task: str) -> Path | None:
    if value is None:
        return None
    if value == "latest":
        return latest_checkpoint(task)
    return Path(value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the trained robot dog policy in MuJoCo.")
    parser.add_argument("--task", default="walk", choices=["stand", "walk", "target"])
    parser.add_argument("--policy", default="ppo", choices=["ppo", "reference", "random"])
    parser.add_argument("--checkpoint", default="latest", help="Policy checkpoint path, or 'latest'.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN, help="Terrain preset to request from supported envs.")
    parser.add_argument("--terrain-seed", type=int, help="Optional terrain generation seed for supported envs.")
    parser.add_argument("--terrain-curriculum", help="Comma-separated terrain curriculum.")
    parser.add_argument("--target-velocity", type=float, help="Target-directed velocity for target task.")
    parser.add_argument("--episode-seconds", type=float, help="Override episode duration for supported envs.")
    parser.add_argument("--target-radius-min", type=float, help="Minimum random target radius for target task.")
    parser.add_argument("--target-radius-max", type=float, help="Maximum random target radius for target task.")
    parser.add_argument("--success-radius", type=float, help="Success radius for target task.")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means until viewer is closed.")
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument(
        "--auto-reset-success",
        action="store_true",
        help="For target mode, sample a new random target immediately after success.",
    )
    args = parser.parse_args()
    terrain_curriculum = (
        tuple(value.strip() for value in args.terrain_curriculum.split(",") if value.strip())
        if args.terrain_curriculum
        else None
    )

    try:
        import mujoco.viewer
        from stable_baselines3 import PPO
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
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
    terrain_text = terrain_label(env, args.terrain)
    policy = None
    checkpoint = resolve_checkpoint(args.checkpoint, args.task) if args.checkpoint else None
    if checkpoint is not None:
        policy = PPO.load(checkpoint, device="cpu")

    state = {
        "paused": args.start_paused,
        "reset": False,
        "quit": False,
        "policy": args.policy,
        "target_step": 0.08,
        "reset_count": 0,
    }

    def reset_env():
        seed = args.seed + state["reset_count"]
        state["reset_count"] += 1
        return env.reset(seed=seed)

    def key_callback(keycode: int) -> None:
        # GLFW key codes: space=32, escape=256, R=82, P=80, 1=49, 2=50, 3=51.
        if keycode in (32, 80):
            state["paused"] = not state["paused"]
        elif keycode == 82:
            state["reset"] = True
        elif keycode in (256, 81):
            state["quit"] = True
        elif keycode == 49:
            state["policy"] = "ppo"
        elif keycode == 50 and hasattr(env, "reference_action"):
            state["policy"] = "reference"
        elif keycode == 51:
            state["policy"] = "random"
        elif args.task == "target" and hasattr(env, "set_target"):
            x, y = [float(v) for v in env.target_xy]
            step = state["target_step"]
            # W/up: +x, S/down: -x, A/left: +y, D/right: -y.
            if keycode in (87, 265):
                env.set_target(x + step, y)
            elif keycode in (83, 264):
                env.set_target(x - step, y)
            elif keycode in (65, 263):
                env.set_target(x, y + step)
            elif keycode in (68, 262):
                env.set_target(x, y - step)

    obs, _ = reset_env()
    start = time.perf_counter()
    next_step = start
    step_dt = env.control_dt

    print("MuJoCo viewer controls:")
    print(f"task={args.task} policy={state['policy']} terrain={terrain_text}")
    print("  Space/P: pause or resume")
    print("  R: reset")
    print("  1: PPO policy")
    if hasattr(env, "reference_action"):
        print("  2: reference controller")
    print("  3: random actions")
    if args.task == "target":
        print("  W/S or Up/Down: move target along world X")
        print("  A/D or Left/Right: move target along world Y")
    print("  Q/Esc or close the viewer: quit")
    if checkpoint is not None:
        print(f"checkpoint={checkpoint}")

    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
            while viewer.is_running() and not state["quit"]:
                now = time.perf_counter()
                if args.duration > 0 and now - start >= args.duration:
                    break
                if state["reset"]:
                    obs, _ = reset_env()
                    state["reset"] = False
                    next_step = now
                if not state["paused"] and now >= next_step:
                    if state["policy"] == "ppo" and policy is not None:
                        action, _ = policy.predict(obs, deterministic=True)
                    elif state["policy"] == "reference" and hasattr(env, "reference_action"):
                        action = env.reference_action()
                    else:
                        action = env.action_space.sample()
                    obs, _, terminated, truncated, info = env.step(action)
                    target_success = bool(info.get("success", False))
                    if (terminated and not target_success) or truncated or (args.auto_reset_success and target_success):
                        obs, _ = reset_env()
                    next_step += step_dt

                viewer.sync()
                time.sleep(0.002)
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
