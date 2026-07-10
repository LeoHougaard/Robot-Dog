#!/usr/bin/env python3
"""Tkinter viewer for target-driving when the native MuJoCo viewer is blank."""

from __future__ import annotations

import argparse
import inspect
import random
import sys
import time
import tkinter as tk
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


def short_path(path: Path | None) -> str:
    if path is None:
        return "none"
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an offscreen-rendered robot dog policy viewer.")
    parser.add_argument("--task", default="target", choices=["stand", "walk", "target"])
    parser.add_argument("--policy", default="ppo", choices=["ppo", "reference", "random"])
    parser.add_argument("--checkpoint", default="latest", help="Policy checkpoint path, 'latest', or empty.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN, help="Terrain preset to request from supported envs.")
    parser.add_argument("--terrain-seed", type=int, help="Optional terrain generation seed for supported envs.")
    parser.add_argument("--terrain-curriculum", help="Comma-separated terrain curriculum.")
    parser.add_argument(
        "--random-reset",
        action="store_true",
        help="Use a fresh random seed, start pose, target, and curriculum surface on every reset.",
    )
    parser.add_argument("--target-velocity", type=float, help="Target-directed velocity for target task.")
    parser.add_argument("--episode-seconds", type=float, help="Override episode duration for supported envs.")
    parser.add_argument("--target-radius-min", type=float, help="Minimum random target radius for target task.")
    parser.add_argument("--target-radius-max", type=float, help="Maximum random target radius for target task.")
    parser.add_argument("--success-radius", type=float, help="Success radius for target task.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
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
        from PIL import Image, ImageTk
        from stable_baselines3 import PPO
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv
        from sim.view_policy import resolve_checkpoint
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
    terrain_text = terrain_label(env, args.terrain)
    checkpoint: Path | None = None
    policy = None

    state = {
        "policy": args.policy,
        "paused": False,
        "target_step": 0.08,
        "obs": None,
        "last_info": {},
        "last_step_time": time.perf_counter(),
        "last_tick_time": time.perf_counter(),
        "reset_count": 0,
        "checkpoint": None,
        "checkpoint_mtime": None,
        "checkpoint_status": "not loaded",
    }

    root = tk.Tk()
    root.title(f"Robot Dog {args.task} viewer - terrain={terrain_text}")
    root.geometry(f"{args.width}x{args.height}")

    toolbar = tk.Frame(root)
    toolbar.pack(fill=tk.X)
    refresh_button = tk.Button(toolbar, text="Refresh policy")
    refresh_button.pack(side=tk.LEFT)
    reset_button = tk.Button(toolbar, text="Reset")
    reset_button.pack(side=tk.LEFT, padx=(6, 0))
    run_indicator = tk.Canvas(toolbar, width=18, height=18, highlightthickness=0)
    run_indicator.pack(side=tk.LEFT, padx=(10, 4))
    run_indicator_dot = run_indicator.create_oval(3, 3, 15, 15, fill="#2fa84f", outline="#1d6f34", width=1)
    run_indicator_text = tk.StringVar(value="Running")
    run_indicator_label = tk.Label(toolbar, textvariable=run_indicator_text, anchor="w")
    run_indicator_label.pack(side=tk.LEFT)

    image_label = tk.Label(root, bg="black")
    image_label.pack(fill=tk.BOTH, expand=True)
    status = tk.StringVar(value="")
    status_label = tk.Label(root, textvariable=status, anchor="w")
    status_label.pack(fill=tk.X)

    reset_rng = random.SystemRandom()

    def reset_env() -> None:
        if args.random_reset:
            seed = reset_rng.randrange(0, 2**31)
            if terrain_curriculum and hasattr(env, "terrain_episode_index"):
                terrain_index = reset_rng.randrange(len(terrain_curriculum))
                env.terrain_episode_index = terrain_index * 20 + reset_rng.randrange(20)
        else:
            seed = args.seed + state["reset_count"]
        state["reset_count"] += 1
        state["obs"], reset_info = env.reset(seed=seed)
        state["last_info"] = reset_info

    reset_env()
    reset_button.configure(command=reset_env)

    def load_policy(reset_after_load: bool = False) -> None:
        nonlocal checkpoint, policy
        if not args.checkpoint:
            checkpoint = None
            policy = None
            state["checkpoint"] = None
            state["checkpoint_mtime"] = None
            state["checkpoint_status"] = "no checkpoint requested"
            return
        try:
            resolved = resolve_checkpoint(args.checkpoint, args.task)
            loaded_policy = PPO.load(resolved, device="cpu") if resolved is not None else None
        except Exception as exc:
            state["checkpoint_status"] = f"refresh failed: {exc}"
            return
        checkpoint = resolved
        policy = loaded_policy
        state["checkpoint"] = checkpoint
        state["checkpoint_mtime"] = checkpoint.stat().st_mtime if checkpoint is not None else None
        state["checkpoint_status"] = f"loaded {short_path(checkpoint)}"
        if reset_after_load:
            reset_env()

    def refresh_policy() -> None:
        load_policy(reset_after_load=True)

    refresh_button.configure(command=refresh_policy)
    load_policy(reset_after_load=False)

    def set_target_delta(dx: float, dy: float) -> None:
        if args.task == "target" and hasattr(env, "set_target"):
            x, y = [float(v) for v in env.target_xy]
            env.set_target(x + dx, y + dy)

    def on_key(event) -> None:
        key = event.keysym.lower()
        step = float(state["target_step"])
        if key in ("space", "p"):
            state["paused"] = not state["paused"]
        elif key == "r":
            reset_env()
        elif key == "f5":
            refresh_policy()
        elif key in ("q", "escape"):
            root.destroy()
        elif key == "1":
            state["policy"] = "ppo"
        elif key == "2" and hasattr(env, "reference_action"):
            state["policy"] = "reference"
        elif key == "3":
            state["policy"] = "random"
        elif key in ("w", "up"):
            set_target_delta(step, 0.0)
        elif key in ("s", "down"):
            set_target_delta(-step, 0.0)
        elif key in ("a", "left"):
            set_target_delta(0.0, step)
        elif key in ("d", "right"):
            set_target_delta(0.0, -step)

    def select_action():
        if state["policy"] == "ppo" and policy is not None:
            action, _ = policy.predict(state["obs"], deterministic=True)
            return action
        if state["policy"] == "reference" and hasattr(env, "reference_action"):
            return env.reference_action()
        return env.action_space.sample()

    def update_status() -> None:
        info = state["last_info"]
        target_text = ""
        if args.task == "target":
            target_text = (
                f" target=({float(env.target_xy[0]):+.2f},{float(env.target_xy[1]):+.2f})"
                f" distance={info.get('target_distance', float('nan')):.3f}"
                f" success={info.get('success', False)}"
            )
        status.set(
            f"policy={state['policy']} paused={state['paused']}"
            f" checkpoint={short_path(state['checkpoint'])}"
            f" terrain={terrain_label(env, args.terrain)}"
            f" start={info.get('reset_pose', 'unknown')}"
            f" z={info.get('base_height', float('nan')):.3f}"
            f" upright={info.get('upright', float('nan')):.3f}"
            f"{target_text}"
            " | Reset/R randomizes start+terrain, Refresh/F5 reloads latest, W/S/A/D move target"
        )

    def update_run_indicator() -> None:
        tick_age = time.perf_counter() - float(state["last_tick_time"])
        if state["paused"]:
            fill = "#d89b28"
            outline = "#8a6114"
            label = "Paused"
        elif tick_age > 1.0:
            fill = "#c83e3e"
            outline = "#842020"
            label = "Stale"
        else:
            fill = "#2fa84f"
            outline = "#1d6f34"
            label = "Running"
        run_indicator.itemconfigure(run_indicator_dot, fill=fill, outline=outline)
        run_indicator_text.set(label)

    def tick() -> None:
        try:
            if not state["paused"]:
                action = select_action()
                obs, _reward, terminated, truncated, info = env.step(action)
                state["obs"] = obs
                state["last_info"] = info
                state["last_step_time"] = time.perf_counter()
                target_success = bool(info.get("success", False))
                if (terminated and not target_success) or truncated or (args.auto_reset_success and target_success):
                    reset_env()

            frame = env.render()
            image = Image.fromarray(frame).resize((args.width, max(1, args.height - 64)))
            photo = ImageTk.PhotoImage(image=image)
            image_label.configure(image=photo)
            image_label.image = photo
            state["last_tick_time"] = time.perf_counter()
            update_run_indicator()
            update_status()
            root.after(int(1000 * env.control_dt), tick)
        except tk.TclError:
            pass

    def on_close() -> None:
        env.close()
        root.destroy()

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(0, tick)
    root.focus_force()

    if checkpoint is not None:
        print(f"checkpoint={checkpoint}")
    print(f"terrain={terrain_text}")
    print("Tk viewer controls: Refresh/F5 reload latest policy, W/S/A/D move target, 1 learned, 2 reference, 3 random, Space pause, R reset.")
    root.mainloop()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
