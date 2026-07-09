#!/usr/bin/env python3
"""Train the simple quadruped standing policy with PPO."""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TERRAIN = "flat"
DEFAULT_TERRAIN_CURRICULUM = ("flat", "mild", "rough")


def supported_env_kwargs(env_cls, kwargs: dict):
    try:
        parameters = inspect.signature(env_cls).parameters
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in parameters and value is not None}


def parse_terrain_curriculum(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return DEFAULT_TERRAIN_CURRICULUM
    if normalized.lower() in {"0", "false", "none", "off"}:
        return None
    levels = tuple(level.strip() for level in normalized.replace("/", ",").split(",") if level.strip())
    return levels or DEFAULT_TERRAIN_CURRICULUM


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            if xpu.is_available():
                return "xpu"
        except Exception:
            pass
    return "cpu"


def run_random_smoke(
    episodes: int,
    max_steps: int,
    seed: int,
    task: str,
    target_velocity: float | None,
    episode_seconds: float | None,
    target_radius_min: float | None,
    target_radius_max: float | None,
    success_radius: float | None,
    terrain: str,
    terrain_seed: int | None,
    terrain_curriculum: tuple[str, ...] | None,
) -> int:
    try:
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        env = make_env(
            task,
            seed,
            target_velocity,
            episode_seconds,
            target_radius_min,
            target_radius_max,
            success_radius,
            terrain,
            terrain_seed,
            terrain_curriculum,
        )
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    returns: list[float] = []
    try:
        for episode in range(episodes):
            _, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False
            info = {}
            while not (terminated or truncated) and steps < max_steps:
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
            returns.append(total_reward)
            print(
                f"episode={episode} return={total_reward:.3f} steps={steps} "
                f"base_height={info.get('base_height', float('nan')):.3f} "
                f"upright={info.get('upright', float('nan')):.3f} "
                f"forward_distance={info.get('forward_distance', float('nan')):.3f}"
            )
    finally:
        env.close()

    print(f"random_mean_return={sum(returns) / max(1, len(returns)):.3f}")
    return 0


def make_env(
    task: str,
    seed: int,
    target_velocity: float | None = None,
    episode_seconds: float | None = None,
    target_radius_min: float | None = None,
    target_radius_max: float | None = None,
    success_radius: float | None = None,
    terrain: str = DEFAULT_TERRAIN,
    terrain_seed: int | None = None,
    terrain_curriculum: tuple[str, ...] | None = None,
):
    from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv

    common_kwargs = {
        "seed": seed,
        "terrain": terrain,
        "terrain_seed": terrain_seed,
        "terrain_curriculum": terrain_curriculum,
    }
    if task == "target":
        target_kwargs = {
            **common_kwargs,
            "target_velocity": target_velocity,
            "episode_seconds": episode_seconds,
            "target_radius_min": target_radius_min,
            "target_radius_max": target_radius_max,
            "success_radius": success_radius,
        }
        return SimpleQuadTargetEnv(**supported_env_kwargs(SimpleQuadTargetEnv, target_kwargs))
    if task == "walk":
        env_kwargs = {
            **common_kwargs,
            "target_velocity": target_velocity or 0.12,
            "episode_seconds": episode_seconds,
        }
        return SimpleQuadWalkEnv(**supported_env_kwargs(SimpleQuadWalkEnv, env_kwargs))
    stand_kwargs = {**common_kwargs, "episode_seconds": episode_seconds}
    return SimpleQuadStandEnv(**supported_env_kwargs(SimpleQuadStandEnv, stand_kwargs))


def terrain_seed_for_rank(terrain_seed: int | None, rank: int) -> int | None:
    if terrain_seed is None:
        return None
    return terrain_seed + rank


def make_env_init(args: argparse.Namespace, rank: int, monitor_filename: str | None = None):
    def _init():
        env = make_env(
            args.task,
            args.seed + rank,
            args.target_velocity,
            args.episode_seconds,
            args.target_radius_min,
            args.target_radius_max,
            args.success_radius,
            args.terrain,
            terrain_seed_for_rank(args.terrain_seed, rank),
            args.terrain_curriculum,
        )
        if monitor_filename is None:
            return env

        from stable_baselines3.common.monitor import Monitor

        return Monitor(env, filename=monitor_filename)

    return _init


def make_training_env(args: argparse.Namespace, output_dir: Path):
    if args.num_envs == 1:
        return make_env_init(args, 0, str(output_dir / "monitor.csv"))()

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    env_fns = [make_env_init(args, rank) for rank in range(args.num_envs)]
    if args.vec_env == "dummy":
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)
    return VecMonitor(env, filename=str(output_dir / "monitor.csv"))


class WallTimeLimitCallback:
    """Stable-Baselines callback that stops learning after a wall-clock limit."""

    def __init__(self, max_wall_seconds: float) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, limit_seconds: float) -> None:
                super().__init__()
                self.limit_seconds = float(limit_seconds)
                self.started_at = 0.0

            def _on_training_start(self) -> None:
                self.started_at = time.monotonic()

            def _on_step(self) -> bool:
                if self.limit_seconds <= 0.0:
                    return True
                elapsed = time.monotonic() - self.started_at
                if elapsed >= self.limit_seconds:
                    print(f"wall_time_limit_reached={elapsed:.1f}s")
                    return False
                return True

        self.callback = _Callback(max_wall_seconds)


def train_ppo(args: argparse.Namespace) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        print(f"Stable-Baselines3 import failed: {exc}", file=sys.stderr)
        print("Install sim dependencies, then rerun training.", file=sys.stderr)
        return 1

    try:
        from sim.envs.simple_quad_stand import SimpleQuadStandEnv, SimpleQuadTargetEnv, SimpleQuadWalkEnv
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    device = select_device(args.device)
    if args.num_envs > 1 and args.device == "auto" and device != "cpu":
        print(f"auto_device_override=cpu requested_auto_selected={device} reason=parallel_env_rollout")
        device = "cpu"
    if device == "xpu":
        try:
            import torch

            torch.distributions.Distribution.set_default_validate_args(False)
        except Exception as exc:
            print(f"warning: unable to disable torch distribution validation on XPU: {exc}")
    run_name = f"simple_quad_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = make_training_env(args, output_dir)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.init_checkpoint:
        model = PPO.load(
            args.init_checkpoint,
            env=env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.98,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
        )
    if args.pretrain_reference_steps > 0:
        pretrain_env = env
        if args.num_envs > 1:
            pretrain_env = make_env_init(args, 0, str(output_dir / "pretrain_monitor.csv"))()
            print("pretrain_envs=1")
        pretrain_reference_policy(
            model,
            pretrain_env,
            steps=args.pretrain_reference_steps,
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            device=device,
            dagger_rounds=args.dagger_rounds,
            dagger_steps=args.dagger_steps,
        )
        if pretrain_env is not env:
            pretrain_env.close()
    print(f"training_device={device}")
    print(f"num_envs={args.num_envs}")
    if args.num_envs > 1:
        print(f"vec_env={args.vec_env}")
        print(f"rollout_transitions_per_update={args.n_steps * args.num_envs}")
    print(f"output_dir={output_dir}")
    if args.init_checkpoint:
        print(f"init_checkpoint={Path(args.init_checkpoint).resolve()}")
    print(f"terrain={args.terrain}")
    if args.terrain_seed is not None:
        print(f"terrain_seed={args.terrain_seed}")
    if args.terrain_curriculum is not None:
        print(f"terrain_curriculum={','.join(args.terrain_curriculum)}")
    if not args.skip_ppo and args.total_timesteps > 0:
        callbacks = []
        if args.checkpoint_freq > 0:
            from stable_baselines3.common.callbacks import CheckpointCallback

            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=max(1, args.checkpoint_freq // args.num_envs),
                    save_path=str(checkpoint_dir),
                    name_prefix="policy",
                )
            )
            print(f"checkpoint_freq={args.checkpoint_freq}")
            print(f"checkpoint_dir={checkpoint_dir}")
        if args.max_wall_seconds > 0:
            callbacks.append(WallTimeLimitCallback(args.max_wall_seconds).callback)
            print(f"max_wall_seconds={args.max_wall_seconds}")
        callback = None
        if len(callbacks) == 1:
            callback = callbacks[0]
        elif callbacks:
            from stable_baselines3.common.callbacks import CallbackList

            callback = CallbackList(callbacks)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=args.progress_bar, callback=callback)

    checkpoint = output_dir / "policy.zip"
    model.save(checkpoint)
    env.close()
    print(f"saved_checkpoint={checkpoint}")
    return 0


def pretrain_reference_policy(
    model,
    monitored_env,
    steps: int,
    epochs: int,
    batch_size: int,
    device: str,
    dagger_rounds: int = 0,
    dagger_steps: int = 0,
) -> None:
    """Warm-start a PPO MLP policy from env.reference_action() when available."""

    try:
        import numpy as np
        import torch
    except ImportError:
        return

    env = getattr(monitored_env, "env", monitored_env)
    if not hasattr(env, "reference_action"):
        return

    def reset_reference_episode(episode_index: int):
        return env.reset(seed=1 + episode_index)

    episode = 0
    episode_steps = 0
    torch_device = torch.device(device)

    def collect_dataset(collect_steps: int, rollout_policy: bool, seed_offset: int):
        episode_index = seed_offset
        episode_steps = 0
        obs, _ = reset_reference_episode(episode_index)
        observations = []
        actions = []
        for _ in range(max(1, collect_steps)):
            reference = env.reference_action()
            observations.append(obs.copy())
            actions.append(reference.copy())
            if rollout_policy:
                rollout_action, _ = model.predict(obs, deterministic=True)
            else:
                rollout_action = reference
            obs, _, terminated, truncated, _ = env.step(rollout_action)
            episode_steps += 1
            if terminated or truncated or episode_steps >= 350:
                episode_index += 1
                episode_steps = 0
                obs, _ = reset_reference_episode(episode_index)
        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
        )

    def train_supervised(obs_array, action_array, train_epochs: int, label: str) -> None:
        obs_tensor = torch.as_tensor(obs_array, device=torch_device)
        action_tensor = torch.as_tensor(action_array, device=torch_device)

        model.policy.train()
        count = obs_tensor.shape[0]
        for epoch in range(max(1, train_epochs)):
            permutation = torch.randperm(count, device=torch_device)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, count, batch_size):
                indices = permutation[start : start + batch_size]
                batch_obs = obs_tensor[indices]
                batch_actions = action_tensor[indices]
                distribution = model.policy.get_distribution(batch_obs)
                predicted = distribution.distribution.mean
                loss = torch.mean((predicted - batch_actions) ** 2)
                model.policy.optimizer.zero_grad()
                loss.backward()
                model.policy.optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            print(f"{label}_epoch={epoch} mse={epoch_loss / max(1, batches):.6f} samples={count}")

    obs_array, action_array = collect_dataset(steps, rollout_policy=False, seed_offset=1)
    train_supervised(obs_array, action_array, epochs, "pretrain")

    for dagger_round in range(max(0, dagger_rounds)):
        new_obs, new_actions = collect_dataset(
            dagger_steps or max(1, steps // 2),
            rollout_policy=True,
            seed_offset=10000 + dagger_round * 1000,
        )
        obs_array = np.concatenate([obs_array, new_obs], axis=0)
        action_array = np.concatenate([action_array, new_actions], axis=0)
        train_supervised(
            obs_array,
            action_array,
            max(1, epochs // 3),
            f"dagger_round={dagger_round}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train simple_quad_v0 standing policy.")
    parser.add_argument("--task", default="stand", choices=["stand", "walk", "target"], help="Training task.")
    parser.add_argument("--robot", default="simple_quad_v0", choices=["simple_quad_v0"], help="Robot model.")
    parser.add_argument("--total-timesteps", type=int, default=2048, help="PPO training timesteps.")
    parser.add_argument("--timesteps", type=int, dest="total_timesteps", help="Alias for --total-timesteps.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "xpu", "cpu"], help="Torch device for PPO.")
    parser.add_argument("--output-dir", default="sim/runs", help="Training run output directory.")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to collect PPO rollouts from.",
    )
    parser.add_argument(
        "--vec-env",
        default="subproc",
        choices=["subproc", "dummy"],
        help="Vector environment backend used when --num-envs is greater than 1.",
    )
    parser.add_argument("--n-steps", type=int, default=128, help="PPO rollout steps.")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO batch size.")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.002, help="PPO entropy coefficient.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=0,
        help="Save a PPO checkpoint every N total environment transitions. Disabled by default.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="Stop PPO after this many wall-clock seconds, then save the final policy. Disabled by default.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Stable-Baselines3 verbosity.")
    parser.add_argument("--target-velocity", type=float, help="Walking or target-directed velocity in m/s.")
    parser.add_argument("--episode-seconds", type=float, help="Override episode duration for supported envs.")
    parser.add_argument("--target-radius-min", type=float, help="Minimum random target radius for target task.")
    parser.add_argument("--target-radius-max", type=float, help="Maximum random target radius for target task.")
    parser.add_argument("--success-radius", type=float, help="Success radius for target task.")
    parser.add_argument("--init-checkpoint", help="Optional PPO checkpoint to fine-tune instead of starting fresh.")
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN, help="Terrain preset to request from supported envs.")
    parser.add_argument("--terrain-seed", type=int, help="Optional terrain generation seed for supported envs.")
    parser.add_argument(
        "--terrain-curriculum",
        nargs="?",
        const=",".join(DEFAULT_TERRAIN_CURRICULUM),
        help="Enable terrain curriculum for supported envs. Defaults to flat,mild,rough when no value is supplied.",
    )
    parser.add_argument(
        "--pretrain-reference-steps",
        type=int,
        default=0,
        help="Warm-start walking policy from the built-in reference trot before PPO.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument(
        "--dagger-rounds",
        type=int,
        default=0,
        help="After reference pretraining, collect this many policy-rollout datasets labeled by the reference action.",
    )
    parser.add_argument(
        "--dagger-steps",
        type=int,
        default=0,
        help="Policy-rollout samples per DAgger round. Defaults to half of --pretrain-reference-steps.",
    )
    parser.add_argument("--skip-ppo", action="store_true", help="Save after reference pretraining without PPO updates.")
    parser.add_argument("--progress-bar", action="store_true", help="Enable SB3 progress bar output.")
    parser.add_argument(
        "--random-smoke",
        action="store_true",
        help="Run random policy episodes through the env instead of PPO.",
    )
    parser.add_argument("--random-episodes", type=int, default=1)
    parser.add_argument("--random-max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    args.terrain_curriculum = parse_terrain_curriculum(args.terrain_curriculum)

    if args.random_smoke:
        return run_random_smoke(
            args.random_episodes,
            args.random_max_steps,
            args.seed,
            args.task,
            args.target_velocity,
            args.episode_seconds,
            args.target_radius_min,
            args.target_radius_max,
            args.success_radius,
            args.terrain,
            args.terrain_seed,
            args.terrain_curriculum,
        )
    return train_ppo(args)


if __name__ == "__main__":
    raise SystemExit(main())
