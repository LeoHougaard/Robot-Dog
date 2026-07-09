#!/usr/bin/env python3
"""Supervise a long far-target PPO training session with restarts."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sim/runs/target_far_overnight"
DEFAULT_SOURCE_DIRS = (
    REPO_ROOT / "sim/runs/target_far_overnight",
    REPO_ROOT / "sim/runs/target_far_run_ppo",
    REPO_ROOT / "sim/runs/target_far_dagger",
    REPO_ROOT / "sim/runs/target_random_dagger",
)
DEFAULT_FALLBACK_CHECKPOINT = (
    REPO_ROOT / "sim/runs/target_random_dagger/promoted_fast_flat_20260708_183850/policy.zip"
)
LATEST_MARKER = "LATEST_OVERNIGHT_POLICY.txt"
VIEW_COMMAND = "VIEW_OVERNIGHT_POLICY.ps1"
STATUS_JSON = "overnight_status.json"
SUPERVISOR_LOG = "overnight_supervisor.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str, log_path: Path) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def prevent_sleep() -> None:
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    es_awaymode_required = 0x00000040
    ctypes.windll.kernel32.SetThreadExecutionState(
        es_continuous | es_system_required | es_awaymode_required
    )


def release_sleep_request() -> None:
    if sys.platform != "win32":
        return
    es_continuous = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def checkpoint_candidates(roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(
            path
            for path in root.rglob("*.zip")
            if path.is_file() and path.parent.name != "latest" and not path.name.endswith(".tmp.zip")
        )
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def newest_checkpoint(roots: list[Path], fallback: Path | None) -> Path:
    candidates = checkpoint_candidates(roots)
    if candidates:
        return candidates[0]
    if fallback is not None and fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("No checkpoint found to start overnight training.")


def quote_ps_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def write_view_command(output_dir: Path, checkpoint: Path) -> None:
    command = (
        f"python sim/view_policy_tk.py --task target --policy ppo --checkpoint {quote_ps_path(checkpoint)} "
        "--terrain flat --episode-seconds 30 --target-radius-min 1.1 --target-radius-max 1.6 "
        "--success-radius 0.22 --target-velocity 0.30 --seed 1000"
    )
    (output_dir / LATEST_MARKER).write_text(str(checkpoint) + "\n", encoding="utf-8")
    (output_dir / VIEW_COMMAND).write_text(command + "\n", encoding="utf-8")


def copy_latest_policy(output_dir: Path, checkpoint: Path) -> Path:
    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    latest_policy = latest_dir / "policy.zip"
    temp_policy = latest_dir / "policy.tmp.zip"
    shutil.copy2(checkpoint, temp_policy)
    temp_policy.replace(latest_policy)
    write_view_command(output_dir, latest_policy.resolve())
    return latest_policy.resolve()


def write_status(output_dir: Path, status: dict) -> None:
    status_path = output_dir / STATUS_JSON
    temp_path = output_dir / f"{STATUS_JSON}.tmp"
    temp_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(status_path)


def train_command(args: argparse.Namespace, checkpoint: Path, max_wall_seconds: int) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "sim/train.py"),
        "--task",
        "target",
        "--device",
        "cpu",
        "--init-checkpoint",
        str(checkpoint),
        "--total-timesteps",
        str(args.chunk_timesteps),
        "--num-envs",
        str(args.num_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--ent-coef",
        str(args.ent_coef),
        "--terrain",
        "flat",
        "--episode-seconds",
        "30",
        "--target-radius-min",
        "1.1",
        "--target-radius-max",
        "1.6",
        "--success-radius",
        "0.22",
        "--target-velocity",
        "0.30",
        "--output-dir",
        str(args.output_dir),
        "--verbose",
        "0",
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--max-wall-seconds",
        str(max_wall_seconds),
    ]


def stop_process(process: subprocess.Popen, log_path: Path) -> None:
    if process.poll() is not None:
        return
    log(f"deadline_grace_expired terminating_child pid={process.pid}", log_path)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log(f"deadline_grace_expired killing_child pid={process.pid}", log_path)
        process.kill()
        process.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run supervised overnight far-target PPO training.")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--checkpoint-freq", type=int, default=100000)
    parser.add_argument("--chunk-timesteps", type=int, default=10_000_000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--deadline-grace-seconds", type=float, default=300.0)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / SUPERVISOR_LOG
    source_dirs = [args.output_dir, *DEFAULT_SOURCE_DIRS]
    fallback = args.initial_checkpoint.resolve() if args.initial_checkpoint else DEFAULT_FALLBACK_CHECKPOINT
    deadline = time.monotonic() + max(1.0, args.hours * 3600.0)
    final_status = 0
    child_index = 0

    log(f"overnight_start hours={args.hours} output_dir={args.output_dir}", log_path)
    log("sleep_prevention=enabled_while_supervisor_runs", log_path)
    prevent_sleep()
    try:
        while time.monotonic() < deadline:
            checkpoint = newest_checkpoint(source_dirs, fallback)
            copied_policy = copy_latest_policy(args.output_dir, checkpoint)
            remaining = max(1, int(deadline - time.monotonic()))
            child_index += 1
            child_stdout = args.output_dir / f"train_child_{child_index:03d}.stdout.log"
            child_stderr = args.output_dir / f"train_child_{child_index:03d}.stderr.log"
            command = train_command(args, checkpoint, remaining)
            status = {
                "status": "running",
                "updated_at": utc_now(),
                "deadline_utc": datetime.fromtimestamp(time.time() + remaining, timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "child_index": child_index,
                "child_pid": None,
                "input_checkpoint": str(checkpoint),
                "latest_policy": str(copied_policy),
                "viewer_command_file": str(args.output_dir / VIEW_COMMAND),
                "stdout_log": str(child_stdout),
                "stderr_log": str(child_stderr),
                "command": command,
            }
            write_status(args.output_dir, status)
            log(f"child_start index={child_index} remaining={remaining}s checkpoint={checkpoint}", log_path)

            child_env = dict(os.environ)
            child_env["PYTHONUNBUFFERED"] = "1"
            with child_stdout.open("ab") as stdout, child_stderr.open("ab") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    env=child_env,
                )
                status["child_pid"] = process.pid
                write_status(args.output_dir, status)
                while process.poll() is None:
                    prevent_sleep()
                    newest = newest_checkpoint(source_dirs, fallback)
                    if newest != checkpoint:
                        copied_policy = copy_latest_policy(args.output_dir, newest)
                        status["latest_policy"] = str(copied_policy)
                        status["latest_checkpoint"] = str(newest)
                        status["updated_at"] = utc_now()
                        write_status(args.output_dir, status)
                    if time.monotonic() > deadline + args.deadline_grace_seconds:
                        stop_process(process, log_path)
                        break
                    time.sleep(max(5.0, args.poll_seconds))

                return_code = process.wait()

            newest = newest_checkpoint(source_dirs, fallback)
            copied_policy = copy_latest_policy(args.output_dir, newest)
            status.update(
                {
                    "updated_at": utc_now(),
                    "child_return_code": return_code,
                    "latest_checkpoint": str(newest),
                    "latest_policy": str(copied_policy),
                }
            )
            write_status(args.output_dir, status)
            log(f"child_exit index={child_index} return_code={return_code} latest_checkpoint={newest}", log_path)

            if return_code != 0:
                final_status = return_code
                if time.monotonic() < deadline:
                    log(f"child_restart_after_failure delay={args.restart_delay_seconds}s", log_path)
                    time.sleep(args.restart_delay_seconds)
                    continue
                break

        checkpoint = newest_checkpoint(source_dirs, fallback)
        copied_policy = copy_latest_policy(args.output_dir, checkpoint)
        write_status(
            args.output_dir,
            {
                "status": "complete",
                "updated_at": utc_now(),
                "latest_checkpoint": str(checkpoint),
                "latest_policy": str(copied_policy),
                "viewer_command_file": str(args.output_dir / VIEW_COMMAND),
            },
        )
        log(f"overnight_complete latest_policy={copied_policy}", log_path)
    finally:
        release_sleep_request()
        log("sleep_prevention=released", log_path)

    return final_status


if __name__ == "__main__":
    sys.exit(main())
