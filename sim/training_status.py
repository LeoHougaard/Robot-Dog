#!/usr/bin/env python3
"""Report whether an SB3 training run is still making progress."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MonitorRow:
    reward: float
    length: int
    elapsed_seconds: float


def find_latest_run(output_dir: Path) -> Path | None:
    candidates = [
        path
        for path in output_dir.glob("simple_quad_*")
        if path.is_dir() and (path / "monitor.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path / "monitor.csv").stat().st_mtime)


def read_monitor_rows(path: Path) -> list[MonitorRow]:
    rows: list[MonitorRow] = []
    with path.open("r", encoding="utf-8") as file:
        filtered = (line for line in file if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(filtered)
        for row in reader:
            try:
                rows.append(
                    MonitorRow(
                        reward=float(row["r"]),
                        length=int(float(row["l"])),
                        elapsed_seconds=float(row["t"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def file_age_seconds(path: Path, now: float) -> float:
    return max(0.0, now - path.stat().st_mtime)


def format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def summarize_window(rows: list[MonitorRow], tail_episodes: int) -> tuple[int, int, float, float] | None:
    if len(rows) < 2:
        return None
    window = rows[-max(2, tail_episodes) :]
    elapsed = window[-1].elapsed_seconds - window[0].elapsed_seconds
    if elapsed <= 0.0:
        return None
    episode_count = len(window)
    transition_count = sum(row.length for row in window)
    transitions_per_second = transition_count / elapsed
    mean_reward = sum(row.reward for row in window) / episode_count
    return episode_count, transition_count, transitions_per_second, mean_reward


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a training run is making progress.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific run directory containing monitor.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sim/runs/target_far_run_ppo"),
        help="Training output directory. The latest simple_quad_* run is used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--max-stale-seconds",
        type=float,
        default=300.0,
        help="Mark a run stale when monitor.csv has not changed for this many seconds.",
    )
    parser.add_argument(
        "--tail-episodes",
        type=int,
        default=100,
        help="Number of recent episodes to use for throughput and reward summary.",
    )
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Return a non-zero exit code when the latest run looks stale.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve() if args.run_dir else find_latest_run(args.output_dir.resolve())
    if run_dir is None:
        print(f"status=missing output_dir={args.output_dir.resolve()}")
        return 2

    monitor_path = run_dir / "monitor.csv"
    if not monitor_path.exists():
        print(f"status=missing_monitor run_dir={run_dir}")
        return 2

    now = time.time()
    rows = read_monitor_rows(monitor_path)
    monitor_age = file_age_seconds(monitor_path, now)
    checkpoint = latest_checkpoint(run_dir)
    final_policy = run_dir / "policy.zip"
    status = "active" if monitor_age <= args.max_stale_seconds else "stale"
    if final_policy.exists() and status == "stale":
        status = "complete_or_idle"

    print(f"status={status}")
    print(f"run_dir={run_dir}")
    print(f"monitor_age={format_age(monitor_age)}")
    print(f"episodes={len(rows)}")
    if rows:
        latest = rows[-1]
        print(f"latest_reward={latest.reward:.3f}")
        print(f"latest_episode_steps={latest.length}")
        print(f"run_elapsed={format_age(latest.elapsed_seconds)}")

    window = summarize_window(rows, args.tail_episodes)
    if window is not None:
        episode_count, transition_count, transitions_per_second, mean_reward = window
        print(f"window_episodes={episode_count}")
        print(f"window_transitions={transition_count}")
        print(f"window_transitions_per_second={transitions_per_second:.1f}")
        print(f"window_mean_reward={mean_reward:.3f}")

    if checkpoint is not None:
        print(f"latest_checkpoint={checkpoint}")
        print(f"checkpoint_age={format_age(file_age_seconds(checkpoint, now))}")
    else:
        print("latest_checkpoint=none")

    if final_policy.exists():
        print(f"final_policy={final_policy}")

    if args.fail_on_stale and status == "stale":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
