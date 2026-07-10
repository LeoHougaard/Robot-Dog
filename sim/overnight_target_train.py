#!/usr/bin/env python3
"""Supervise a long far-target PPO training session with restarts."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
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
BEST_MARKER = "BEST_OVERNIGHT_POLICY.txt"
VIEW_COMMAND = "VIEW_OVERNIGHT_POLICY.ps1"
STATUS_JSON = "overnight_status.json"
SUPERVISOR_LOG = "overnight_supervisor.log"
EVALUATION_JSONL = "checkpoint_evaluations.jsonl"
EVALUATION_LOG = "checkpoint_evaluation.log"
VISUAL_CHECK_JSONL = "visual_checks.jsonl"
VISUAL_CHECK_LOG = "visual_check.log"
VISUAL_REVIEW_MD = "visual_review.md"
VISUAL_PREVIEW_DIR = "visual_previews"
AUTO_RESEARCH_MD = "auto_research.md"
PROMOTED_POLICY_ALIASES = ("best", "latest")


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
    excluded_policy_dirs = set(PROMOTED_POLICY_ALIASES)
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(
            path
            for path in root.rglob("*.zip")
            if path.is_file()
            and path.parent.name not in excluded_policy_dirs
            and not path.name.endswith(".tmp.zip")
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


def quote_ps_scalar(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def target_cli_args(
    args: argparse.Namespace,
    terrain_override: str | None = None,
    include_curriculum: bool = True,
) -> list[str]:
    command = [
        "--terrain",
        str(terrain_override or args.terrain),
        "--episode-seconds",
        str(args.episode_seconds),
        "--target-radius-min",
        str(args.target_radius_min),
        "--target-radius-max",
        str(args.target_radius_max),
        "--success-radius",
        str(args.success_radius),
        "--target-velocity",
        str(args.target_velocity),
    ]
    if args.terrain_seed is not None:
        command.extend(["--terrain-seed", str(args.terrain_seed)])
    if include_curriculum and args.terrain_curriculum:
        command.extend(["--terrain-curriculum", str(args.terrain_curriculum)])
    return command


def target_cli_args_powershell(args: argparse.Namespace) -> str:
    pairs = [
        ("--terrain", args.terrain),
        ("--episode-seconds", args.episode_seconds),
        ("--target-radius-min", args.target_radius_min),
        ("--target-radius-max", args.target_radius_max),
        ("--success-radius", args.success_radius),
        ("--target-velocity", args.target_velocity),
    ]
    if args.terrain_seed is not None:
        pairs.append(("--terrain-seed", args.terrain_seed))
    if args.terrain_curriculum:
        pairs.append(("--terrain-curriculum", args.terrain_curriculum))
    return " ".join(f"{name} {quote_ps_scalar(value)}" for name, value in pairs)


def write_view_command(output_dir: Path, checkpoint: Path, args: argparse.Namespace) -> None:
    command = (
        f"python sim/view_policy_tk.py --task target --policy ppo --checkpoint {quote_ps_path(checkpoint)} "
        f"{target_cli_args_powershell(args)} --seed {int(args.eval_seed)}"
    )
    (output_dir / LATEST_MARKER).write_text(str(checkpoint) + "\n", encoding="utf-8")
    (output_dir / VIEW_COMMAND).write_text(command + "\n", encoding="utf-8")


def copy_promoted_policy(output_dir: Path, checkpoint: Path, args: argparse.Namespace) -> Path:
    source_checkpoint = checkpoint.resolve()
    promoted_policy: Path | None = None
    for alias in PROMOTED_POLICY_ALIASES:
        alias_dir = output_dir / alias
        alias_dir.mkdir(parents=True, exist_ok=True)
        policy_path = alias_dir / "policy.zip"
        if source_checkpoint != policy_path.resolve():
            temp_policy = alias_dir / "policy.tmp.zip"
            shutil.copy2(source_checkpoint, temp_policy)
            temp_policy.replace(policy_path)
        promoted_policy = policy_path.resolve()
    latest_policy = (output_dir / "latest" / "policy.zip").resolve()
    best_policy = (output_dir / "best" / "policy.zip").resolve()
    (output_dir / BEST_MARKER).write_text(str(source_checkpoint) + "\n", encoding="utf-8")
    write_view_command(output_dir, latest_policy, args)
    (output_dir / "VIEW_BEST_POLICY.ps1").write_text(
        (
            f"python sim/view_policy_tk.py --task target --policy ppo --checkpoint {quote_ps_path(best_policy)} "
            f"{target_cli_args_powershell(args)} --seed {int(args.eval_seed)}"
        )
        + "\n",
        encoding="utf-8",
    )
    return promoted_policy or latest_policy


def write_status(output_dir: Path, status: dict) -> None:
    status_path = output_dir / STATUS_JSON
    temp_path = output_dir / f"{STATUS_JSON}.tmp"
    temp_path.write_text(
        json.dumps(json_safe(status), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(status_path)


def metric_float(metrics: dict, key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    return value


def checkpoint_fingerprint(checkpoint: Path) -> dict:
    resolved = checkpoint.resolve()
    stat = resolved.stat()
    return {
        "checkpoint": str(resolved),
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def evaluation_key(record: dict) -> str:
    return f"{record.get('checkpoint')}|{record.get('mtime_ns')}|{record.get('size')}"


def load_evaluation_records(output_dir: Path) -> list[dict]:
    path = output_dir / EVALUATION_JSONL
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "checkpoint" in record:
                records.append(record)
    return records


def append_evaluation_record(output_dir: Path, record: dict) -> None:
    path = output_dir / EVALUATION_JSONL
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n")


def parse_evaluation_metrics(stdout: str) -> dict:
    metrics: dict[str, float | str] = {}
    float_keys = {
        "mean_return",
        "mean_length",
        "success_rate",
        "mean_time_to_success",
        "target_distance_reduction",
        "mean_path_efficiency",
        "recovery_success_rate",
    }
    string_keys = {
        "successes",
        "failure_sectors",
        "terrain",
        "terrain_seed",
        "terrain_curriculum",
        "terrain_counts",
        "reset_pose_counts",
        "success_pose_counts",
    }
    for line in stdout.splitlines():
        for token in line.strip().split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key in float_keys:
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = float("nan")
            elif key in string_keys:
                metrics[key] = value
    return metrics


def parse_key_value_stdout(stdout: str) -> dict:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        for token in line.strip().split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            values[key] = value
    return values


def parse_number(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_visual_records(output_dir: Path) -> list[dict]:
    path = output_dir / VISUAL_CHECK_JSONL
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "created_at" in record:
                records.append(record)
    return records


def append_visual_record(output_dir: Path, record: dict) -> None:
    path = output_dir / VISUAL_CHECK_JSONL
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_safe(record), sort_keys=True, allow_nan=False) + "\n")


def render_command(args: argparse.Namespace, checkpoint: Path, seed: int, output_path: Path) -> list[str]:
    terrain_levels = [
        value.strip() for value in str(args.terrain_curriculum or "flat").split(",") if value.strip()
    ]
    terrain = terrain_levels[(seed - int(args.eval_seed)) % len(terrain_levels)]
    return [
        sys.executable,
        str(REPO_ROOT / "sim/render_policy.py"),
        "--task",
        "target",
        "--policy",
        "ppo",
        "--checkpoint",
        str(checkpoint),
        "--steps",
        str(args.visual_check_steps),
        "--frames",
        str(args.visual_check_frames),
        "--output",
        str(output_path),
        "--seed",
        str(seed),
        *target_cli_args(args, terrain_override=terrain, include_curriculum=False),
    ]


def visual_run_findings(metrics: dict, max_steps: int) -> list[str]:
    findings: list[str] = []
    if metrics.get("render_return_code", 0) != 0:
        findings.append("render_failed")
        return findings

    steps = parse_number(metrics.get("steps"), 0.0)
    success = parse_bool(metrics.get("success"))
    status = str(metrics.get("status", "unknown"))
    base_height = parse_number(metrics.get("base_height"))
    upright = parse_number(metrics.get("upright"))
    reduction = parse_number(metrics.get("target_distance_reduction"), 0.0)

    if success:
        findings.append("success")
    elif status == "terminated":
        findings.append("failed_terminated")
    else:
        findings.append("no_success")

    if not success and steps < max(80.0, 0.35 * max_steps):
        findings.append("early_stop")
    if base_height < 0.07 or upright < math.cos(0.85):
        findings.append("fallen_or_tipped")
    elif upright > math.cos(0.55):
        findings.append("upright")

    if reduction >= 0.30:
        findings.append("clear_target_progress")
    elif reduction >= 0.10:
        findings.append("some_target_progress")
    elif reduction < -0.05:
        findings.append("target_regression")
    else:
        findings.append("little_target_progress")

    if not success and steps >= 0.85 * max_steps:
        findings.append("survives_without_success")
    return findings


def visual_signature(runs: list[dict]) -> list[str]:
    signature: list[str] = []
    for run in runs:
        metrics = run.get("metrics", {})
        steps_bucket = int(parse_number(metrics.get("steps"), 0.0) // 100.0)
        reduction_bucket = int(round(parse_number(metrics.get("target_distance_reduction"), 0.0) * 10.0))
        run_tokens = sorted(str(finding) for finding in run.get("findings", []))
        signature.append(
            f"run={run.get('run_index')}|steps={steps_bucket}|reduction={reduction_bucket}|"
            + ",".join(run_tokens)
        )
    return signature


def signature_similarity(current: list[str], previous: list[str]) -> float:
    current_set = set(current)
    previous_set = set(previous)
    if not current_set and not previous_set:
        return 1.0
    union = current_set | previous_set
    if not union:
        return 0.0
    return len(current_set & previous_set) / len(union)


def clamp_interval(value: float, args: argparse.Namespace) -> float:
    return max(
        float(args.visual_check_min_interval_seconds),
        min(float(args.visual_check_max_interval_seconds), float(value)),
    )


def next_visual_interval(
    current_interval: float,
    current_signature: list[str],
    previous_record: dict | None,
    args: argparse.Namespace,
) -> tuple[float, str, float | None]:
    if previous_record is None:
        return clamp_interval(current_interval, args), "baseline", None

    previous_signature = [str(value) for value in previous_record.get("finding_signature", [])]
    similarity = signature_similarity(current_signature, previous_signature)
    if similarity >= 0.80:
        return clamp_interval(current_interval * 1.50, args), "too_similar", similarity
    if similarity <= 0.55:
        return clamp_interval(current_interval * 0.75, args), "different", similarity
    return clamp_interval(current_interval, args), "mixed", similarity


def run_visual_check(
    args: argparse.Namespace,
    checkpoint: Path,
    previous_record: dict | None,
    check_index: int,
    current_interval: float,
    log_path: Path,
) -> dict:
    visual_dir = args.output_dir / VISUAL_PREVIEW_DIR / f"check_{check_index:04d}"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_log_path = args.output_dir / VISUAL_CHECK_LOG
    runs: list[dict] = []
    log(f"visual_check_start index={check_index} checkpoint={checkpoint}", log_path)

    for run_index in range(max(1, int(args.visual_check_runs))):
        seed = int(args.eval_seed + run_index)
        output_path = visual_dir / f"run_{run_index:02d}_seed_{seed}.png"
        command = render_command(args, checkpoint, seed, output_path)
        run_record: dict = {
            "run_index": run_index,
            "seed": seed,
            "image": str(output_path.resolve()),
            "command": command,
            "metrics": {},
            "findings": [],
        }
        with visual_log_path.open("a", encoding="utf-8") as visual_log:
            visual_log.write(f"\n{utc_now()} visual_check={check_index} run={run_index} seed={seed}\n")
            visual_log.write("command=" + " ".join(command) + "\n")
            try:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=max(1.0, args.visual_check_timeout_seconds),
                )
            except subprocess.TimeoutExpired as exc:
                run_record["metrics"] = {"render_return_code": 124}
                run_record["findings"] = ["render_timeout"]
                run_record["error"] = f"visual_render_timeout_after_{args.visual_check_timeout_seconds}s"
                visual_log.write(str(run_record["error"]) + "\n")
                if exc.stdout:
                    visual_log.write(str(exc.stdout) + "\n")
                if exc.stderr:
                    visual_log.write(str(exc.stderr) + "\n")
                runs.append(run_record)
                continue

            visual_log.write(result.stdout)
            if result.stderr:
                visual_log.write("\n[stderr]\n")
                visual_log.write(result.stderr)

        metrics = parse_key_value_stdout(result.stdout)
        metrics["render_return_code"] = result.returncode
        run_record["metrics"] = metrics
        if result.returncode != 0:
            run_record["error"] = f"visual_render_failed_return_code_{result.returncode}"
        run_record["findings"] = visual_run_findings(metrics, args.visual_check_steps)
        runs.append(run_record)

    finding_signature = visual_signature(runs)
    interval_after, interval_reason, similarity = next_visual_interval(
        current_interval,
        finding_signature,
        previous_record,
        args,
    )
    all_findings = sorted({finding for run in runs for finding in run.get("findings", [])})
    record = {
        "created_at": utc_now(),
        "check_index": check_index,
        "checkpoint": str(checkpoint.resolve()),
        "interval_before_seconds": float(current_interval),
        "interval_after_seconds": float(interval_after),
        "interval_reason": interval_reason,
        "similarity_to_previous": similarity,
        "finding_signature": finding_signature,
        "findings": all_findings,
        "runs": runs,
    }
    append_visual_record(args.output_dir, record)
    write_visual_review(args.output_dir, record)
    log(
        f"visual_check_complete index={check_index} reason={interval_reason} "
        f"next_interval={interval_after:.1f}s findings={','.join(all_findings) or 'none'}",
        log_path,
    )
    return record


def write_visual_review(output_dir: Path, record: dict) -> None:
    lines = [
        "# Latest Visual Check",
        "",
        f"Updated: {utc_now()}",
        f"Checkpoint: `{record.get('checkpoint')}`",
        f"Interval before: {float(record.get('interval_before_seconds', 0.0)):.1f}s",
        f"Interval after: {float(record.get('interval_after_seconds', 0.0)):.1f}s",
        f"Reason: {record.get('interval_reason')}",
        f"Similarity to previous: {record.get('similarity_to_previous')}",
        "",
        "## Findings",
        "",
    ]
    findings = record.get("findings", [])
    if findings:
        lines.extend(f"- {finding}" for finding in findings)
    else:
        lines.append("- none")
    lines.extend(["", "## Screenshots", ""])
    for run in record.get("runs", []):
        image = run.get("image")
        try:
            image_ref = Path(str(image)).resolve().relative_to(output_dir.resolve()).as_posix()
        except (OSError, ValueError):
            image_ref = str(image)
        run_findings = ", ".join(str(value) for value in run.get("findings", [])) or "none"
        lines.append(f"- Run {run.get('run_index')} seed {run.get('seed')}: `{image}`")
        lines.append(f"  Findings: {run_findings}")
        lines.append(f"  ![run {run.get('run_index')} seed {run.get('seed')}]({image_ref})")
    (output_dir / VISUAL_REVIEW_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_evaluation(metrics: dict, max_steps: int) -> float:
    success_rate = metric_float(metrics, "success_rate")
    distance_reduction = metric_float(metrics, "target_distance_reduction")
    mean_length = metric_float(metrics, "mean_length")
    mean_return = metric_float(metrics, "mean_return")
    mean_time_to_success = metric_float(metrics, "mean_time_to_success", default=float("nan"))
    path_efficiency = metric_float(metrics, "mean_path_efficiency")
    recovery_success_rate = metric_float(metrics, "recovery_success_rate")
    length_fraction = max(0.0, min(1.0, mean_length / max(1, max_steps)))
    fast_fall_penalty = max(0.0, 0.45 - length_fraction) * 300.0
    time_bonus = 0.0
    if success_rate > 0.0 and math.isfinite(mean_time_to_success):
        time_bonus = 2.0 * max(0.0, 30.0 - mean_time_to_success)
    return (
        1000.0 * success_rate
        + 180.0 * distance_reduction
        + 90.0 * length_fraction
        + 80.0 * path_efficiency
        + 500.0 * recovery_success_rate
        + 0.01 * mean_return
        + time_bonus
        - fast_fall_penalty
    )


def evaluation_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "sim/evaluate.py"),
        "--task",
        "target",
        "--policy",
        "ppo",
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(args.eval_episodes),
        "--max-steps",
        str(args.eval_max_steps),
        "--seed",
        str(args.eval_seed),
        *(["--seed-blocks", args.eval_seed_blocks] if args.eval_seed_blocks else []),
        *target_cli_args(args),
    ]


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, log_path: Path) -> dict:
    fingerprint = checkpoint_fingerprint(checkpoint)
    command = evaluation_command(args, checkpoint)
    record = {
        **fingerprint,
        "evaluated_at": utc_now(),
        "command": command,
        "ok": False,
        "score": -1_000_000_000.0,
        "metrics": {},
    }
    log(f"eval_start checkpoint={checkpoint}", log_path)
    eval_log_path = args.output_dir / EVALUATION_LOG
    with eval_log_path.open("a", encoding="utf-8") as eval_log:
        eval_log.write(f"\n{utc_now()} checkpoint={checkpoint}\n")
        eval_log.write("command=" + " ".join(command) + "\n")
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=max(1.0, args.eval_timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            record["error"] = f"evaluation_timeout_after_{args.eval_timeout_seconds}s"
            eval_log.write(record["error"] + "\n")
            if exc.stdout:
                eval_log.write(str(exc.stdout) + "\n")
            if exc.stderr:
                eval_log.write(str(exc.stderr) + "\n")
            log(f"eval_timeout checkpoint={checkpoint}", log_path)
            return record

        eval_log.write(result.stdout)
        if result.stderr:
            eval_log.write("\n[stderr]\n")
            eval_log.write(result.stderr)

    metrics = parse_evaluation_metrics(result.stdout)
    record["return_code"] = result.returncode
    record["metrics"] = metrics
    if result.returncode == 0 and metrics:
        record["ok"] = True
        record["score"] = score_evaluation(metrics, args.eval_max_steps)
        log(
            "eval_complete "
            f"score={record['score']:.3f} success_rate={metric_float(metrics, 'success_rate'):.3f} "
            f"mean_length={metric_float(metrics, 'mean_length'):.1f} "
            f"distance_reduction={metric_float(metrics, 'target_distance_reduction'):.3f} "
            f"checkpoint={checkpoint}",
            log_path,
        )
    else:
        record["error"] = f"evaluation_failed_return_code_{result.returncode}"
        log(f"eval_failed return_code={result.returncode} checkpoint={checkpoint}", log_path)
    return record


def evaluate_once(
    args: argparse.Namespace,
    checkpoint: Path,
    records: list[dict],
    log_path: Path,
) -> dict:
    fingerprint = checkpoint_fingerprint(checkpoint)
    key = evaluation_key(fingerprint)
    for record in records:
        if evaluation_key(record) == key:
            return record
    record = evaluate_checkpoint(args, checkpoint, log_path)
    records.append(record)
    append_evaluation_record(args.output_dir, record)
    return record


def best_evaluation(records: list[dict]) -> dict | None:
    valid = [
        record
        for record in records
        if record.get("ok") and Path(str(record.get("checkpoint", ""))).exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda record: metric_float(record, "score", default=-1_000_000_000.0))


def guidance_from_metrics(metrics: dict, max_steps: int) -> list[str]:
    guidance: list[str] = []
    success_rate = metric_float(metrics, "success_rate")
    mean_length = metric_float(metrics, "mean_length")
    distance_reduction = metric_float(metrics, "target_distance_reduction")
    length_fraction = mean_length / max(1, max_steps)
    if length_fraction < 0.35:
        guidance.append("Fast termination is still present; inspect crash states before trusting PPO updates.")
    if success_rate <= 0.0 and distance_reduction < 0.10:
        guidance.append("Policy is not making target progress; bias the next intervention toward reference/DAgger recovery.")
    if success_rate > 0.0 and distance_reduction > 0.20:
        guidance.append("Preserve this checkpoint and make PPO changes conservatively around it.")
    failure_sectors = str(metrics.get("failure_sectors", "none"))
    if failure_sectors not in {"", "none"}:
        guidance.append(f"Evaluate hard target sectors explicitly: {failure_sectors}.")
    return guidance or ["Continue the current recipe and compare the next checkpoint by evaluation score."]


def write_autoresearch_summary(
    output_dir: Path,
    args: argparse.Namespace,
    records: list[dict],
    latest_candidate: Path | None,
) -> None:
    best = best_evaluation(records)
    visual_records = load_visual_records(output_dir)
    latest_visual = visual_records[-1] if visual_records else None
    lines = [
        "# Target Autoresearch Status",
        "",
        "This run uses an ENPIRE-inspired loop for the sim target task.",
        "",
        "- EN: `SimpleQuadTargetEnv` supplies reset, target sampling, safety termination, and reward signals.",
        "- PI: `sim/train.py` launches conservative PPO fine-tuning chunks from the current checkpoint.",
        "- R: `sim/evaluate.py` evaluates checkpoints with fixed target/evaluation settings.",
        "- E: this supervisor scores evaluations and promotes the best checkpoint instead of the newest one.",
        "",
        f"Updated: {utc_now()}",
        f"Output dir: `{output_dir}`",
        f"Latest candidate: `{latest_candidate}`" if latest_candidate is not None else "Latest candidate: none",
        "",
    ]
    if best is None:
        lines.extend(
            [
                "## Best Evaluation",
                "",
                "No successful checkpoint evaluation has been recorded yet.",
            ]
        )
    else:
        metrics = best.get("metrics", {})
        lines.extend(
            [
                "## Best Evaluation",
                "",
                f"- Source checkpoint: `{best.get('checkpoint')}`",
                f"- Score: {metric_float(best, 'score'):.3f}",
                f"- Success rate: {metric_float(metrics, 'success_rate'):.3f}",
                f"- Mean length: {metric_float(metrics, 'mean_length'):.1f} / {args.eval_max_steps}",
                f"- Target distance reduction: {metric_float(metrics, 'target_distance_reduction'):.3f}",
                f"- Failure sectors: {metrics.get('failure_sectors', 'none')}",
                "",
                "## Agent Guidance",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in guidance_from_metrics(metrics, args.eval_max_steps))
    lines.extend(["", "## Latest Visual Check", ""])
    if latest_visual is None:
        lines.append("No visual check has been recorded yet.")
    else:
        lines.extend(
            [
                f"- Created: {latest_visual.get('created_at')}",
                f"- Interval reason: {latest_visual.get('interval_reason')}",
                f"- Next interval: {float(latest_visual.get('interval_after_seconds', 0.0)):.1f}s",
                f"- Findings: {', '.join(str(item) for item in latest_visual.get('findings', [])) or 'none'}",
                f"- Review file: `{output_dir / VISUAL_REVIEW_MD}`",
            ]
        )
        for run in latest_visual.get("runs", []):
            lines.append(f"- Screenshot run {run.get('run_index')} seed {run.get('seed')}: `{run.get('image')}`")
    output_path = output_dir / AUTO_RESEARCH_MD
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def promote_best_or_fallback(
    args: argparse.Namespace,
    records: list[dict],
    fallback_checkpoint: Path,
    log_path: Path,
) -> tuple[Path, Path, dict | None]:
    best = best_evaluation(records)
    if best is not None:
        checkpoint = Path(str(best["checkpoint"])).resolve()
        promoted_policy = copy_promoted_policy(args.output_dir, checkpoint, args)
        log(
            f"promoted_best score={metric_float(best, 'score'):.3f} checkpoint={checkpoint} policy={promoted_policy}",
            log_path,
        )
        return checkpoint, promoted_policy, best
    checkpoint = fallback_checkpoint.resolve()
    promoted_policy = copy_promoted_policy(args.output_dir, checkpoint, args)
    log(f"promoted_fallback checkpoint={checkpoint} policy={promoted_policy}", log_path)
    return checkpoint, promoted_policy, None


def train_command(
    args: argparse.Namespace,
    checkpoint: Path,
    max_wall_seconds: int,
    child_index: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "sim/train.py"),
        "--task",
        "target",
        "--device",
        "cpu",
        "--seed",
        str(int(args.seed) + int(child_index)),
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
        "--clip-range",
        str(args.clip_range),
        *target_cli_args(args),
        "--output-dir",
        str(args.output_dir),
        "--verbose",
        "0",
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--max-wall-seconds",
        str(max_wall_seconds),
    ]
    if args.target_kl is not None:
        command.extend(["--target-kl", str(args.target_kl)])
    return command


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


def deadline_utc_after_monotonic(target_monotonic: float) -> str:
    remaining = max(0.0, target_monotonic - time.monotonic())
    return datetime.fromtimestamp(time.time() + remaining, timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENPIRE-style supervised far-target PPO training.")
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=1, help="Base seed; each restarted child receives a new seed.")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--checkpoint-freq", type=int, default=100000)
    parser.add_argument("--chunk-timesteps", type=int, default=10_000_000)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--restart-delay-seconds", type=float, default=20.0)
    parser.add_argument("--deadline-grace-seconds", type=float, default=300.0)
    parser.add_argument("--terrain", default="curriculum")
    parser.add_argument("--terrain-seed", type=int)
    parser.add_argument(
        "--terrain-curriculum",
        default="flat,mild,rough,hard",
        help="Comma-separated surfaces mixed throughout training and deterministic evaluation.",
    )
    parser.add_argument("--episode-seconds", type=float, default=30.0)
    parser.add_argument("--target-radius-min", type=float, default=1.1)
    parser.add_argument("--target-radius-max", type=float, default=1.6)
    parser.add_argument("--success-radius", type=float, default=0.22)
    parser.add_argument("--target-velocity", type=float, default=0.30)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=1500)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument(
        "--eval-seed-blocks",
        default="",
        help="Comma-separated evaluation base seeds; evaluates --eval-episodes per block.",
    )
    parser.add_argument("--eval-interval-seconds", type=float, default=300.0)
    parser.add_argument("--eval-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--min-checkpoint-age-seconds", type=float, default=10.0)
    parser.add_argument("--visual-check-interval-seconds", type=float, default=180.0)
    parser.add_argument("--visual-check-min-interval-seconds", type=float, default=60.0)
    parser.add_argument("--visual-check-max-interval-seconds", type=float, default=1800.0)
    parser.add_argument("--visual-check-runs", type=int, default=3)
    parser.add_argument("--visual-check-steps", type=int, default=600)
    parser.add_argument("--visual-check-frames", type=int, default=6)
    parser.add_argument("--visual-check-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--disable-visual-checks",
        action="store_true",
        help="Disable rendered rollout contact sheets and adaptive visual review interval updates.",
    )
    parser.add_argument(
        "--disable-auto-eval",
        action="store_true",
        help="Promote the newest checkpoint without running deterministic checkpoint evaluation.",
    )
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / SUPERVISOR_LOG
    if args.initial_checkpoint:
        source_dirs = [args.output_dir]
        fallback = args.initial_checkpoint.resolve()
    else:
        source_dirs = [args.output_dir, *DEFAULT_SOURCE_DIRS]
        fallback = DEFAULT_FALLBACK_CHECKPOINT
    deadline = time.monotonic() + max(1.0, args.hours * 3600.0)
    final_status = 0
    child_index = 0
    records = load_evaluation_records(args.output_dir)
    visual_records = load_visual_records(args.output_dir)
    latest_visual_record = visual_records[-1] if visual_records else None
    visual_check_index = int(latest_visual_record.get("check_index", 0)) + 1 if latest_visual_record else 1
    visual_interval = clamp_interval(
        latest_visual_record.get("interval_after_seconds", args.visual_check_interval_seconds)
        if latest_visual_record
        else args.visual_check_interval_seconds,
        args,
    )
    next_visual_check_at = time.monotonic()
    latest_candidate: Path | None = None
    promoted_checkpoint = newest_checkpoint(source_dirs, fallback)

    log(f"enpire_start hours={args.hours} output_dir={args.output_dir}", log_path)
    log("sleep_prevention=enabled_while_supervisor_runs", log_path)
    prevent_sleep()
    try:
        if not args.disable_auto_eval:
            evaluate_once(args, promoted_checkpoint, records, log_path)
        promoted_checkpoint, promoted_policy, best_record = promote_best_or_fallback(
            args,
            records,
            promoted_checkpoint,
            log_path,
        )
        write_autoresearch_summary(args.output_dir, args, records, promoted_checkpoint)
        if not args.disable_visual_checks:
            latest_visual_record = run_visual_check(
                args,
                promoted_checkpoint,
                latest_visual_record,
                visual_check_index,
                visual_interval,
                log_path,
            )
            visual_interval = float(latest_visual_record["interval_after_seconds"])
            visual_check_index += 1
            next_visual_check_at = time.monotonic() + visual_interval
            write_autoresearch_summary(args.output_dir, args, records, promoted_checkpoint)

        while time.monotonic() < deadline:
            checkpoint = promoted_checkpoint
            remaining = max(1, int(deadline - time.monotonic()))
            child_index += 1
            child_stdout = args.output_dir / f"train_child_{child_index:03d}.stdout.log"
            child_stderr = args.output_dir / f"train_child_{child_index:03d}.stderr.log"
            command = train_command(args, checkpoint, remaining, child_index)
            status = {
                "status": "running",
                "updated_at": utc_now(),
                "deadline_utc": datetime.fromtimestamp(time.time() + remaining, timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "child_index": child_index,
                "child_pid": None,
                "input_checkpoint": str(checkpoint),
                "latest_candidate": str(latest_candidate) if latest_candidate is not None else None,
                "best_checkpoint": str(promoted_checkpoint),
                "best_evaluation": best_record,
                "latest_policy": str(promoted_policy),
                "viewer_command_file": str(args.output_dir / VIEW_COMMAND),
                "autoresearch_file": str(args.output_dir / AUTO_RESEARCH_MD),
                "visual_review_file": str(args.output_dir / VISUAL_REVIEW_MD),
                "visual_check_interval_seconds": float(visual_interval),
                "next_visual_check_utc": None
                if args.disable_visual_checks
                else deadline_utc_after_monotonic(next_visual_check_at),
                "latest_visual_check": latest_visual_record,
                "stdout_log": str(child_stdout),
                "stderr_log": str(child_stderr),
                "command": command,
            }
            write_status(args.output_dir, status)
            log(f"child_start index={child_index} remaining={remaining}s checkpoint={checkpoint}", log_path)

            child_env = dict(os.environ)
            child_env["PYTHONUNBUFFERED"] = "1"
            last_eval_at = 0.0
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
                    latest_candidate = newest
                    now = time.monotonic()
                    candidate_age = max(0.0, time.time() - newest.stat().st_mtime)
                    should_evaluate = (
                        not args.disable_auto_eval
                        and newest != checkpoint
                        and candidate_age >= args.min_checkpoint_age_seconds
                        and now - last_eval_at >= args.eval_interval_seconds
                    )
                    if should_evaluate:
                        evaluate_once(args, newest, records, log_path)
                        promoted_checkpoint, promoted_policy, best_record = promote_best_or_fallback(
                            args,
                            records,
                            newest,
                            log_path,
                        )
                        write_autoresearch_summary(args.output_dir, args, records, latest_candidate)
                        last_eval_at = now
                    elif args.disable_auto_eval and newest != promoted_checkpoint:
                        promoted_checkpoint = newest
                        promoted_policy = copy_promoted_policy(args.output_dir, newest, args)

                    if not args.disable_visual_checks and time.monotonic() >= next_visual_check_at:
                        latest_visual_record = run_visual_check(
                            args,
                            promoted_checkpoint,
                            latest_visual_record,
                            visual_check_index,
                            visual_interval,
                            log_path,
                        )
                        visual_interval = float(latest_visual_record["interval_after_seconds"])
                        visual_check_index += 1
                        next_visual_check_at = time.monotonic() + visual_interval
                        write_autoresearch_summary(args.output_dir, args, records, latest_candidate)

                    status.update(
                        {
                            "updated_at": utc_now(),
                            "latest_candidate": str(latest_candidate),
                            "best_checkpoint": str(promoted_checkpoint),
                            "best_evaluation": best_record,
                            "latest_policy": str(promoted_policy),
                            "visual_check_interval_seconds": float(visual_interval),
                            "next_visual_check_utc": None
                            if args.disable_visual_checks
                            else deadline_utc_after_monotonic(next_visual_check_at),
                            "latest_visual_check": latest_visual_record,
                        }
                    )
                    write_status(args.output_dir, status)
                    if time.monotonic() > deadline + args.deadline_grace_seconds:
                        stop_process(process, log_path)
                        break
                    time.sleep(max(5.0, args.poll_seconds))

                return_code = process.wait()

            newest = newest_checkpoint(source_dirs, fallback)
            latest_candidate = newest
            if not args.disable_auto_eval:
                evaluate_once(args, newest, records, log_path)
                promoted_checkpoint, promoted_policy, best_record = promote_best_or_fallback(
                    args,
                    records,
                    newest,
                    log_path,
                )
            else:
                promoted_checkpoint = newest
                promoted_policy = copy_promoted_policy(args.output_dir, newest, args)
                best_record = None
            if not args.disable_visual_checks and time.monotonic() >= next_visual_check_at:
                latest_visual_record = run_visual_check(
                    args,
                    promoted_checkpoint,
                    latest_visual_record,
                    visual_check_index,
                    visual_interval,
                    log_path,
                )
                visual_interval = float(latest_visual_record["interval_after_seconds"])
                visual_check_index += 1
                next_visual_check_at = time.monotonic() + visual_interval
            write_autoresearch_summary(args.output_dir, args, records, latest_candidate)
            status.update(
                {
                    "updated_at": utc_now(),
                    "child_return_code": return_code,
                    "latest_candidate": str(newest),
                    "best_checkpoint": str(promoted_checkpoint),
                    "best_evaluation": best_record,
                    "latest_policy": str(promoted_policy),
                    "visual_check_interval_seconds": float(visual_interval),
                    "next_visual_check_utc": None
                    if args.disable_visual_checks
                    else deadline_utc_after_monotonic(next_visual_check_at),
                    "latest_visual_check": latest_visual_record,
                }
            )
            write_status(args.output_dir, status)
            log(
                f"child_exit index={child_index} return_code={return_code} "
                f"latest_candidate={newest} best_checkpoint={promoted_checkpoint}",
                log_path,
            )

            if return_code != 0:
                final_status = return_code
                if time.monotonic() < deadline:
                    log(f"child_restart_after_failure delay={args.restart_delay_seconds}s", log_path)
                    time.sleep(args.restart_delay_seconds)
                    continue
                break

        latest_candidate = newest_checkpoint(source_dirs, fallback)
        if not args.disable_auto_eval:
            evaluate_once(args, latest_candidate, records, log_path)
            promoted_checkpoint, promoted_policy, best_record = promote_best_or_fallback(
                args,
                records,
                latest_candidate,
                log_path,
            )
        else:
            promoted_checkpoint = latest_candidate
            promoted_policy = copy_promoted_policy(args.output_dir, latest_candidate, args)
            best_record = None
        latest_visual_checkpoint = (
            str(latest_visual_record.get("checkpoint")) if latest_visual_record is not None else None
        )
        if not args.disable_visual_checks and latest_visual_checkpoint != str(promoted_checkpoint.resolve()):
            latest_visual_record = run_visual_check(
                args,
                promoted_checkpoint,
                latest_visual_record,
                visual_check_index,
                visual_interval,
                log_path,
            )
            visual_interval = float(latest_visual_record["interval_after_seconds"])
            visual_check_index += 1
            next_visual_check_at = time.monotonic() + visual_interval
        write_autoresearch_summary(args.output_dir, args, records, latest_candidate)
        write_status(
            args.output_dir,
            {
                "status": "complete",
                "updated_at": utc_now(),
                "latest_candidate": str(latest_candidate),
                "best_checkpoint": str(promoted_checkpoint),
                "best_evaluation": best_record,
                "latest_policy": str(promoted_policy),
                "viewer_command_file": str(args.output_dir / VIEW_COMMAND),
                "autoresearch_file": str(args.output_dir / AUTO_RESEARCH_MD),
                "visual_review_file": str(args.output_dir / VISUAL_REVIEW_MD),
                "visual_check_interval_seconds": float(visual_interval),
                "next_visual_check_utc": None
                if args.disable_visual_checks
                else deadline_utc_after_monotonic(next_visual_check_at),
                "latest_visual_check": latest_visual_record,
            },
        )
        log(f"enpire_complete best_checkpoint={promoted_checkpoint} latest_policy={promoted_policy}", log_path)
    finally:
        release_sleep_request()
        log("sleep_prevention=released", log_path)

    return final_status


if __name__ == "__main__":
    sys.exit(main())
