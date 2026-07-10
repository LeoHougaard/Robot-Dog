"""Create convex PPO policy blends for conservative checkpoint search."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = PPO.load(args.base, device="cpu")
    candidate = PPO.load(args.candidate, device="cpu")
    base_state = base.policy.state_dict()
    candidate_state = candidate.policy.state_dict()
    if base_state.keys() != candidate_state.keys():
        raise ValueError("Policy checkpoints have different parameter structures")

    for alpha in args.alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        blended: dict[str, torch.Tensor] = {}
        for name, base_value in base_state.items():
            candidate_value = candidate_state[name]
            if torch.is_floating_point(base_value):
                blended[name] = torch.lerp(base_value, candidate_value, alpha)
            else:
                blended[name] = base_value.clone()
        base.policy.load_state_dict(blended, strict=True)
        output = args.output_dir / f"blend_{alpha:.6f}.zip"
        base.save(output)
        print(output)


if __name__ == "__main__":
    main()
