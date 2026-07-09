#!/usr/bin/env python3
"""Check the PyTorch device preference for simulation.

The sim should prefer Intel XPU when the installed PyTorch build exposes a
working backend, then fall back to CPU.
"""

from __future__ import annotations

import sys


def xpu_available(torch_module: object) -> bool:
    xpu = getattr(torch_module, "xpu", None)
    if xpu is None or not hasattr(xpu, "is_available"):
        return False

    try:
        return bool(xpu.is_available())
    except Exception as exc:
        print(f"xpu_available_error={exc}", file=sys.stderr)
        return False


def probe_device(torch_module: object, device_name: str) -> tuple[str, float]:
    try:
        device = torch_module.device(device_name)
        tensor = torch_module.ones((2, 2), device=device)
        result = (tensor @ tensor).sum().item()
        return device_name, float(result)
    except Exception as exc:
        if device_name != "cpu":
            print(f"{device_name}_probe_error={exc}", file=sys.stderr)
            return probe_device(torch_module, "cpu")
        raise


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch import failed: {exc}", file=sys.stderr)
        return 1

    preferred_device = "xpu" if xpu_available(torch) else "cpu"
    selected_device, probe_result = probe_device(torch, preferred_device)

    print(f"torch_version={torch.__version__}")
    print(f"xpu_available={preferred_device == 'xpu'}")
    print(f"selected_device={selected_device}")
    print(f"probe_result={probe_result:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
