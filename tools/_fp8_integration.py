"""Optional native FP8 gate: no backend allocation before hardware validation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


def hardware_fp8_reason(capability):
    if tuple(capability) < (8, 9):
        return f"SM{capability[0]}{capability[1]} lacks native FP8 support (requires SM89+)"
    return None


def probe_fp8():
    result = {
        "suite": "fp8",
        "backend": "transformer_engine",
        "status": "unsupported",
        "native_fp8_qualified": False,
    }
    if not torch.cuda.is_available():
        return {
            **result,
            "reason": "CUDA is unavailable; native FP8 requires supported GPU hardware",
        }
    capability = torch.cuda.get_device_capability()
    reason = hardware_fp8_reason(capability)
    result["capability"] = list(capability)
    if reason:
        return {**result, "reason": reason}
    try:
        backend = importlib.import_module("transformer_engine.pytorch")
        available, reason = backend.is_fp8_available(return_reason=True)
    except (ImportError, OSError, AttributeError, RuntimeError) as error:
        return {
            **result,
            "reason": f"Transformer Engine unavailable: {type(error).__name__}: {error}",
        }
    if not available:
        return {**result, "reason": f"Transformer Engine capability check rejected FP8: {reason}"}
    return {
        **result,
        "status": "available-unqualified",
        "reason": "Hardware/backend gates passed; native forward/backward qualification is required",
        "backend_version": getattr(backend, "__version__", "unknown"),
    }


def run_fp8_suite(output, require_fp8=False):
    result = probe_fp8()
    # Availability is never training qualification. A capable host must execute the
    # backend-specific numerical/cache/restart matrix before acquiring support.
    result["required"] = bool(require_fp8)
    result["exit_code"] = 1 if require_fp8 and not result["native_fp8_qualified"] else 0
    Path(output, "fp8.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
