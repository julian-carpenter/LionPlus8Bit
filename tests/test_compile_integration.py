"""Compilation reports must distinguish captured work from eager fallback."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_capture_counter_measures_real_graphs_and_recompilation():
    from _compile_integration import CaptureCounter

    backend = CaptureCounter()
    function = torch.compile(lambda x, scale: x.sin() * scale, backend=backend, fullgraph=True)
    x = torch.randn(8)
    assert torch.equal(function(x, 0.5), x.sin() * 0.5)
    assert torch.equal(function(x, 0.7), x.sin() * 0.7)
    assert backend.graph_count >= 1
    assert backend.operation_count > 0


def test_zero_captured_graphs_never_qualify_optimizer_compile():
    from _compile_integration import compilation_verdict

    assert compilation_verdict(0, parity=True) == "eager-fallback"
    assert compilation_verdict(2, parity=True) == "partial-capture"
    assert compilation_verdict(2, parity=False) == "failed"


def test_optimizer_capture_preserves_adagc_dynamic_active_sets_and_checkpoint():
    from _compile_integration import _optimizer_case

    result = _optimizer_case("cpu", clipping="adagc")
    assert result["parity"]
    assert result["graphs"] > 0
    assert result["checkpoint_roundtrip"]
    assert result["changing_active_sets"]
    assert result["strict_path"]
    assert result["fullgraph"]["status"] == "unsupported"


def test_compile_precision_rejects_unknown_mode_before_running(tmp_path):
    import pytest
    from _compile_integration import run_compile_suite

    with pytest.raises(ValueError, match="precision"):
        run_compile_suite(tmp_path, precision="unknown")
