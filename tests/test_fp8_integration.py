"""Native FP8 gates never count emulation as qualification."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_sm75_rejected_before_optional_backend_import():
    from _fp8_integration import hardware_fp8_reason

    assert "SM75" in hardware_fp8_reason((7, 5))
    assert hardware_fp8_reason((8, 9)) is None


def test_cpu_fp8_probe_is_explicitly_unsupported(monkeypatch):
    import _fp8_integration as integration

    monkeypatch.setattr(integration.torch.cuda, "is_available", lambda: False)
    result = integration.probe_fp8()
    assert result["status"] == "unsupported"
    assert result["reason"]
    assert result["native_fp8_qualified"] is False


def test_capable_backend_uses_public_availability_reason_interface(monkeypatch):
    import types
    import _fp8_integration as integration

    monkeypatch.setattr(integration.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(integration.torch.cuda, "get_device_capability", lambda: (9, 0))
    calls = []

    def check(*, return_reason=False):
        calls.append(return_reason)
        return (True, "") if return_reason else True

    backend = types.SimpleNamespace(is_fp8_available=check, __version__="fixture")
    monkeypatch.setattr(integration.importlib, "import_module", lambda name: backend)
    result = integration.probe_fp8()
    assert calls == [True]
    assert result["status"] == "available-unqualified"
    assert result["native_fp8_qualified"] is False


def test_fp8_example_refuses_cpu_without_optional_backend_allocation():
    import os
    import subprocess
    import sys

    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "examples/train_fp8.py"),
        "--steps",
        "1",
    ]
    result = subprocess.run(
        command,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "CUDA is unavailable" in result.stdout + result.stderr
