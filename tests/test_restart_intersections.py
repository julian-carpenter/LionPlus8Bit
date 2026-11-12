"""Recovery intersections, including layout, native compilation and remapping."""

import importlib.util
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(os.environ.get("LIONPLUS_RESTART_RUNNER", ROOT / "tools/restart_integration.py"))


def runtime():
    spec = importlib.util.spec_from_file_location("restart_intersections_runtime", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transposed_fixture_keeps_same_logical_weights_with_dense_transposed_storage():
    api = runtime()
    torch.manual_seed(919)
    control = api.StochasticModel(layout="contiguous")
    torch.manual_seed(919)
    target = api.StochasticModel(layout="transposed")
    assert torch.equal(control.layers[0].weight, target.layers[0].weight)
    assert target.layers[0].weight.stride() == (1, 65)
    assert not target.layers[0].weight.is_contiguous()


def test_channels_last_fixture_has_quantizable_convolution_and_matching_forward():
    target = runtime().StochasticModel(layout="channels-last")
    weight = target.layers[0].weight
    assert weight.numel() > 4096
    assert weight.is_contiguous(memory_format=torch.channels_last)
    assert not weight.is_contiguous()
    control = copy.deepcopy(target).to(memory_format=torch.contiguous_format)
    inputs = torch.randn(2, 65)
    torch.manual_seed(42)
    expected = control(inputs)
    torch.manual_seed(42)
    actual = target(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_cpu_remap_rejected_before_output_creation(tmp_path):
    output = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--device",
            "cpu",
            "--resume-visible-devices",
            "2,3",
            "--world-size",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "--resume-visible-devices requires --device cuda" in result.stderr
    assert not output.exists()


def test_remap_device_count_must_match_fixed_world_size(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--device",
            "cuda",
            "--resume-visible-devices",
            "2",
            "--world-size",
            "2",
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "resume device count must match fixed world size" in result.stderr


@pytest.mark.parametrize(
    "layout,backend", [("transposed", None), ("channels-last", None), ("contiguous", "inductor")]
)
def test_actual_crash_recovery_for_layout_and_native_compiler(tmp_path, layout, backend):
    output = tmp_path / "run"
    command = [
        sys.executable,
        str(RUNNER),
        "--device",
        "cpu",
        "--world-size",
        "1",
        "--loader-workers",
        "0",
        "--fault-point",
        "after-step",
        "--layout",
        layout,
        "--precision",
        "fp32",
        "--output",
        str(output),
    ]
    if backend:
        command += ["--compiled-backend", backend]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MAX_JOBS="1")
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "summary.json").read_text())
    assert report["status"] == "passed"
    assert report["layout"] == layout
    assert report["cases"][0]["bitwise_equal"]
    assert report["cases"][0]["restored_checkpoint_bitwise_equal"]
    if backend:
        assert report["compiled_backend"] == backend
        assert report["cases"][0]["compiler_records"][0]["graphs"] > 0
        assert report["native_compiled_cache_recreation_qualified"] is True
