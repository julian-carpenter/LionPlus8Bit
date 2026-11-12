import importlib.util
from pathlib import Path
import sys
import pytest
import torch


def test_required_gpu_count_is_not_silently_ignored():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--require-gpus", "999", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode != 0
    assert "999" in result.stderr


def test_capability_output_refuses_overwrite(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "integration_common", root / "tools/_integration_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(FileExistsError):
        module.output_directory(tmp_path)


def ddp_runner():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ddp_integration", root / "tools/ddp_integration.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "tools"))
    spec.loader.exec_module(module)
    return module


def test_uneven_sample_weight_matches_full_batch_gradient():
    runner = ddp_runner()
    weight = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)
    x = torch.tensor(
        [[1.0, 2.0], [-2.0, 3.0], [0.5, -1.0], [2.0, -0.5], [1.5, 0.2]], dtype=torch.float64
    )
    expected = torch.autograd.grad((x @ weight).square().mean(), weight)[0]
    gradients = []
    for partition in (x[:2], x[2:]):
        local = weight.detach().clone().requires_grad_()
        coefficient = runner.global_mean_loss_weight(len(partition), len(x), 2)
        gradients.append(
            torch.autograd.grad((partition @ local).square().mean() * coefficient, local)[0]
        )
    torch.testing.assert_close(torch.stack(gradients).mean(0), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("local,total,world", [(0, 5, 2), (3, 2, 2), (2, 5, 0), (True, 5, 2)])
def test_global_mean_weight_rejects_empty_or_inconsistent_participation(local, total, world):
    with pytest.raises(ValueError, match="sample counts|world size"):
        ddp_runner().global_mean_loss_weight(local, total, world)


def test_public_scaler_agreement_checks_growth_tracker_and_scale():
    runner = ddp_runner()
    state = dict(
        scale=32.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=30, _growth_tracker=4
    )
    runner.validate_scaler_states([state, dict(state)])
    with pytest.raises(RuntimeError, match="scaler"):
        runner.validate_scaler_states([state, dict(state, scale=64.0)])
    with pytest.raises(RuntimeError, match="scaler"):
        runner.validate_scaler_states([state, dict(state, _growth_tracker=5)])


def test_post_reduction_finiteness_gate_accepts_only_coordinated_status():
    runner = ddp_runner()
    assert runner.validate_finite_flags([True, True]) is True
    assert runner.validate_finite_flags([False, False]) is False
    with pytest.raises(RuntimeError, match="Rank-local nonfinite"):
        runner.validate_finite_flags([True, False])


def test_input_exhaustion_stops_every_rank_before_backward():
    runner = ddp_runner()
    assert runner.validate_input_availability([True, True]) is True
    assert runner.validate_input_availability([False, False]) is False
    with pytest.raises(RuntimeError, match="exhausted input"):
        runner.validate_input_availability([False, True])


def test_independent_gradient_norm_does_not_sum_replicated_rank_norms():
    norm = ddp_runner().independent_gradient_norm(
        [torch.tensor([3.0, 4.0]), None, torch.tensor([12.0])]
    )
    assert norm.dtype == torch.float64
    assert float(norm) == 13.0


@pytest.mark.parametrize("transpose", [False, True])
def test_gradient_replay_preserves_bucket_alignment_and_strides(transpose):
    source = torch.arange(13, dtype=torch.float32)[1:].view(3, 4)
    if transpose:
        source = source.t()
    replay = ddp_runner().gradient_replay_copy(source)
    assert replay.data_ptr() % 256 == source.data_ptr() % 256
    assert replay.stride() == source.stride()
    assert replay.shape == source.shape
    assert replay.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(replay, source, atol=0, rtol=0)
