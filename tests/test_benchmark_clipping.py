import importlib.util
from pathlib import Path

import torch


path = Path(__file__).resolve().parents[1] / "tools" / "benchmark_clipping.py"
spec = importlib.util.spec_from_file_location("benchmark_clipping", path)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_replay_recovery_is_one_tiny_trigger_without_gradient_drift():
    ps = [torch.nn.Parameter(torch.ones(8)) for _ in range(4)]
    replay = benchmark.ReplayInputs(ps, "recovery")
    originals = [g.clone() for g in replay.base]
    before = replay.prepare(99, 100, "none")
    assert before["condition"] == "ordinary"
    trigger = replay.prepare(100, 100, "none")
    assert trigger["condition"] == "tiny"
    assert ps[-1].grad[0] == torch.tensor(1e-30)
    after = replay.prepare(101, 100, "none")
    assert after["condition"] == "recovery"
    for p, original, base in zip(ps, originals, replay.base):
        torch.testing.assert_close(p.grad, original, rtol=0, atol=0)
        torch.testing.assert_close(base, original, rtol=0, atol=0)
    replay.prepare(115, 100, "widespread")
    replay.prepare(116, 100, "none")
    for p, original in zip(ps, originals):
        torch.testing.assert_close(p.grad, original, rtol=0, atol=0)


def test_replay_zero_strided_and_variable_active_cases():
    ps = [torch.nn.Parameter(torch.ones(8)) for _ in range(4)]
    replay = benchmark.ReplayInputs(ps, "zero")
    replay.prepare(100, 100, "none")
    assert torch.count_nonzero(ps[-1].grad) == 0
    assert torch.count_nonzero(ps[0].grad) > 0
    replay = benchmark.ReplayInputs(ps, "strided")
    replay.prepare(100, 100, "none")
    assert not ps[-1].grad.is_contiguous()
    assert all(p.grad.is_contiguous() for p in ps[:-1])
    replay = benchmark.ReplayInputs(ps, "variable-active")
    replay.prepare(100, 100, "none")
    first = [p.grad is not None for p in ps]
    replay.prepare(101, 100, "none")
    assert [p.grad is not None for p in ps] == [not active for active in first]


def test_conditional_summary_exposes_spikes_and_recovery_costs():
    samples = [
        dict(wall_ms=1.0, cuda_ms=0.8, spike=False, condition="ordinary", paths={}),
        dict(wall_ms=9.0, cuda_ms=8.0, spike=True, condition="tiny", paths={"strict_tensors": 1}),
        dict(
            wall_ms=3.0, cuda_ms=2.5, spike=False, condition="recovery", paths={"strict_tensors": 1}
        ),
    ]
    summary = benchmark.conditional_summary(samples)
    assert summary["spike"]["count"] == 1
    assert summary["spike"]["median_wall_ms"] == 9.0
    assert summary["nonspike"]["count"] == 2
    assert summary["condition:recovery"]["median_wall_ms"] == 3.0
    assert summary["path:strict"]["count"] == 2


def test_storage_bytes_counts_shared_native_codebook_only_once():
    codebook = torch.zeros(4, dtype=torch.float32)
    state = torch.zeros(3, dtype=torch.uint8)
    assert benchmark.storage_bytes([codebook, codebook[:2], codebook, state, None]) == 19


def test_storage_bytes_recurses_nested_cached_metadata():
    from collections import OrderedDict

    history = torch.zeros(4, dtype=torch.float32)
    indices = torch.zeros(3, dtype=torch.uint8)
    cache = OrderedDict([(0, (history[:2], indices)), (1, [history, None])])
    assert benchmark.storage_bytes([{"history": history, "cache": cache}]) == 19
