from unittest.mock import patch

import pytest
import torch

from lionplus8bit import _numerical_guard as guard


def parameter(values, device="cpu", dtype=torch.float32, strided=False):
    values = torch.tensor(values, device=device, dtype=dtype)
    p = torch.nn.Parameter(torch.zeros_like(values))
    if strided:
        backing = torch.empty((values.numel(), 2), device=device, dtype=dtype)
        backing[:, 0].copy_(values)
        p.grad = backing[:, 0]
    else:
        p.grad = values
    return p


def test_cpu_guard_returns_independent_tiny_and_nonzero_flags():
    ps = [
        parameter([1.0, -1.0]),
        parameter([1.0, 1.0]),
        parameter([1e-30, 1.0], strided=True),
        parameter([0.0, -0.0]),
        parameter([1e-6, 1.0]),
        parameter([]),
    ]
    states = {
        ps[1]: {"state1": torch.zeros(2, dtype=torch.uint8), "absmax1": torch.tensor([1e-14])}
    }
    limits = torch.tensor([1e-20, 1e-20, 1e-20, 1e-20, 1e-4, 1e-20])
    tiny, nonzero = guard.tiny_value_flag(ps, states, limits, return_nonzero=True)
    assert tiny.dtype == nonzero.dtype == torch.int32
    assert tiny.tolist() == [0, 1, 1, 0, 1, 0]
    assert nonzero.tolist() == [1, 1, 1, 0, 1, 0]
    assert guard.tiny_value_flag(ps, states, limits).tolist() == tiny.tolist()
    assert guard.has_tiny_values(ps, states, 1e-20)
    assert not guard.has_tiny_values([ps[0], ps[3]], states, 1e-20)


def test_nonzero_scan_distinguishes_subnormals_zeros_and_nonfinite_values():
    ps = [
        parameter([1e-40]),
        parameter([0.0]),
        parameter([float("inf")]),
        parameter([float("nan")]),
    ]
    tiny, nonzero = guard.tiny_value_flag(ps, {}, torch.full((4,), 1e-20), return_nonzero=True)
    assert tiny.tolist() == [1, 0, 0, 0]
    assert nonzero.tolist() == [1, 0, 1, 1]
    empty_tiny, empty_nonzero = guard.tiny_value_flag([], {}, torch.empty(0), return_nonzero=True)
    assert empty_tiny.shape == empty_nonzero.shape == (0,)
    assert not guard.has_tiny_values([], {}, 1e-20)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [255, 256, 257, 1025])
def test_cuda_device_limits_uint8_state_and_mixed_layout_match_eager(dtype, size):
    if guard.triton is None:
        pytest.skip("Triton is needed to compare the fused path")
    ps = [parameter([1.0] * size, "cuda", dtype, strided=i == 1) for i in range(4)]
    ps[0].grad[0] = 1e-4
    ps[1].grad.zero_()
    ps[2].grad[0] = 1e-4
    states = {
        ps[3]: {
            "state1": torch.zeros(size, device="cuda", dtype=torch.uint8),
            "absmax1": torch.full(((size + 255) // 256,), 1e-14, device="cuda"),
        }
    }
    limits = torch.tensor([1e-3, 1e-20, 1e-5, 1e-20], device="cuda")
    calls = []
    kernel = guard._guard

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(args[0])
                return kernel[grid](*args, **kwargs)

            return launch

    with patch.object(guard, "_guard", RecordingKernel()):
        actual = guard.tiny_value_flag(ps, states, limits, return_nonzero=True)
    with patch.object(guard, "triton", None):
        expected = guard.tiny_value_flag(ps, states, limits, return_nonzero=True)
    assert len(calls) == 3  # Only the strided tensor takes the eager path.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert actual[0].tolist() == [1, 0, 0, 1]
    assert actual[1].tolist() == [1, 0, 1, 1]
