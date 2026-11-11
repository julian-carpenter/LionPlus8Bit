"""Exercise optimized defaults with real bnb CPU operations on hosted CI."""

import copy
import math

import pytest
import torch

from lionplus8bit import LionPlus8bit


def test_default_global_clipping_matches_equations():
    p = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    q = torch.nn.Parameter(torch.tensor([0.1]))
    optimizer = LionPlus8bit(
        [
            {"params": [p], "lr": 0.03, "weight_decay": 0.1},
            {"params": [q], "lr": 0.07, "weight_decay": 0},
        ],
        betas=(0.6, 0.8),
        max_grad_norm=2.5,
    )
    x = [p.detach().double().clone(), q.detach().double().clone()]
    momentum = [torch.zeros_like(v) for v in x]
    for values in [([3.0, -1.0], [4.0]), ([-10.0, 2.0], [-5.0]), ([0.0, 0.0], [0.0])]:
        norm = math.sqrt(sum(v * v for row in values for v in row))
        scale = min(1.0, 2.5 / norm) if norm else 1.0
        for i, (parameter, group, row) in enumerate(zip([p, q], optimizer.param_groups, values)):
            parameter.grad = torch.tensor(row)
            gradient = torch.tensor(row, dtype=torch.float64) * scale
            direction = (0.6 * momentum[i] + 0.4 * gradient).sign()
            x[i] = x[i] * (1 - group["lr"] * group["weight_decay"]) - group["lr"] * direction
            momentum[i] = 0.8 * momentum[i] + 0.2 * gradient
        original = [p.grad, q.grad]
        snapshots = [g.clone() for g in original]
        optimizer.step()
        for i, parameter in enumerate([p, q]):
            torch.testing.assert_close(parameter.double(), x[i], rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(
                optimizer.state[parameter]["state1"].double(), momentum[i], rtol=2e-6, atol=2e-7
            )
            assert parameter.grad is original[i]
            torch.testing.assert_close(parameter.grad, snapshots[i], rtol=0, atol=0)


def test_default_quantization_and_lazy_checkpoint_policy():
    params = [torch.nn.Parameter(torch.ones(8192)) for _ in range(2)]
    original = LionPlus8bit(params, min_8bit_size=8193)
    params[0].grad = torch.ones_like(params[0])
    original.step()
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = LionPlus8bit(restored_params)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    restored_params[1].grad = torch.ones_like(restored_params[1])
    restored.step()
    assert restored.state[restored_params[1]]["state1"].dtype == torch.float32
    assert restored.native_gradient_scale and restored.batched_norm
    large = torch.nn.Parameter(torch.ones(8192))
    small = torch.nn.Parameter(torch.ones(3))
    optimizer = LionPlus8bit([large, small])
    large.grad, small.grad = torch.ones_like(large), torch.ones_like(small)
    optimizer.step()
    assert optimizer.state[large]["state1"].dtype == torch.uint8
    assert optimizer.state[small]["state1"].dtype == torch.float32


def test_default_rejects_nonfinite_before_update():
    p = torch.nn.Parameter(torch.ones(4))
    optimizer = LionPlus8bit([p])
    p.grad = torch.full_like(p, float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(p, torch.ones_like(p), rtol=0, atol=0)


def test_default_extreme_clipping_ratio():
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = LionPlus8bit([p], lr=0.01, betas=(0, 0), weight_decay=0, max_grad_norm=1e-30)
    p.grad = torch.full_like(p, 1e20)
    optimizer.step()
    torch.testing.assert_close(
        optimizer.state[p]["state1"], torch.full_like(p, 1e-30 / math.sqrt(2)), rtol=1e-6, atol=0
    )


def test_default_rejects_parameter_layout_before_initialization():
    p = torch.nn.Parameter(torch.ones(3, 4).t())
    optimizer = LionPlus8bit([p])
    p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="contiguous"):
        optimizer.step()
    assert not optimizer.state
